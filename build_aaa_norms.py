#!/usr/bin/env python3
"""
build_aaa_norms.py — Build Pitch+ norms (Stuff+/Loc+/Tunnel+/Pitch+) for AAA.

Fetches AAA Statcast data from Baseball Savant, scores through the same
LightGBM models as MLB, then computes AAA-specific mean/std per pitch type
for each component. Writes models/pitch_plus_norm_aaa.json.

Usage:
    cd ~/pitch-plus-api
    python3 build_aaa_norms.py --year 2025
    python3 build_aaa_norms.py --year 2026
    python3 build_aaa_norms.py --parquet aaa_statcast_2025.parquet  # if you already have data
"""
import argparse, json, sys, time, gc
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import requests


SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
CHUNK_DAYS = 14


def fetch_aaa_statcast(year, start_date=None, end_date=None):
    """Fetch AAA Statcast data from Baseball Savant in chunks."""
    if start_date is None:
        start_date = f"{year}-04-01"
    if end_date is None:
        today = date.today()
        end_date = min(f"{year}-09-30", today.strftime("%Y-%m-%d"))

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    print(f"  Fetching AAA Statcast: {start.date()} → {end.date()}")

    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=CHUNK_DAYS - 1), end)
        s = cur.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        print(f"    {s} to {e}...", end=" ", flush=True)

        params = {
            "all": "true",
            "hfPT": "",
            "hfAB": "",
            "hfGT": "R|",  # regular season
            "hfPR": "",
            "hfZ": "",
            "stadium": "",
            "hfBBL": "",
            "hfNewZones": "",
            "hfPull": "",
            "hfC": "",
            "hfSea": str(year),
            "hfSit": "",
            "player_type": "pitcher",
            "hfOuts": "",
            "opponent": "",
            "pitcher_throws": "",
            "batter_stands": "",
            "hfSA": "",
            "game_date_gt": s,
            "game_date_lt": e,
            "hfMo": "",
            "team": "",
            "home_road": "",
            "hfRO": "",
            "position": "",
            "hfInfield": "",
            "hfOutfield": "",
            "hfInn": "",
            "hfBBT": "",
            "hfFlag": "",
            "metric_1": "",
            "group_by": "name",
            "min_pitches": "0",
            "min_results": "0",
            "min_pas": "0",
            "sort_col": "pitches",
            "player_event_sort": "api_p_release_speed",
            "sort_order": "desc",
            "chk_": "on",
            "type": "details",
            "player_level_results": "aaa",  # AAA filter
            "csv": "true",
        }

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(SAVANT_URL, params=params, headers=headers, timeout=60)
            if r.status_code == 200 and len(r.text) > 100:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                if len(df) > 0:
                    print(f"{len(df):,} pitches")
                    chunks.append(df)
                else:
                    print("0 pitches")
            else:
                print(f"status {r.status_code}")
        except Exception as ex:
            print(f"FAILED: {ex}")

        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(2)  # rate limit
        gc.collect()

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Total: {len(df):,} AAA pitches")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--parquet", default=None, help="Use existing parquet instead of fetching")
    p.add_argument("--save-parquet", default=None, help="Save fetched data to parquet")
    p.add_argument("--models", default="models")
    p.add_argument("--config", default="models/final_model_config.json")
    p.add_argument("--out", default="models/pitch_plus_norm_aaa.json")
    args = p.parse_args()

    sys.path.insert(0, ".")
    from score_pitches import load_models, score_dataframe, _to_float64

    # Get data
    if args.parquet and Path(args.parquet).exists():
        print(f"Loading {args.parquet}...")
        df = pd.read_parquet(args.parquet)
    else:
        print(f"Fetching AAA Statcast for {args.year}...")
        df = fetch_aaa_statcast(args.year)
        if df is None or len(df) == 0:
            print("ERROR: No AAA data fetched")
            sys.exit(1)
        if args.save_parquet:
            df.to_parquet(args.save_parquet, index=False)
            print(f"  Saved to {args.save_parquet}")

    # Check required columns
    needed = ['release_speed', 'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0',
              'ax', 'ay', 'az', 'release_pos_x', 'release_pos_z',
              'plate_x', 'plate_z', 'pitch_type', 'pitcher']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"  Available: {sorted(df.columns.tolist())}")
        sys.exit(1)

    # Add season column if missing
    if 'season' not in df.columns:
        if 'game_date' in df.columns:
            df['season'] = pd.to_datetime(df['game_date']).dt.year
        else:
            df['season'] = args.year

    df = df.dropna(subset=needed)
    df = _to_float64(df)
    print(f"  {len(df):,} pitches after filtering")
    print(f"  Pitch types: {df['pitch_type'].value_counts().head(10).to_dict()}")

    # Score through MLB models
    print("Scoring through Pitch+ models...")
    stuff, sf, tunnel, loc = load_models(Path(args.models))
    with open(args.config) as f:
        config = json.load(f)
    df = score_dataframe(df, stuff, sf, tunnel, loc, config)

    # Build game-level aggregates (matches how React displays)
    if 'game_pk' not in df.columns:
        if 'game_date' in df.columns:
            df['game_pk'] = df['game_date'].astype(str) + '_' + df['pitcher'].astype(str)
        else:
            df['game_pk'] = df.index // 100  # rough grouping

    game_agg = df.groupby(['pitcher', 'game_pk', 'pitch_type']).agg(
        n=('xRV_final', 'size'),
        stuff=('xRV_stuff', 'mean'),
        loc=('xRV_location', 'mean'),
        tun=('xRV_tunnel', 'mean'),
        final=('xRV_final', 'mean'),
    ).reset_index()
    game_agg = game_agg[game_agg['n'] >= 5]  # lower threshold for AAA (fewer pitches per game)

    # Build norms
    norm = {}
    for pt, sub in game_agg.groupby('pitch_type'):
        if len(sub) < 50:  # lower threshold for AAA
            continue
        norm[pt] = {
            'mean': float(sub['final'].median()),
            'std': float(sub['final'].std()),
            'stuff_mean': float(sub['stuff'].median()),
            'stuff_std': float(sub['stuff'].std()),
            'loc_mean': float(sub['loc'].median()),
            'loc_std': float(sub['loc'].std()),
            'tun_mean': float(sub['tun'].median()),
            'tun_std': float(sub['tun'].std()),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(norm, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  AAA PITCH+ NORMS — {args.year}")
    print(f"{'='*70}")
    print(f"  {'PT':<4} {'Pitch+ std':>10} {'Stuff std':>10} {'Loc std':>10} {'Tun std':>10}  {'N':>6}")
    print(f"  {'─'*56}")
    for pt in sorted(norm):
        n = norm[pt]
        ct = len(game_agg[game_agg['pitch_type'] == pt])
        print(f"  {pt:<4} {n['std']:>10.5f} {n['stuff_std']:>10.5f} {n['loc_std']:>10.5f} {n['tun_std']:>10.5f}  {ct:>6}")

    # Compare to MLB norms
    mlb_norm_path = Path(args.models) / "pitch_plus_norm.json"
    if mlb_norm_path.exists():
        mlb = json.load(open(mlb_norm_path))
        print(f"\n  COMPARISON: AAA vs MLB (Pitch+ std):")
        print(f"  {'PT':<4} {'AAA':>8} {'MLB':>8} {'Ratio':>8}")
        print(f"  {'─'*30}")
        for pt in sorted(set(norm) & set(mlb)):
            a = norm[pt]['std']
            m = mlb[pt]['std']
            print(f"  {pt:<4} {a:>8.5f} {m:>8.5f} {a/m if m > 0 else 0:>8.2f}x")

    print(f"\nWrote {len(norm)} pitch types to {out_path}")
    print()


if __name__ == "__main__":
    main()
