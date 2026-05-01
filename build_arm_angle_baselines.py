#!/usr/bin/env python3
"""
build_arm_angle_baselines.py — Per-pitcher arm angle stats from Statcast (MLB + AAA).

Fetches arm_angle from Baseball Savant, computes:
  - arm_angle: pitcher's mean arm angle
  - arm_angle_std: pitch-to-pitch consistency
  - pfx_x_slot / pfx_z_slot: expected pfx_x/z for that arm slot

Uses rolling N-start window when available, falls back to season aggregate.

For AAA data: useful for callups — when a AAA pitcher debuts in MLB, they'll
already have arm angle stats indexed by MLBAMID.

Usage:
    cd ~/pitch-plus-api

    # MLB only (default)
    python3 build_arm_angle_baselines.py --year 2025

    # AAA only
    python3 build_arm_angle_baselines.py --year 2025 --level aaa

    # Both — merges into one file, MLB takes priority, AAA fills gaps
    python3 build_arm_angle_baselines.py --year 2025 --level both
"""
import argparse, json, time, gc, sys
from pathlib import Path
from datetime import date
from io import StringIO
import numpy as np
import pandas as pd
import requests

SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
CHUNK_DAYS = 14


def fetch_statcast(year, level="mlb", start_date=None, end_date=None):
    """Fetch Statcast with arm_angle. level: 'mlb' or 'aaa'."""
    if start_date is None:
        start_date = f"{year}-03-20"
    if end_date is None:
        today = date.today()
        end_date = min(f"{year}-11-01", today.strftime("%Y-%m-%d"))

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    print(f"  Fetching {level.upper()} Statcast: {start.date()} → {end.date()}")

    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=CHUNK_DAYS - 1), end)
        s = cur.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        print(f"    {s} to {e}...", end=" ", flush=True)

        params = {
            "all": "true", "hfPT": "", "hfAB": "", "hfGT": "R|", "hfPR": "",
            "hfZ": "", "stadium": "", "hfBBL": "", "hfNewZones": "", "hfPull": "",
            "hfC": "", "hfSea": str(year), "hfSit": "",
            "player_type": "pitcher", "hfOuts": "", "opponent": "",
            "pitcher_throws": "", "batter_stands": "", "hfSA": "",
            "game_date_gt": s, "game_date_lt": e,
            "hfMo": "", "team": "", "home_road": "", "hfRO": "",
            "position": "", "hfInfield": "", "hfOutfield": "", "hfInn": "",
            "hfBBT": "", "hfFlag": "", "metric_1": "", "group_by": "name",
            "min_pitches": "0", "min_results": "0", "min_pas": "0",
            "sort_col": "pitches", "player_event_sort": "api_p_release_speed",
            "sort_order": "desc", "chk_": "on", "type": "details",
            "csv": "true",
        }
        if level == "aaa":
            params["player_level_results"] = "aaa"

        try:
            r = requests.get(SAVANT_URL, params=params,
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            if r.status_code == 200 and len(r.text) > 200:
                df = pd.read_csv(StringIO(r.text))
                if len(df) > 0:
                    aa_count = df['arm_angle'].notna().sum() if 'arm_angle' in df.columns else 0
                    print(f"{len(df):,} pitches ({aa_count} with arm_angle)")
                    chunks.append(df)
                else:
                    print("0 pitches")
            else:
                print(f"status {r.status_code}")
        except Exception as ex:
            print(f"FAILED: {ex}")

        cur = chunk_end + pd.Timedelta(days=1)
        time.sleep(2)
        gc.collect()

    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Total: {len(df):,} {level.upper()} pitches")
    return df


def compute_baselines(df, rolling_starts=3, source_label="MLB"):
    """Compute per-pitcher arm angle stats."""
    if 'arm_angle' not in df.columns:
        print(f"  WARNING: no arm_angle column in {source_label} data")
        return {}

    df['arm_angle'] = pd.to_numeric(df['arm_angle'], errors='coerce')
    df['pfx_x'] = pd.to_numeric(df['pfx_x'], errors='coerce')
    df['pfx_z'] = pd.to_numeric(df['pfx_z'], errors='coerce')
    df['game_date'] = pd.to_datetime(df['game_date'])

    valid = df.dropna(subset=['arm_angle', 'pitcher'])
    print(f"  {len(valid):,} {source_label} pitches with valid arm_angle")

    out = {}
    for pid, group in valid.groupby('pitcher'):
        if len(group) < 10:
            continue
        group = group.sort_values('game_date')
        game_dates = group['game_date'].unique()
        if len(game_dates) >= rolling_starts:
            recent_dates = sorted(game_dates)[-rolling_starts:]
            recent = group[group['game_date'].isin(recent_dates)]
        else:
            recent = group

        aa_mean = float(recent['arm_angle'].mean())
        aa_std = float(recent['arm_angle'].std()) if len(recent) > 1 else 0.0
        pfx_x = float(recent['pfx_x'].mean()) if recent['pfx_x'].notna().sum() > 5 else 0.0
        pfx_z = float(recent['pfx_z'].mean()) if recent['pfx_z'].notna().sum() > 5 else 0.0

        out[str(int(pid))] = {
            'arm_angle': round(aa_mean, 2),
            'arm_angle_std': round(aa_std, 3),
            'pfx_x_slot': round(pfx_x, 4),
            'pfx_z_slot': round(pfx_z, 4),
            '_source': source_label,
            '_n_pitches': int(len(recent)),
            '_n_games': int(len(recent['game_date'].unique())),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--level", choices=["mlb", "aaa", "both"], default="mlb")
    p.add_argument("--mlb-parquet", default=None)
    p.add_argument("--aaa-parquet", default=None)
    p.add_argument("--save-mlb-parquet", default=None)
    p.add_argument("--save-aaa-parquet", default=None)
    p.add_argument("--rolling-starts", type=int, default=3)
    p.add_argument("--out", default="models/pitcher_arm_angles.json")
    args = p.parse_args()

    mlb_baselines = {}
    aaa_baselines = {}

    # If only doing AAA, preserve existing MLB entries
    if args.level == "aaa" and Path(args.out).exists():
        with open(args.out) as f:
            existing = json.load(f)
        mlb_baselines = {k: v for k, v in existing.items()
                         if v.get("_source", "MLB") == "MLB"}
        print(f"Loaded {len(mlb_baselines)} existing MLB baselines")

    # Fetch MLB
    if args.level in ("mlb", "both"):
        if args.mlb_parquet and Path(args.mlb_parquet).exists():
            print(f"Loading {args.mlb_parquet}...")
            df = pd.read_parquet(args.mlb_parquet) if args.mlb_parquet.endswith(".parquet") else pd.read_csv(args.mlb_parquet)
        else:
            df = fetch_statcast(args.year, level="mlb")
            if df is None:
                print("ERROR: No MLB data")
                sys.exit(1)
            if args.save_mlb_parquet:
                df.to_parquet(args.save_mlb_parquet, index=False)
        print(f"\nComputing MLB baselines...")
        mlb_baselines = compute_baselines(df, args.rolling_starts, "MLB")

    # Fetch AAA
    if args.level in ("aaa", "both"):
        if args.aaa_parquet and Path(args.aaa_parquet).exists():
            print(f"Loading {args.aaa_parquet}...")
            df = pd.read_parquet(args.aaa_parquet) if args.aaa_parquet.endswith(".parquet") else pd.read_csv(args.aaa_parquet)
        else:
            df = fetch_statcast(args.year, level="aaa")
        if df is not None:
            if args.save_aaa_parquet:
                df.to_parquet(args.save_aaa_parquet, index=False)
            print(f"\nComputing AAA baselines...")
            aaa_baselines = compute_baselines(df, args.rolling_starts, "AAA")
        else:
            print("WARNING: No AAA data")

    # Merge: AAA fills gaps where MLB has no data
    merged = dict(aaa_baselines)
    merged.update(mlb_baselines)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(merged, f, separators=(',', ':'))

    n_mlb = sum(1 for v in merged.values() if v.get("_source") == "MLB")
    n_aaa = sum(1 for v in merged.values() if v.get("_source") == "AAA")
    print(f"\n{'='*55}")
    print(f"  Wrote {len(merged)} baselines to {out_path}")
    print(f"    MLB-sourced: {n_mlb}")
    print(f"    AAA-sourced (callup-ready): {n_aaa}")
    print(f"{'='*55}")
    if merged:
        mlb_s = next((k for k, v in merged.items() if v.get("_source") == "MLB"), None)
        aaa_s = next((k for k, v in merged.items() if v.get("_source") == "AAA"), None)
        if mlb_s: print(f"  MLB sample {mlb_s}: {merged[mlb_s]}")
        if aaa_s: print(f"  AAA sample {aaa_s}: {merged[aaa_s]}")


if __name__ == "__main__":
    main()
