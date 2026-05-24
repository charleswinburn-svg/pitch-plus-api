#!/usr/bin/env python3
"""
build_pitcher_baselines.py - Compute trailing-window fastball baselines per pitcher.

Data source priority (highest wins):
  1. Current-season Savant pull (--current-year): rehab returns + debuts
  2. Fallback-year Savant pulls (--fallback-years): for pitchers absent from
     current season (e.g. injury returners whose last action was a prior year)
  3. xRV parquet (--parquet): season-stable veterans + historical baseline
  4. League average cold-start: pitchers with no usable data anywhere

This fixes the "injury returner has no baseline" problem. Cole 2026 didn't pitch
2025 (Tommy John); without --fallback-years 2024 his record would be missing.
With it, his 2024 baseline carries over until he accumulates enough 2026 data.

Usage:
    python3 build_pitcher_baselines.py --parquet pitch_xrv_2025.parquet \\
        --current-year 2026 --fallback-years 2024

Writes: models/pitcher_baselines.json
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


WINDOW_DAYS = 60
FB_TYPES = {'FF', 'SI', 'FC'}
MIN_PITCHES = 10
MIN_PITCHES_RECENT = 30


def compute_vaa(df):
    t = np.clip(50.0 / (-df['vy0'].values.astype(float)), 0.35, 0.55)
    vaa = np.degrees(np.arctan2(
        df['vz0'].values.astype(float) + df['az'].values.astype(float) * t,
        -(df['vy0'].values.astype(float) + df['ay'].values.astype(float) * t)))
    return vaa


def compute_tunnel(df):
    PLATE_Y = 17.0 / 12.0
    y_tun = PLATE_Y + 23.0
    y0 = 50.0
    a_ = 0.5 * df['ay'].values.astype(float)
    b_ = df['vy0'].values.astype(float)
    c_ = y0 - y_tun
    with np.errstate(invalid='ignore', divide='ignore'):
        disc = b_**2 - 4 * a_ * c_
        t_tun = np.where(disc >= 0,
                         (-b_ - np.sqrt(np.maximum(disc, 0))) / (2 * a_), np.nan)
        t_tun = np.where((t_tun > 0) & (t_tun < 0.5), t_tun, np.nan)
    tx = df['release_pos_x'].values.astype(float) + df['vx0'].values.astype(float) * t_tun + 0.5 * df['ax'].values.astype(float) * t_tun**2
    tz = df['release_pos_z'].values.astype(float) + df['vz0'].values.astype(float) * t_tun + 0.5 * df['az'].values.astype(float) * t_tun**2
    return tx, tz


def cast_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def prepare_fastballs(df, label=""):
    if df is None or len(df) == 0:
        if label: print(f"  {label}: 0 fastball pitches")
        return pd.DataFrame()
    numeric_cols = ['release_speed','pfx_x','pfx_z','release_spin_rate','release_extension',
                    'vx0','vy0','vz0','ax','ay','az','release_pos_x','release_pos_z',
                    'plate_x','plate_z']
    df = cast_numeric(df, numeric_cols)
    df = df.dropna(subset=['release_speed','vy0','vz0','ay','az'])
    fb = df[df['pitch_type'].isin(FB_TYPES)].copy()
    if len(fb) == 0:
        if label: print(f"  {label}: 0 fastball pitches")
        return fb
    fb['vaa'] = compute_vaa(fb)
    tx, tz = compute_tunnel(fb)
    fb['tunnel_x'] = tx
    fb['tunnel_z'] = tz
    if label: print(f"  {label}: {len(fb):,} fastball pitches")
    return fb


def baseline_record(sub, primary, cold=False, source=None):
    rec = {
        'fb_type': primary, '_cold_start': cold, '_n': int(len(sub)),
        'fb_velo':      float(sub['release_speed'].mean()),
        'fb_pfx_x':     float(sub['pfx_x'].mean()),
        'fb_pfx_z':     float(sub['pfx_z'].mean()),
        'fb_spin':      float(sub['release_spin_rate'].mean()),
        'fb_extension': float(sub['release_extension'].mean()),
        'fb_vaa':       float(sub['vaa'].mean()),
        'fb_release_x': float(sub['release_pos_x'].mean()),
        'fb_release_z': float(sub['release_pos_z'].mean()),
        'fb_tunnel_x':  float(np.nanmean(sub['tunnel_x'])),
        'fb_tunnel_z':  float(np.nanmean(sub['tunnel_z'])),
        'fb_plate_x':   float(sub['plate_x'].mean()),
        'fb_plate_z':   float(sub['plate_z'].mean()),
    }
    if source: rec['_source'] = source
    return rec


def cold_start_record(league, primary='FF'):
    lg = league.get(primary) or league.get('FF')
    if not lg: return None
    return {
        'fb_type': primary, '_cold_start': True, '_source': 'league',
        'fb_velo': lg['velo'], 'fb_pfx_x': lg['pfx_x'], 'fb_pfx_z': lg['pfx_z'],
        'fb_spin': lg['spin'], 'fb_extension': lg['ext'], 'fb_vaa': lg['vaa'],
        'fb_release_x': lg['rx'], 'fb_release_z': lg['rz'],
        'fb_tunnel_x': lg['tun_x'], 'fb_tunnel_z': lg['tun_z'],
        'fb_plate_x': lg['plate_x'], 'fb_plate_z': lg['plate_z'],
    }


def build_league_avgs(fb):
    league = {}
    for pt in FB_TYPES:
        sub = fb[fb['pitch_type'] == pt]
        if len(sub) >= 100:
            league[pt] = {
                'velo': float(sub['release_speed'].mean()),
                'pfx_x': float(sub['pfx_x'].mean()),
                'pfx_z': float(sub['pfx_z'].mean()),
                'spin': float(sub['release_spin_rate'].mean()),
                'ext': float(sub['release_extension'].mean()),
                'vaa': float(sub['vaa'].mean()),
                'rx': float(sub['release_pos_x'].mean()),
                'rz': float(sub['release_pos_z'].mean()),
                'tun_x': float(np.nanmean(sub['tunnel_x'])),
                'tun_z': float(np.nanmean(sub['tunnel_z'])),
                'plate_x': float(sub['plate_x'].mean()),
                'plate_z': float(sub['plate_z'].mean()),
            }
    return league


def add_records(out, fb, source_tag, min_pitches, overwrite_warm):
    """overwrite_warm: True for current-season (replaces), False for fallback (fills gaps only)."""
    n_new, n_overrode = 0, 0
    if len(fb) == 0: return n_new, n_overrode
    for pid, group in fb.groupby('pitcher'):
        key = str(int(pid))
        counts = group['pitch_type'].value_counts()
        primary = counts.idxmax()
        sub = group[group['pitch_type'] == primary]
        if len(sub) < min_pitches: continue
        existing = out.get(key)
        is_warm = existing is not None and not existing.get('_cold_start')
        if is_warm and not overwrite_warm: continue
        if existing is None: n_new += 1
        elif is_warm: n_overrode += 1
        out[key] = baseline_record(sub, primary, cold=False, source=source_tag)
    return n_new, n_overrode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', nargs='+', required=True)
    p.add_argument('--current-year', type=int, default=None,
                   help='Current season to pull from Savant. Overrides all other sources.')
    p.add_argument('--fallback-years', type=int, nargs='+', default=[],
                   help='Earlier seasons to pull as fallback (e.g. 2024 for TJ returners). '
                        'Walked newest-first - newer overrides older - but all are '
                        'overridden by --current-year if a pitcher has enough current data.')
    p.add_argument('--asof', default=None)
    p.add_argument('--out', default='models/pitcher_baselines.json')
    p.add_argument('--window-days', type=int, default=WINDOW_DAYS)
    p.add_argument('--min-recent', type=int, default=MIN_PITCHES_RECENT)
    args = p.parse_args()

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.now().normalize()
    window_start = asof - pd.Timedelta(days=args.window_days)
    print(f"Window: {window_start.date()} -> {asof.date()} ({args.window_days} days)")

    # Load parquets
    dfs = []
    for p_path in args.parquet:
        print(f"Loading {p_path}...")
        d = pd.read_parquet(p_path)
        if 'game_date' in d.columns:
            d['game_date'] = pd.to_datetime(d['game_date'])
        dfs.append(d)
    parquet_df = pd.concat(dfs, ignore_index=True)
    print(f"  {len(parquet_df):,} pitches from parquet(s)")

    if 'game_date' not in parquet_df.columns:
        print("  WARNING: no game_date column - using all parquet data, no window")
    else:
        parquet_df = parquet_df[(parquet_df['game_date'] >= window_start) &
                                (parquet_df['game_date'] <= asof)]
        print(f"  {len(parquet_df):,} pitches in window")

    parquet_fb = prepare_fastballs(parquet_df, label="Parquet fastballs")

    # Lazy import for Savant
    fetch_statcast = None
    if args.current_year is not None or args.fallback_years:
        sys.path.insert(0, '.')
        try:
            from build_arm_angle_baselines import fetch_statcast
        except ImportError as e:
            print(f"ERROR: cannot import fetch_statcast: {e}")
            return

    # Current-season Savant
    savant_current_fb = pd.DataFrame()
    if args.current_year is not None:
        print(f"Pulling current-season Savant data for {args.current_year}...")
        try:
            cdf = fetch_statcast(args.current_year, level='mlb')
            if cdf is not None and 'game_date' in cdf.columns:
                cdf['game_date'] = pd.to_datetime(cdf['game_date'])
                cdf = cdf[(cdf['game_date'] >= window_start) & (cdf['game_date'] <= asof)]
            savant_current_fb = prepare_fastballs(cdf, label=f"Savant {args.current_year} fastballs")
        except Exception as e:
            print(f"  WARNING: Savant {args.current_year} pull failed ({type(e).__name__}: {e})")

    # Fallback-year pulls (we'll layer them oldest-to-newest, so newer overrides older)
    fallback_pulls = []
    for fy in sorted(args.fallback_years):  # ascending - oldest first
        print(f"Pulling fallback-year Savant data for {fy}...")
        try:
            fdf = fetch_statcast(fy, level='mlb')
            ffb = prepare_fastballs(fdf, label=f"Savant {fy} fastballs")
            fallback_pulls.append((fy, ffb))
        except Exception as e:
            print(f"  WARNING: Savant {fy} pull failed ({type(e).__name__}: {e})")

    # League averages (prefer parquet for stability)
    league_source = parquet_fb if len(parquet_fb) > 0 else savant_current_fb
    if len(league_source) == 0 and fallback_pulls:
        league_source = fallback_pulls[-1][1]  # newest fallback
    league = build_league_avgs(league_source)
    if not league:
        print("ERROR: no pitch type had >= 100 samples for league averages")
        return
    print(f"League avgs built for: {sorted(league.keys())}")

    # Build records by layering from LOWEST priority to HIGHEST.
    # Each layer overwrites the previous if it has data for that pitcher.
    out = {}

    # Lowest: parquet (broad coverage)
    n_new, _ = add_records(out, parquet_fb, 'parquet', MIN_PITCHES, overwrite_warm=True)
    print(f"Parquet layer: {n_new} pitchers")

    # Middle: fallback years, oldest first so newer fallback overrides older
    for fy, ffb in fallback_pulls:
        n_new_f, n_over_f = add_records(out, ffb, f'savant_{fy}',
                                        args.min_recent, overwrite_warm=True)
        print(f"Savant {fy}: {n_new_f} new, {n_over_f} overrode older source")

    # Highest: current-season Savant always wins for active pitchers
    if len(savant_current_fb) > 0:
        n_new_c, n_over_c = add_records(out, savant_current_fb, 'savant_current',
                                        args.min_recent, overwrite_warm=True)
        print(f"Savant {args.current_year} (current): {n_new_c} new, {n_over_c} overrode")

    # Write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))

    # Summary
    by_source = {}
    cold = 0
    for v in out.values():
        if v.get('_cold_start'):
            cold += 1
        else:
            s = v.get('_source', '?')
            by_source[s] = by_source.get(s, 0) + 1
    print(f"\nWrote {len(out)} baselines to {out_path}")
    print(f"  Cold start: {cold}")
    for s, n in sorted(by_source.items()):
        print(f"  {s}: {n}")


if __name__ == '__main__':
    main()
