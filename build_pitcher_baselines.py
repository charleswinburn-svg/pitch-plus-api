#!/usr/bin/env python3
"""
build_pitcher_baselines.py — Compute trailing-60-day fastball baselines per pitcher.

For each pitcher, takes the last 60 days of their fastball pitches (FF/SI/FC,
most-used type) and averages the features. Pitchers with <10 fastballs in the
window fall back to league average as a cold-start value.

Usage:
    python3 build_pitcher_baselines.py --parquet pitch_xrv_2026.parquet --asof 2026-04-12
    # Or build from multiple years combined:
    python3 build_pitcher_baselines.py --parquet pitch_xrv_2025.parquet pitch_xrv_2026.parquet

Writes: models/pitcher_baselines.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


WINDOW_DAYS = 60
FB_TYPES = {'FF', 'SI', 'FC'}
MIN_PITCHES = 10


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', nargs='+', required=True, help='One or more parquets')
    p.add_argument('--asof', default=None, help='As-of date YYYY-MM-DD (default: today)')
    p.add_argument('--out', default='models/pitcher_baselines.json')
    p.add_argument('--window-days', type=int, default=WINDOW_DAYS)
    args = p.parse_args()

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.now().normalize()
    window_start = asof - pd.Timedelta(days=args.window_days)
    print(f"Window: {window_start.date()} → {asof.date()} ({args.window_days} days)")

    # Load + concat
    dfs = []
    for p_path in args.parquet:
        print(f"Loading {p_path}...")
        d = pd.read_parquet(p_path)
        if 'game_date' in d.columns:
            d['game_date'] = pd.to_datetime(d['game_date'])
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    # Filter to window
    if 'game_date' not in df.columns:
        print("ERROR: no game_date column in parquets. Can't compute trailing window.")
        return
    df = df[(df['game_date'] >= window_start) & (df['game_date'] <= asof)]
    print(f"  {len(df):,} pitches in window")

    # Cast
    for c in ['release_speed','pfx_x','pfx_z','release_spin_rate','release_extension',
              'vx0','vy0','vz0','ax','ay','az','release_pos_x','release_pos_z','plate_x','plate_z']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['release_speed','vy0','vz0','ay','az'])

    # Fastballs only
    fb = df[df['pitch_type'].isin(FB_TYPES)].copy()
    fb['vaa'] = compute_vaa(fb)
    tx, tz = compute_tunnel(fb)
    fb['tunnel_x'] = tx
    fb['tunnel_z'] = tz

    # League averages (cold-start fallback)
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

    # Per-pitcher
    out = {}
    for pid, group in fb.groupby('pitcher'):
        counts = group['pitch_type'].value_counts()
        primary = counts.idxmax()
        sub = group[group['pitch_type'] == primary]
        if len(sub) < MIN_PITCHES:
            # Cold start: use league FF (or whatever primary)
            lg = league.get(primary) or league.get('FF')
            if not lg: continue
            out[str(int(pid))] = {
                'fb_type': primary, '_cold_start': True,
                'fb_velo': lg['velo'], 'fb_pfx_x': lg['pfx_x'], 'fb_pfx_z': lg['pfx_z'],
                'fb_spin': lg['spin'], 'fb_extension': lg['ext'], 'fb_vaa': lg['vaa'],
                'fb_release_x': lg['rx'], 'fb_release_z': lg['rz'],
                'fb_tunnel_x': lg['tun_x'], 'fb_tunnel_z': lg['tun_z'],
                'fb_plate_x': lg['plate_x'], 'fb_plate_z': lg['plate_z'],
            }
        else:
            out[str(int(pid))] = {
                'fb_type': primary, '_cold_start': False, '_n': int(len(sub)),
                'fb_velo': float(sub['release_speed'].mean()),
                'fb_pfx_x': float(sub['pfx_x'].mean()),
                'fb_pfx_z': float(sub['pfx_z'].mean()),
                'fb_spin': float(sub['release_spin_rate'].mean()),
                'fb_extension': float(sub['release_extension'].mean()),
                'fb_vaa': float(sub['vaa'].mean()),
                'fb_release_x': float(sub['release_pos_x'].mean()),
                'fb_release_z': float(sub['release_pos_z'].mean()),
                'fb_tunnel_x': float(np.nanmean(sub['tunnel_x'])),
                'fb_tunnel_z': float(np.nanmean(sub['tunnel_z'])),
                'fb_plate_x': float(sub['plate_x'].mean()),
                'fb_plate_z': float(sub['plate_z'].mean()),
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))

    warm = sum(1 for v in out.values() if not v.get('_cold_start'))
    cold = len(out) - warm
    print(f"\nWrote {len(out)} baselines to {out_path}")
    print(f"  {warm} warm ({args.window_days}-day avg), {cold} cold-start (league fallback)")


if __name__ == '__main__':
    main()
