#!/usr/bin/env python3
"""
build_component_norms.py — Build Stuff+/Loc+/Tun+/Pitch+ norms.
Chunked: processes parquet in slices to avoid OOM.

Usage:
    cd ~/pitch-plus-api
    python3 build_component_norms.py --parquet pitch_xrv_2025.parquet
    python3 build_component_norms.py --parquet pitch_xrv_2025.parquet --chunk-rows 30000
"""
import argparse, json, sys, gc
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="pitch_xrv_2025.parquet")
    p.add_argument("--models", default="models")
    p.add_argument("--config", default="models/final_model_config.json")
    p.add_argument("--norm", default="models/pitch_plus_norm.json")
    p.add_argument("--chunk-rows", type=int, default=50_000)
    args = p.parse_args()

    sys.path.insert(0, ".")
    from score_pitches import load_models, score_dataframe, _to_float64

    print(f"Loading {args.parquet}...")
    df_full = pd.read_parquet(args.parquet)
    n_total = len(df_full)
    print(f"  {n_total:,} pitches loaded")

    stuff, sf, tunnel, loc = load_models(Path(args.models))
    with open(args.config) as f:
        config = json.load(f)

    print(f"Scoring in chunks of {args.chunk_rows:,}...")
    agg_chunks = []
    for start in range(0, n_total, args.chunk_rows):
        end = min(start + args.chunk_rows, n_total)
        chunk = df_full.iloc[start:end].copy()
        chunk = _to_float64(chunk)
        chunk = score_dataframe(chunk, stuff, sf, tunnel, loc, config)

        sub_agg = chunk.groupby(['pitcher', 'game_pk', 'pitch_type'], dropna=False).agg(
            n=('xRV_final', 'size'),
            stuff=('xRV_stuff', 'mean'),
            loc=('xRV_location', 'mean'),
            tun=('xRV_tunnel', 'mean'),
            final=('xRV_final', 'mean'),
        ).reset_index()
        agg_chunks.append(sub_agg)

        del chunk
        gc.collect()
        print(f"  Scored {end:,}/{n_total:,} ({100*end/n_total:.0f}%)")

    del df_full
    gc.collect()

    print("Re-aggregating across chunks...")
    all_agg = pd.concat(agg_chunks, ignore_index=True)
    del agg_chunks; gc.collect()

    all_agg['stuff_sum'] = all_agg['stuff'] * all_agg['n']
    all_agg['loc_sum'] = all_agg['loc'] * all_agg['n']
    all_agg['tun_sum'] = all_agg['tun'] * all_agg['n']
    all_agg['final_sum'] = all_agg['final'] * all_agg['n']
    final_agg = all_agg.groupby(['pitcher', 'game_pk', 'pitch_type'], dropna=False).agg(
        n=('n', 'sum'),
        stuff_sum=('stuff_sum', 'sum'),
        loc_sum=('loc_sum', 'sum'),
        tun_sum=('tun_sum', 'sum'),
        final_sum=('final_sum', 'sum'),
    ).reset_index()
    final_agg['stuff'] = final_agg['stuff_sum'] / final_agg['n']
    final_agg['loc'] = final_agg['loc_sum'] / final_agg['n']
    final_agg['tun'] = final_agg['tun_sum'] / final_agg['n']
    final_agg['final'] = final_agg['final_sum'] / final_agg['n']

    game_agg = final_agg[final_agg['n'] >= 5].copy()
    print(f"  {len(game_agg):,} game-level samples (n>=5 per type)")

    norm = json.load(open(args.norm))
    for pt, sub in game_agg.groupby('pitch_type'):
        if pt not in norm or len(sub) < 100:
            continue
        norm[pt]['mean'] = float(sub['final'].median())
        norm[pt]['std'] = float(sub['final'].std())
        norm[pt]['stuff_mean'] = float(sub['stuff'].median())
        norm[pt]['stuff_std'] = float(sub['stuff'].std())
        norm[pt]['loc_mean'] = float(sub['loc'].median())
        norm[pt]['loc_std'] = float(sub['loc'].std())
        norm[pt]['tun_mean'] = float(sub['tun'].median())
        norm[pt]['tun_std'] = float(sub['tun'].std())

    with open(args.norm, "w") as f:
        json.dump(norm, f, indent=2)

    print(f"\nUpdated {args.norm}")
    print(f"  {'PT':<4} {'Pitch+ std':>10} {'Stuff std':>10} {'Loc std':>10} {'Tun std':>10}")
    for pt in sorted(norm):
        if 'stuff_std' in norm[pt]:
            n = norm[pt]
            print(f"  {pt:<4} {n['std']:>10.5f} {n['stuff_std']:>10.5f} {n['loc_std']:>10.5f} {n['tun_std']:>10.5f}")


if __name__ == "__main__":
    main()
