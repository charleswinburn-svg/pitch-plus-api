#!/usr/bin/env python3
"""
build_component_norms.py — Build Stuff+/Loc+/Tun+/Pitch+ norms.

Computes PITCH-LEVEL mean and std per pitch type (and globally) via streaming
sums to avoid OOM on large parquets. Each component (Stuff/Loc/Tun/Pitch+) can
independently use one of three scoping modes:

  per-type    each pitch type centered at 100 — hides cross-type quality differences
  global      one mean/std across all pitches — sweepers naturally rate higher
              than fastballs in absolute xRV terms
  hybrid      global mean (cross-type comparable) + per-type std (within-type
              spread preserved). Recommended for Stuff+/Tun+/Pitch+: sweepers
              correctly rate higher in absolute terms while within-type ranking
              keeps its natural spread.

Why pitch-level (not game-aggregate):
    The z-score formula at scoring time is (pitch_xRV - mean) / std applied to
    a single pitch. So mean and std must be calibrated on the same unit —
    individual pitches — not on game-level aggregates. Game aggregates artificially
    shrink the std (averaging cancels noise), which makes elite individual pitches
    look like extreme outliers and grades collapse.

Usage:
    cd ~/pitch-plus-api
    python3 build_component_norms.py --parquet pitch_xrv_2025.parquet
    # default modes: stuff/tun/pitch=hybrid, loc=per-type

    # all per-type:
    python3 build_component_norms.py --parquet pitch_xrv_2025.parquet \\
        --stuff-mode per-type --tun-mode per-type --pitch-mode per-type

    # all global:
    python3 build_component_norms.py --parquet pitch_xrv_2025.parquet \\
        --stuff-mode global --tun-mode global --pitch-mode global --loc-mode global
"""
import argparse, json, sys, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


COMPONENTS = [
    ('final', 'xRV_final'),    # Pitch+ overall
    ('stuff', 'xRV_stuff'),
    ('loc',   'xRV_location'),
    ('tun',   'xRV_tunnel'),
]


class StreamingStats:
    """Accumulate count, sum, sum-of-squares per pitch type for each component.
    mean = sum / n
    var  = (sumsq - sum*sum/n) / (n - 1)    [sample variance, matches pandas .std()]
    """
    def __init__(self):
        self.n      = defaultdict(lambda: defaultdict(int))      # [pt][comp] -> count
        self.sum    = defaultdict(lambda: defaultdict(float))    # [pt][comp] -> sum
        self.sumsq  = defaultdict(lambda: defaultdict(float))    # [pt][comp] -> sum of squares
        self.n_global    = defaultdict(int)
        self.sum_global  = defaultdict(float)
        self.sumsq_global = defaultdict(float)

    def update(self, df):
        """Update sums from a chunk DataFrame containing pitch_type + component cols."""
        for pt, sub in df.groupby('pitch_type', dropna=False):
            for comp, col in COMPONENTS:
                vals = sub[col].dropna().values.astype('float64')
                if len(vals) == 0:
                    continue
                self.n[pt][comp]     += len(vals)
                self.sum[pt][comp]   += vals.sum()
                self.sumsq[pt][comp] += (vals * vals).sum()
        for comp, col in COMPONENTS:
            vals = df[col].dropna().values.astype('float64')
            if len(vals) == 0:
                continue
            self.n_global[comp]     += len(vals)
            self.sum_global[comp]   += vals.sum()
            self.sumsq_global[comp] += (vals * vals).sum()

    @staticmethod
    def _mean_std(n, s, ss):
        if n < 2:
            return float('nan'), float('nan')
        mean = s / n
        # sample variance (ddof=1) — matches pandas Series.std() default
        var = (ss - s * s / n) / (n - 1)
        var = max(var, 0.0)
        return float(mean), float(var ** 0.5)

    def per_type(self, pt, comp):
        return self._mean_std(self.n[pt][comp], self.sum[pt][comp], self.sumsq[pt][comp])

    def global_(self, comp):
        return self._mean_std(self.n_global[comp], self.sum_global[comp], self.sumsq_global[comp])

    def n_for(self, pt):
        return max(self.n[pt][c] for c, _ in COMPONENTS)


def resolve_mode(mode, global_mean, global_std, type_mean, type_std):
    """Pick which (mean, std) to use for a (pitch_type, component) cell."""
    if mode == 'global':
        return global_mean, global_std
    if mode == 'hybrid':
        # cross-type comparable mean (sweepers rate higher than fastballs in absolute terms)
        # but within-type spread is preserved
        return global_mean, type_std
    # per-type
    return type_mean, type_std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="pitch_xrv_2025.parquet")
    p.add_argument("--models",  default="models")
    p.add_argument("--config",  default="models/final_model_config.json")
    p.add_argument("--norm",    default="models/pitch_plus_norm.json")
    p.add_argument("--chunk-rows", type=int, default=50_000)
    for comp in ['stuff', 'loc', 'tun', 'pitch']:
        default = 'per-type' if comp == 'loc' else 'hybrid'
        p.add_argument(f'--{comp}-mode',
                       choices=['per-type', 'global', 'hybrid'],
                       default=default)
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

    print(f"Scoring in chunks of {args.chunk_rows:,} (pitch-level streaming stats)...")
    stats = StreamingStats()
    for start in range(0, n_total, args.chunk_rows):
        end = min(start + args.chunk_rows, n_total)
        chunk = df_full.iloc[start:end].copy()
        chunk = _to_float64(chunk)
        chunk = score_dataframe(chunk, stuff, sf, tunnel, loc, config)
        stats.update(chunk[['pitch_type'] + [col for _, col in COMPONENTS]])
        del chunk
        gc.collect()
        print(f"  Scored {end:,}/{n_total:,} ({100*end/n_total:.0f}%)")

    del df_full
    gc.collect()

    # Global (cross-type) pitch-level stats
    G = {}
    for comp, _ in COMPONENTS:
        m, s = stats.global_(comp)
        G[f'{comp}_mean'] = m
        G[f'{comp}_std']  = s

    print(f"\nGlobal (cross-type) PITCH-LEVEL stats:")
    print(f"  Pitch+:  mean={G['final_mean']:.5f}, std={G['final_std']:.5f}  (n={stats.n_global['final']:,})")
    print(f"  Stuff+:  mean={G['stuff_mean']:.5f}, std={G['stuff_std']:.5f}  (n={stats.n_global['stuff']:,})")
    print(f"  Loc+:    mean={G['loc_mean']:.5f}, std={G['loc_std']:.5f}  (n={stats.n_global['loc']:,})")
    print(f"  Tun+:    mean={G['tun_mean']:.5f}, std={G['tun_std']:.5f}  (n={stats.n_global['tun']:,})")

    print(f"\nModes: stuff={args.stuff_mode}  loc={args.loc_mode}  "
          f"tun={args.tun_mode}  pitch={args.pitch_mode}")

    # Map component -> norm-file key prefix and mode arg name
    NORM_KEY  = {'final': '',      'stuff': 'stuff_', 'loc': 'loc_', 'tun': 'tun_'}
    MODE_ATTR = {'final': 'pitch_mode', 'stuff': 'stuff_mode',
                 'loc':   'loc_mode',   'tun':   'tun_mode'}

    norm = json.load(open(args.norm))
    for pt in list(norm.keys()):
        if stats.n_for(pt) < 100:
            continue
        for comp, _ in COMPONENTS:
            tm, ts = stats.per_type(pt, comp)
            if not (np.isfinite(tm) and np.isfinite(ts) and ts > 0):
                # Degenerate per-type sample — fall back to global for both
                m, s = G[f'{comp}_mean'], G[f'{comp}_std']
            else:
                mode = getattr(args, MODE_ATTR[comp])
                m, s = resolve_mode(mode,
                                    G[f'{comp}_mean'], G[f'{comp}_std'],
                                    tm, ts)
            norm[pt][f'{NORM_KEY[comp]}mean'] = m
            norm[pt][f'{NORM_KEY[comp]}std']  = s

    with open(args.norm, "w") as f:
        json.dump(norm, f, indent=2)

    print(f"\nUpdated {args.norm}")
    print(f"  {'PT':<4} {'n':>8}  "
          f"{'Pitch+ μ':>10} {'Pitch+ σ':>10}  "
          f"{'Stuff μ':>10} {'Stuff σ':>10}  "
          f"{'Loc μ':>10} {'Loc σ':>10}  "
          f"{'Tun μ':>10} {'Tun σ':>10}")
    for pt in sorted(norm):
        if 'stuff_std' not in norm[pt]:
            continue
        n = norm[pt]
        print(f"  {pt:<4} {stats.n_for(pt):>8,}  "
              f"{n['mean']:>10.5f} {n['std']:>10.5f}  "
              f"{n['stuff_mean']:>10.5f} {n['stuff_std']:>10.5f}  "
              f"{n['loc_mean']:>10.5f} {n['loc_std']:>10.5f}  "
              f"{n['tun_mean']:>10.5f} {n['tun_std']:>10.5f}")


if __name__ == "__main__":
    main()
