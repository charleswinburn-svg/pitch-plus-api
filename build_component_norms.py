#!/usr/bin/env python3
"""
build_component_norms.py — Build Stuff+/Loc+/Tun+/Pitch+ norms.

Computes PITCH-LEVEL mean and std per pitch type (and globally), via streaming
sums to avoid OOM on large parquets.

Why pitch-level:
    The z-score formula at scoring time is (pitch_xRV - mean) / std applied to
    a single pitch. So mean and std must be calibrated on the same unit —
    individual pitches — not on game-level aggregates. Game aggregates artificially
    shrink the std (averaging cancels noise), which made elite individual pitches
    look like extreme outliers and pushed grades to wrong values.

Usage:
    cd ~/pitch-plus-api
    python3 build_component_norms.py --parquet pitch_xrv_2025.parquet
    python3 build_component_norms.py --parquet pitch_xrv_2025.parquet --chunk-rows 30000
"""
import argparse, json, sys, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


COMPONENTS = [
    ('final', 'xRV_final'),
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
                self.n[pt][comp]      += len(vals)
                self.sum[pt][comp]    += vals.sum()
                self.sumsq[pt][comp]  += (vals * vals).sum()
        for comp, col in COMPONENTS:
            vals = df[col].dropna().values.astype('float64')
            if len(vals) == 0:
                continue
            self.n_global[comp]      += len(vals)
            self.sum_global[comp]    += vals.sum()
            self.sumsq_global[comp]  += (vals * vals).sum()

    @staticmethod
    def _mean_std(n, s, ss):
        if n < 2:
            return float('nan'), float('nan')
        mean = s / n
        # sample variance (ddof=1) — matches pandas Series.std() default
        var = (ss - s * s / n) / (n - 1)
        var = max(var, 0.0)  # guard against tiny negatives from floating-point
        return float(mean), float(var ** 0.5)

    def per_type(self, pt, comp):
        return self._mean_std(self.n[pt][comp], self.sum[pt][comp], self.sumsq[pt][comp])

    def global_(self, comp):
        return self._mean_std(self.n_global[comp], self.sum_global[comp], self.sumsq_global[comp])

    def n_for(self, pt):
        return max(self.n[pt][c] for c, _ in COMPONENTS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="pitch_xrv_2025.parquet")
    p.add_argument("--models", default="models")
    p.add_argument("--config", default="models/final_model_config.json")
    p.add_argument("--norm", default="models/pitch_plus_norm.json")
    p.add_argument("--chunk-rows", type=int, default=50_000)
    p.add_argument("--global-stuff", action="store_true",
                   help="Use one global mean/std for Stuff+ (cross-pitch-type comparison)")
    p.add_argument("--global-loc", action="store_true",
                   help="Use one global mean/std for Loc+ (rarely wanted)")
    p.add_argument("--global-tun", action="store_true",
                   help="Use one global mean/std for Tun+")
    p.add_argument("--global-pitch", action="store_true",
                   help="Use one global mean/std for Pitch+ overall")
    p.add_argument("--global-all", action="store_true",
                   help="Shortcut: enable all four global flags")
    args = p.parse_args()

    if args.global_all:
        args.global_stuff = args.global_loc = args.global_tun = args.global_pitch = True

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

    # Map component name -> norm-file key prefix (final maps to bare 'mean'/'std')
    NORM_KEY = {'final': '', 'stuff': 'stuff_', 'loc': 'loc_', 'tun': 'tun_'}

    print(f"\nGlobal (cross-type) PITCH-LEVEL stats:")
    print(f"  Pitch+:  mean={G['final_mean']:.5f}, std={G['final_std']:.5f}  (n={stats.n_global['final']:,})")
    print(f"  Stuff+:  mean={G['stuff_mean']:.5f}, std={G['stuff_std']:.5f}  (n={stats.n_global['stuff']:,})")
    print(f"  Loc+:    mean={G['loc_mean']:.5f}, std={G['loc_std']:.5f}  (n={stats.n_global['loc']:,})")
    print(f"  Tun+:    mean={G['tun_mean']:.5f}, std={G['tun_std']:.5f}  (n={stats.n_global['tun']:,})")

    print(f"\nNorm scope:")
    print(f"  Pitch+:  {'GLOBAL' if args.global_pitch else 'per-type'}")
    print(f"  Stuff+:  {'GLOBAL' if args.global_stuff else 'per-type'}")
    print(f"  Loc+:    {'GLOBAL' if args.global_loc else 'per-type'}")
    print(f"  Tun+:    {'GLOBAL' if args.global_tun else 'per-type'}")

    norm = json.load(open(args.norm))
    use_global = {
        'final': args.global_pitch,
        'stuff': args.global_stuff,
        'loc':   args.global_loc,
        'tun':   args.global_tun,
    }

    for pt in list(norm.keys()):
        if stats.n_for(pt) < 100:
            continue
        for comp, _ in COMPONENTS:
            if use_global[comp]:
                m, s = G[f'{comp}_mean'], G[f'{comp}_std']
            else:
                m, s = stats.per_type(pt, comp)
                if not (np.isfinite(m) and np.isfinite(s) and s > 0):
                    # Fall back to global if per-type sample is too small / degenerate
                    m, s = G[f'{comp}_mean'], G[f'{comp}_std']
            norm[pt][f'{NORM_KEY[comp]}mean'] = m
            norm[pt][f'{NORM_KEY[comp]}std']  = s

    with open(args.norm, "w") as f:
        json.dump(norm, f, indent=2)

    print(f"\nUpdated {args.norm}")
    print(f"  {'PT':<4} {'n':>8}  {'Pitch+ μ':>10} {'Pitch+ σ':>10}  "
          f"{'Stuff μ':>10} {'Stuff σ':>10}  {'Loc μ':>10} {'Loc σ':>10}  "
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
