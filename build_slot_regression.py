#!/usr/bin/env python3
"""
build_slot_regression.py — Compute per-pitch-type linear regression coefficients
mapping arm_angle → expected pfx_x and pfx_z. Matches training-time logic.

For each pitch type:
  expected_pfx_x = slope_x * arm_angle + intercept_x
  expected_pfx_z = slope_z * arm_angle + intercept_z

Live API can then compute pfx_*_dev_from_slot for ANY pitcher (even debuts)
using just their arm_angle estimate + the actual pitch's pfx values.

Usage:
    cd ~/pitch-plus-api
    python3 build_slot_regression.py --parquet statcast_mlb_2025.parquet
    python3 build_slot_regression.py --parquet statcast_mlb_2025.parquet --years 2024,2025
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="Statcast parquet with arm_angle, pfx_x, pfx_z, pitch_type")
    p.add_argument("--out", default="models/slot_regression.json")
    args = p.parse_args()

    print(f"Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)

    needed = ["arm_angle", "pfx_x", "pfx_z", "pitch_type"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        sys.exit(1)

    # Numeric coercion
    for c in ["arm_angle", "pfx_x", "pfx_z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"  {len(df):,} pitches loaded")
    print(f"  arm_angle coverage: {df['arm_angle'].notna().mean()*100:.1f}%")
    print(f"  pitch types: {sorted(df['pitch_type'].dropna().unique().tolist())}")

    coefs = {}
    print(f"\n{'PT':<4} {'slope_x':>10} {'int_x':>10} {'slope_z':>10} {'int_z':>10}  {'N':>8}")
    print("─" * 60)

    for pt in sorted(df["pitch_type"].dropna().unique()):
        sub = df[df["pitch_type"] == pt].dropna(subset=["arm_angle", "pfx_x", "pfx_z"])
        if len(sub) < 100:
            continue

        # Linear regression: pfx_x = slope * arm_angle + intercept
        slope_x, int_x = np.polyfit(sub["arm_angle"].values, sub["pfx_x"].values, 1)
        slope_z, int_z = np.polyfit(sub["arm_angle"].values, sub["pfx_z"].values, 1)

        coefs[pt] = {
            "slope_x": float(slope_x),
            "intercept_x": float(int_x),
            "slope_z": float(slope_z),
            "intercept_z": float(int_z),
            "n": int(len(sub)),
        }
        print(f"{pt:<4} {slope_x:>10.5f} {int_x:>10.5f} {slope_z:>10.5f} {int_z:>10.5f}  {len(sub):>8,}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coefs, f, indent=2)

    print(f"\nWrote {len(coefs)} pitch types to {out_path}")
    print(f"\nExample lookup for FF + arm_angle=45°:")
    if "FF" in coefs:
        c = coefs["FF"]
        ex_x = c["slope_x"] * 45 + c["intercept_x"]
        ex_z = c["slope_z"] * 45 + c["intercept_z"]
        print(f"  expected pfx_x = {ex_x:.3f}, expected pfx_z = {ex_z:.3f}")


if __name__ == "__main__":
    main()
