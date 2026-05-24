#!/usr/bin/env python3
"""
build_slot_regression.py — Compute slot regression and drag residual coefficients.

Slot regression: per (pitch_type, p_throws) linear regression of
  arm_side_break (handedness-corrected pfx_x) and pfx_z on arm_angle.

Drag residual: per pitch_type multiple regression
  ay ~ intercept + slope_velo * release_speed + slope_spin * release_spin_rate

Output format (what the server expects):
  {
    "slot": {
      "FF_R": {"slope_x": ..., "intercept_x": ..., "slope_z": ..., "intercept_z": ...},
      "FF_L": {...},
      ...
    },
    "drag": {
      "FF": {"intercept": ..., "slope_velo": ..., "slope_spin": ...},
      ...
    }
  }

Usage:
    python3 build_slot_regression.py --parquet /path/to/pitch_xrv_2025.parquet
    python3 build_slot_regression.py --parquet /path/to/pitch_xrv_2025.parquet --min-n 200
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="Statcast parquet with arm_angle, pfx_x, pfx_z, pitch_type, p_throws, ay")
    p.add_argument("--out", default="models/slot_regression.json")
    p.add_argument("--min-n", type=int, default=200,
                   help="Minimum pitches per group to include (default: 200)")
    args = p.parse_args()

    print(f"Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)

    needed = ["arm_angle", "pfx_x", "pfx_z", "pitch_type", "p_throws", "ay",
              "release_speed", "release_spin_rate"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        sys.exit(1)

    for c in ["arm_angle", "pfx_x", "pfx_z", "ay", "release_speed", "release_spin_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # arm_side_break: handedness-corrected horizontal break (positive = arm side)
    hand_sign = np.where(df["p_throws"] == "L", -1.0, 1.0)
    df["arm_side_break"] = df["pfx_x"] * hand_sign

    print(f"  {len(df):,} pitches loaded")
    print(f"  arm_angle coverage: {df['arm_angle'].notna().mean()*100:.1f}%")
    print(f"  pitch types: {sorted(df['pitch_type'].dropna().unique().tolist())}")

    # ── Slot regression: per (pitch_type, p_throws) ──────────────────────────
    slot = {}
    print(f"\nSlot regression (arm_side_break and pfx_z ~ arm_angle):")
    print(f"{'Key':<8} {'slope_x':>10} {'int_x':>10} {'slope_z':>10} {'int_z':>10} {'N':>8}")
    print("─" * 60)

    for pt in sorted(df["pitch_type"].dropna().unique()):
        for hand in ["L", "R"]:
            sub = df[
                (df["pitch_type"] == pt) & (df["p_throws"] == hand)
            ].dropna(subset=["arm_angle", "arm_side_break", "pfx_z"])

            if len(sub) < args.min_n:
                continue

            slope_x, int_x = np.polyfit(sub["arm_angle"].values, sub["arm_side_break"].values, 1)
            slope_z, int_z = np.polyfit(sub["arm_angle"].values, sub["pfx_z"].values, 1)

            key = f"{pt}_{hand}"
            slot[key] = {
                "slope_x": float(slope_x),
                "intercept_x": float(int_x),
                "slope_z": float(slope_z),
                "intercept_z": float(int_z),
                "n": int(len(sub)),
            }
            print(f"{key:<8} {slope_x:>10.5f} {int_x:>10.5f} {slope_z:>10.5f} {int_z:>10.5f}  {len(sub):>8,}")

    # ── Drag residual: per pitch_type ─────────────────────────────────────────
    drag = {}
    print(f"\nDrag residual (ay ~ 1 + release_speed + release_spin_rate):")
    print(f"{'PT':<4} {'intercept':>12} {'slope_velo':>12} {'slope_spin':>12} {'N':>8}")
    print("─" * 60)

    for pt in sorted(df["pitch_type"].dropna().unique()):
        sub = df[df["pitch_type"] == pt].dropna(
            subset=["ay", "release_speed", "release_spin_rate"]
        )
        if len(sub) < args.min_n:
            continue

        X = np.column_stack([
            np.ones(len(sub)),
            sub["release_speed"].values,
            sub["release_spin_rate"].values,
        ])
        y = sub["ay"].values
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        intercept, slope_velo, slope_spin = coef

        drag[pt] = {
            "intercept": float(intercept),
            "slope_velo": float(slope_velo),
            "slope_spin": float(slope_spin),
            "n": int(len(sub)),
        }
        print(f"{pt:<4} {intercept:>12.4f} {slope_velo:>12.6f} {slope_spin:>12.6f}  {len(sub):>8,}")

    out = {"slot": slot, "drag": drag}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {len(slot)} slot keys and {len(drag)} drag keys to {out_path}")

    if "FF_R" in slot:
        c = slot["FF_R"]
        ex_x = c["slope_x"] * 45 + c["intercept_x"]
        ex_z = c["slope_z"] * 45 + c["intercept_z"]
        print(f"\nExample: FF_R at arm_angle=45° → expected arm_side_break={ex_x:.3f}, pfx_z={ex_z:.3f}")


if __name__ == "__main__":
    main()
