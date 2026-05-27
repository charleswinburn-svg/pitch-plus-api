"""
plot_stuff_pitch_plus_kde.py
----------------------------
Reads the season pitcher_pitch_type_grades_*.json and draws two figures:

  Fig 1 — Stuff+ KDE by pitch type
  Fig 2 — Pitch+ KDE by pitch type

Usage:
    python3 plot_stuff_pitch_plus_kde.py                   # 2026, min 50 pitches
    python3 plot_stuff_pitch_plus_kde.py --year 2025
    python3 plot_stuff_pitch_plus_kde.py --min-n 100
    python3 plot_stuff_pitch_plus_kde.py --out my_kde.png  # save to file
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# ── Pitch type display names ───────────────────────────────────────────────
PITCH_NAMES = {
    "FF": "4-Seam FB",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CS": "Slow Curve",
    "CH": "Changeup",
    "SC": "Screwball",
    "FS": "Splitter",
    "FO": "Forkball",
    "KN": "Knuckleball",
    "EP": "Eephus",
}

# ── Distinct colours (one per pitch type, colourblind-friendly-ish) ─────────
PITCH_COLORS = {
    "FF": "#ef4444",
    "SI": "#f97316",
    "FC": "#eab308",
    "SL": "#84cc16",
    "ST": "#22c55e",
    "SV": "#14b8a6",
    "CU": "#3b82f6",
    "KC": "#6366f1",
    "CS": "#a855f7",
    "CH": "#ec4899",
    "SC": "#f43f5e",
    "FS": "#0ea5e9",
    "FO": "#f59e0b",
    "KN": "#64748b",
    "EP": "#78716c",
}


def load_grades(year: int, min_n: int) -> dict[str, list[float]]:
    """Return {pitch_type: [stuff_plus, ...], ...} and same for pitch_plus."""
    grades_file = Path(__file__).parent / "season" / f"pitcher_pitch_type_grades_{year}.json"
    if not grades_file.exists():
        sys.exit(f"ERROR: {grades_file} not found. Run score_pitches.py for year {year} first.")

    with open(grades_file) as f:
        data = json.load(f)

    stuff_by_pt: dict[str, list[float]] = defaultdict(list)
    pitch_by_pt: dict[str, list[float]] = defaultdict(list)

    for by_pt in data.values():
        for pt, g in by_pt.items():
            if g.get("n", 0) < min_n:
                continue
            if g.get("stuff_plus") is not None:
                stuff_by_pt[pt].append(float(g["stuff_plus"]))
            if g.get("pitch_plus") is not None:
                pitch_by_pt[pt].append(float(g["pitch_plus"]))

    return stuff_by_pt, pitch_by_pt


def _kde_ax(ax, by_pt: dict[str, list[float]], metric: str, year: int, min_n: int):
    """Draw KDE curves on *ax* for every pitch type that has enough data."""
    x = np.linspace(50, 150, 500)

    # Sort by sample size descending so legend is ordered by usage
    sorted_pts = sorted(by_pt.items(), key=lambda kv: -len(kv[1]))

    for pt, vals in sorted_pts:
        if len(vals) < 5:          # need at least 5 points for a KDE
            continue
        arr = np.array(vals)
        kde = gaussian_kde(arr, bw_method="scott")
        density = kde(x)
        color = PITCH_COLORS.get(pt, "#888888")
        name  = PITCH_NAMES.get(pt, pt)
        label = f"{name} ({pt})  n={len(vals)}"
        ax.plot(x, density, lw=2, color=color, label=label)
        ax.fill_between(x, density, alpha=0.08, color=color)

    ax.axvline(100, color="#ffffff", lw=1, ls="--", alpha=0.4)
    ax.set_xlabel(f"{metric} (100 = league average)", fontsize=11, color="#cccccc")
    ax.set_ylabel("Density", fontsize=11, color="#cccccc")
    ax.set_title(f"{year} {metric} by Pitch Type  (min {min_n} pitches)", fontsize=13, color="#eeeeee", pad=10)
    ax.legend(
        fontsize=8.5,
        loc="upper left",
        framealpha=0.25,
        labelcolor="#dddddd",
        edgecolor="#444",
    )
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


def main():
    parser = argparse.ArgumentParser(description="KDE plots of Stuff+ and Pitch+ by pitch type.")
    parser.add_argument("--year",  type=int, default=2026, help="Season year (default 2026)")
    parser.add_argument("--min-n", type=int, default=50,   help="Min pitches to include a pitcher-pitch-type (default 50)")
    parser.add_argument("--out",   type=str, default=None, help="Save figure to this path instead of showing")
    args = parser.parse_args()

    stuff_by_pt, pitch_by_pt = load_grades(args.year, args.min_n)

    if not stuff_by_pt:
        sys.exit("No data found — check --year and --min-n values.")

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#111827")
    for ax in axes:
        ax.set_facecolor("#1f2937")

    _kde_ax(axes[0], stuff_by_pt, "Stuff+", args.year, args.min_n)
    _kde_ax(axes[1], pitch_by_pt, "Pitch+", args.year, args.min_n)

    fig.suptitle(f"{args.year} Pitch Quality Distributions", fontsize=15, color="#f9fafb", y=1.01)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
