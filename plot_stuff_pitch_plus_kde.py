"""
plot_stuff_pitch_plus_kde.py
----------------------------
Reads the season pitcher_pitch_type_grades_*.json and draws two figures:

  Fig 1 — Stuff+ KDE grid (one subplot per pitch type)
  Fig 2 — Pitch+ KDE grid (one subplot per pitch type)

Usage:
    python3 plot_stuff_pitch_plus_kde.py                   # 2026, min 50 pitches
    python3 plot_stuff_pitch_plus_kde.py --year 2025
    python3 plot_stuff_pitch_plus_kde.py --min-n 100
    python3 plot_stuff_pitch_plus_kde.py --out my_kde.png  # save to file (appends _stuff / _pitch)
"""

import argparse
import json
import math
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

# Preferred display order (most common first)
PT_ORDER = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "SV", "FO", "CS", "KN", "SC", "EP"]

PITCH_COLORS = {
    "FF": "#D22D49",
    "SI": "#FE9D00",
    "FC": "#933F2C",
    "SL": "#EEE716",
    "CU": "#00D1ED",
    "CH": "#1DBE3A",
    "FS": "#3BACAC",
    "ST": "#DDB33A",
    "KC": "#6236CD",
    "CS": "#0068FF",
    "SV": "#93AFD4",
    "KN": "#3C44CD",
    "SC": "#60DB33",
    "FO": "#98DDB1",
    "EP": "#999999",
    "FA": "#D22D49",
    "GY": "#FFFF99",
}

BG_DARK  = "#111827"
CELL_BG  = "#1f2937"
GRID_COL = "#374151"


def load_grades(year: int, min_n: int):
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


def _draw_single(ax, pt: str, vals: list[float], metric: str):
    """Draw one KDE panel for a single pitch type."""
    color = PITCH_COLORS.get(pt, "#888888")
    name  = PITCH_NAMES.get(pt, pt)
    arr   = np.array(vals)

    x = np.linspace(50, 150, 500)
    kde     = gaussian_kde(arr, bw_method="scott")
    density = kde(x)

    ax.plot(x, density, lw=2.2, color=color)
    ax.fill_between(x, density, alpha=0.20, color=color)

    # league-average reference line
    ax.axvline(100, color="#ffffff", lw=0.8, ls="--", alpha=0.35)

    # mean marker
    mean_val = arr.mean()
    ax.axvline(mean_val, color=color, lw=1.2, ls=":", alpha=0.8)

    ax.set_title(
        f"{name}  ({pt})",
        fontsize=10, color="#eeeeee", pad=5, fontweight="bold",
    )
    ax.text(
        0.97, 0.93, f"n={len(vals)}\nμ={mean_val:.1f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color="#9ca3af",
    )

    ax.set_xlim(50, 150)
    ax.set_xlabel(metric, fontsize=7, color="#9ca3af", labelpad=2)
    ax.set_ylabel("Density", fontsize=7, color="#9ca3af", labelpad=2)
    ax.tick_params(colors="#6b7280", labelsize=7)
    ax.set_facecolor(CELL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)


def make_grid_figure(by_pt: dict[str, list[float]], metric: str, year: int, min_n: int):
    """Build a figure with one subplot per pitch type, sorted by usage."""
    # Sort by preferred order, fall back to alpha for unknowns
    pts = sorted(
        [pt for pt, v in by_pt.items() if len(v) >= 5],
        key=lambda pt: (PT_ORDER.index(pt) if pt in PT_ORDER else 99, pt),
    )

    n = len(pts)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.2, nrows * 3.2),
        squeeze=False,
    )
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        f"{year} {metric} Distributions by Pitch Type  (min {min_n} pitches)",
        fontsize=14, color="#f9fafb", y=1.01,
    )

    for i, pt in enumerate(pts):
        r, c = divmod(i, ncols)
        _draw_single(axes[r][c], pt, by_pt[pt], metric)

    # Hide any leftover empty cells
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout(h_pad=2.5, w_pad=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Per-pitch-type KDE grids for Stuff+ and Pitch+.")
    parser.add_argument("--year",  type=int, default=2026)
    parser.add_argument("--min-n", type=int, default=50)
    parser.add_argument("--out",   type=str, default=None,
                        help="Base path for output — two files are written: <out>_stuff.png and <out>_pitch.png")
    args = parser.parse_args()

    stuff_by_pt, pitch_by_pt = load_grades(args.year, args.min_n)
    if not stuff_by_pt:
        sys.exit("No data found — check --year and --min-n values.")

    plt.style.use("dark_background")

    fig_stuff = make_grid_figure(stuff_by_pt, "Stuff+", args.year, args.min_n)
    fig_pitch = make_grid_figure(pitch_by_pt, "Pitch+", args.year, args.min_n)

    if args.out:
        base = args.out.removesuffix(".png")
        for fig, suffix in [(fig_stuff, "_stuff.png"), (fig_pitch, "_pitch.png")]:
            path = base + suffix
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"Saved → {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
