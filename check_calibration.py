"""
check_calibration.py
--------------------
Reads the pre-computed season grades JSON and reports the usage-weighted
league-average Stuff+, Loc+, Tun+, and Pitch+.  League average should be
100.0 for each metric; values outside ±1 indicate the norm file needs to
be rebuilt with build_component_norms.py.

Usage:
    python3 check_calibration.py                 # 2026, min 50 pitches
    python3 check_calibration.py --year 2025
    python3 check_calibration.py --min-n 100
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def check(year: int, min_n: int):
    grades_file = Path(__file__).parent / "season" / f"pitcher_pitch_type_grades_{year}.json"
    if not grades_file.exists():
        sys.exit(f"ERROR: {grades_file} not found.")

    with open(grades_file) as f:
        data = json.load(f)

    totals   = defaultdict(float)
    counts   = defaultdict(int)
    skipped  = 0
    included = 0

    for by_pt in data.values():
        for pt, g in by_pt.items():
            n = g.get("n", 0)
            if n < min_n:
                skipped += 1
                continue
            included += 1
            for metric, key in [
                ("Stuff+",  "stuff_plus"),
                ("Loc+",    "loc_plus"),
                ("Tun+",    "tun_plus"),
                ("Pitch+",  "pitch_plus"),
            ]:
                v = g.get(key)
                if v is not None:
                    totals[metric]  += v * n
                    counts[metric]  += n

    print(f"\n{'Metric':<10}  {'League avg':>10}  {'Pitcher-type pairs':>20}")
    print("-" * 46)
    all_ok = True
    for metric in ("Stuff+", "Loc+", "Tun+", "Pitch+"):
        c = counts[metric]
        avg = totals[metric] / c if c else float("nan")
        flag = ""
        if abs(avg - 100.0) > 1.0:
            flag = "  ← DRIFT DETECTED"
            all_ok = False
        print(f"{metric:<10}  {avg:>10.2f}  {c:>20,}{flag}")

    print()
    print(f"  Included: {included:,} pitcher-type pairs (n ≥ {min_n})")
    print(f"  Skipped:  {skipped:,} pitcher-type pairs (n < {min_n})")
    if all_ok:
        print("\n  ✓ All metrics within ±1 of 100 — norms are well-calibrated.")
    else:
        print(
            "\n  ✗ One or more metrics have drifted. Rebuild norms with:\n"
            "      python3 build_component_norms.py --parquet tmp/pitch_xrv_"
            f"{year}.parquet\n"
            "    then re-run score_pitches.py."
        )


def main():
    parser = argparse.ArgumentParser(description="Check league-average Plus calibration.")
    parser.add_argument("--year",  type=int, default=2026)
    parser.add_argument("--min-n", type=int, default=50,
                        help="Min pitches per pitcher-type to include (default 50)")
    args = parser.parse_args()
    check(args.year, args.min_n)


if __name__ == "__main__":
    main()
