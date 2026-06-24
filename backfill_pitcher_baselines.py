#!/usr/bin/env python3
"""
backfill_pitcher_baselines.py — Add a missing fastball baseline for one (or a
few) pitchers, without rebuilding the whole pitcher_baselines.json.

Why this exists
---------------
pitcher_baselines.json is built off prior-season data, so in-season debuts
(e.g. Trey Gibson, MLBAMID 694346, BAL — first MLB action in 2026) have no
entry. With no fastball baseline the live API path (engineer_and_score) leaves
fb_velo = NaN, which means every delta feature is left at 0.0
(delta_velo, delta_spin, delta_vaa, movement_separation, release_diff_*, ...)
and the FB/OFF stuff sub-model routing falls back to pitch-type membership
only. The result is wonky Stuff+ — offspeed pitches in particular grade far
too low because the model thinks they have zero velocity/movement separation
from the fastball.

This script pulls ONLY the requested pitcher(s) from Baseball Savant (filtered
server-side via pitchers_lookup[]), computes the fastball baseline with the
repo's own prepare_fastballs()/baseline_record() so the numbers are identical
to a full rebuild, and merges just those entries into the JSON. Every other
pitcher is left byte-for-byte untouched.

Run on a host with outbound access to baseballsavant.mlb.com, from the repo
root, with the project venv active:

    python3 backfill_pitcher_baselines.py                 # default: Gibson, 2026
    python3 backfill_pitcher_baselines.py 694346 696258   # several pitchers
    python3 backfill_pitcher_baselines.py --season 2026 --window-days 60

Then restart / redeploy the API so it reloads models/pitcher_baselines.json
(the file is read once at startup).
"""
import argparse
import json
import sys
import time
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# Reuse the exact baseline math the nightly build uses, so a backfilled entry
# is indistinguishable from one produced by build_pitcher_baselines.py.
from build_pitcher_baselines import prepare_fastballs, baseline_record, MIN_PITCHES

SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
CHUNK_DAYS = 14

# Trey Gibson (BAL) — 2026 debut, the pitcher whose Stuff+ line was wonky.
DEFAULT_PIDS = [694346]


def fetch_pitcher_savant(pid, year, start_date, end_date):
    """Pull one pitcher's Statcast 'details' rows for [start_date, end_date].

    Mirrors build_arm_angle_baselines.fetch_statcast's request, but filters to
    a single pitcher with pitchers_lookup[] so this is a handful of tiny
    requests instead of a full-league pull.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    print(f"  Savant pull for {pid}: {start.date()} -> {end.date()}")

    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=CHUNK_DAYS - 1), end)
        s = cur.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        print(f"    {s} to {e}...", end=" ", flush=True)

        params = {
            "all": "true", "hfPT": "", "hfAB": "", "hfGT": "R|", "hfPR": "",
            "hfZ": "", "stadium": "", "hfBBL": "", "hfNewZones": "", "hfPull": "",
            "hfC": "", "hfSea": str(year), "hfSit": "",
            "player_type": "pitcher", "hfOuts": "", "opponent": "",
            "pitcher_throws": "", "batter_stands": "", "hfSA": "",
            "game_date_gt": s, "game_date_lt": e,
            "hfMo": "", "team": "", "home_road": "", "hfRO": "",
            "position": "", "hfInfield": "", "hfOutfield": "", "hfInn": "",
            "hfBBT": "", "hfFlag": "", "metric_1": "", "group_by": "name",
            "min_pitches": "0", "min_results": "0", "min_pas": "0",
            "sort_col": "pitches", "player_event_sort": "api_p_release_speed",
            "sort_order": "desc", "chk_": "on", "type": "details",
            "csv": "true",
            "pitchers_lookup[]": str(pid),
        }

        df = None
        for attempt in range(4):
            try:
                r = requests.get(SAVANT_URL, params=params,
                                 headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
                if r.status_code == 200 and len(r.text) > 200:
                    df = pd.read_csv(StringIO(r.text))
                else:
                    df = pd.DataFrame()
                break
            except Exception as ex:  # transient network error -> backoff
                if attempt == 3:
                    print(f"failed ({type(ex).__name__}: {ex})")
                    df = pd.DataFrame()
                else:
                    time.sleep(2 ** attempt)
        if df is not None and len(df) > 0:
            print(f"{len(df):,} pitches")
            chunks.append(df)
        else:
            print("0 pitches")
        cur = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def build_one(pid, season, start_date, end_date, min_fastballs):
    df = fetch_pitcher_savant(pid, season, start_date, end_date)
    if df.empty:
        print(f"  ! no Statcast rows for {pid} — skipping")
        return None
    if "pitcher" in df.columns:  # defensive: pitchers_lookup[] should already filter
        df = df[df["pitcher"] == pid].copy()
    if df.empty:
        print(f"  ! no rows attributed to {pid} — skipping")
        return None

    fb = prepare_fastballs(df, label=f"  {pid} fastballs")
    if len(fb) < min_fastballs:
        print(f"  ! only {len(fb)} fastballs (< {min_fastballs}) — skipping")
        return None

    # Primary fastball = most-thrown of {FF, SI, FC}, exactly like add_records().
    primary = fb["pitch_type"].value_counts().idxmax()
    sub = fb[fb["pitch_type"] == primary]
    rec = baseline_record(sub, primary, cold=False, source=f"savant_{season}_backfill")
    print(f"  primary FB = {primary} (n={len(sub)}): "
          f"velo={rec['fb_velo']:.1f}, spin={rec['fb_spin']:.0f}, "
          f"pfx_x={rec['fb_pfx_x']:.2f}, pfx_z={rec['fb_pfx_z']:.2f}, "
          f"vaa={rec['fb_vaa']:.2f}, throws={rec.get('p_throws')}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pids", nargs="*", type=int, default=DEFAULT_PIDS,
                    help=f"MLBAM pitcher id(s) to backfill (default: {DEFAULT_PIDS})")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--start", default=None,
                    help="Window start YYYY-MM-DD (default: {season}-03-20)")
    ap.add_argument("--end", default=None,
                    help="Window end YYYY-MM-DD (default: today)")
    ap.add_argument("--window-days", type=int, default=None,
                    help="If set, use a trailing N-day window ending at --end "
                         "(matches build_pitcher_baselines.py's 60-day window). "
                         "Default: full season-to-date, better for low-IP debuts.")
    ap.add_argument("--min-fastballs", type=int, default=MIN_PITCHES,
                    help=f"Minimum fastballs required to write a baseline (default {MIN_PITCHES})")
    ap.add_argument("--out", default=str(REPO / "models" / "pitcher_baselines.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print computed records but do not write the file")
    args = ap.parse_args()

    end = args.end or min(f"{args.season}-11-01", date.today().strftime("%Y-%m-%d"))
    if args.window_days is not None:
        start = (pd.Timestamp(end) - pd.Timedelta(days=args.window_days)).strftime("%Y-%m-%d")
    else:
        start = args.start or f"{args.season}-03-20"

    out_path = Path(args.out)
    baselines = json.load(open(out_path))
    print(f"Loaded {len(baselines)} existing baselines from {out_path}\n")

    new_recs = {}
    for pid in args.pids:
        print(f"Pitcher {pid}:")
        rec = build_one(pid, args.season, start, end, args.min_fastballs)
        if rec is not None:
            new_recs[str(pid)] = rec
        print()

    if not new_recs:
        print("Nothing to write.")
        return

    if args.dry_run:
        print("--dry-run: not writing. Records:")
        print(json.dumps(new_recs, indent=2))
        return

    added = [p for p in new_recs if p not in baselines]
    updated = [p for p in new_recs if p in baselines]
    baselines.update(new_recs)
    with open(out_path, "w") as f:
        json.dump(baselines, f, separators=(",", ":"))  # compact, matches build script

    print(f"Wrote {out_path}: {len(baselines)} baselines "
          f"(added {added or '[]'}, updated {updated or '[]'}).")
    print("Restart / redeploy the API so it reloads pitcher_baselines.json.")


if __name__ == "__main__":
    main()
