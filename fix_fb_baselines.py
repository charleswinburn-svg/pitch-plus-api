#!/usr/bin/env python3
"""
fix_fb_baselines.py — one-off repair: re-anchor stored fastball baselines on the
pitcher's FASTEST fastball type, the definition the stuff model was trained with.

build_pitcher_baselines.py used to store the MOST-THROWN fastball type, so
cutter-heavy pitchers (e.g. Drew Rasmussen, stored FC 90.2) got their 95+ mph
four-seamer/sinker routed to the off-speed sub-model. The builders now pick the
fastest type; this rewrites only the existing records that disagree, from the
season parquet, and leaves every other pitcher's baseline untouched.

Dry run by default (prints the changes). --write backs up the file first.

Usage (on the droplet, from /var/www/pitch-plus-api):
    .venv/bin/python fix_fb_baselines.py --parquet /var/www/pasttheeyetest.com/pitch_xrv_2026.parquet
    .venv/bin/python fix_fb_baselines.py --parquet /var/www/pasttheeyetest.com/pitch_xrv_2026.parquet --write
"""
import argparse, json, shutil, sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_pitcher_baselines import prepare_fastballs, baseline_record, MIN_PITCHES  # noqa: E402
from baseline_fallback import primary_fastball_type                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', nargs='+', required=True, help='season Statcast parquet(s)')
    ap.add_argument('--baselines', default=str(HERE / 'models' / 'pitcher_baselines.json'))
    ap.add_argument('--write', action='store_true', help='apply the changes (default: dry run)')
    args = ap.parse_args()

    path = Path(args.baselines)
    baselines = json.loads(path.read_text())
    df = pd.concat([pd.read_parquet(p) for p in args.parquet], ignore_index=True)
    fb = prepare_fastballs(df, label='Season fastballs')
    fb = fb[pd.to_numeric(fb['pitcher'], errors='coerce').notna()]

    changes = []
    for pid, group in fb.groupby('pitcher'):
        key = str(int(pid))
        old = baselines.get(key)
        if not old or old.get('_cold_start'):
            continue                      # no stored record / league cold start: nothing to fix
        fastest = primary_fastball_type(group)
        if fastest is None or old.get('fb_type') == fastest:
            continue
        sub = group[group['pitch_type'] == fastest]
        if len(sub) < MIN_PITCHES:
            continue
        new = baseline_record(sub, fastest, cold=False,
                              source=f"{old.get('_source', 'stored')}+fastest_fix")
        changes.append((key, old, new))

    print(f'\n{len(changes)} of {len(baselines)} stored baselines use a fastball type other than the '
          f'pitcher\'s fastest:')
    for key, old, new in sorted(changes, key=lambda c: c[1].get('fb_velo', 0) - c[2]['fb_velo']):
        print(f"  {key:>7s}  {old.get('fb_type')} {old.get('fb_velo', float('nan')):5.1f} mph "
              f"-> {new['fb_type']} {new['fb_velo']:5.1f} mph  (n={new['_n']})")

    if not args.write:
        print('\nDry run — nothing written. Re-run with --write to apply.')
        return
    if not changes:
        print('\nNothing to write.')
        return
    backup = path.with_name(f"{path.stem}.bak-{datetime.now():%Y%m%d-%H%M%S}{path.suffix}")
    shutil.copy2(path, backup)
    for key, _, new in changes:
        baselines[key] = new
    path.write_text(json.dumps(baselines, separators=(',', ':')))
    print(f'\nBacked up -> {backup}\nWrote {len(changes)} re-anchored baselines -> {path}')


if __name__ == '__main__':
    main()
