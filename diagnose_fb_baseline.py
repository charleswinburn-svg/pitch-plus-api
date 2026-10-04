#!/usr/bin/env python3
"""
diagnose_fb_baseline.py — check whether a pitcher's stored fastball baseline
matches the stuff model's fastball definition, and what it does to his grades.

The stuff model was trained with family = "within N mph of the pitcher's FASTEST
fastball type (FF/SI/FC)" (models/stuff_model_metadata.json), but
build_pitcher_baselines.py stores the MOST-THROWN fastball type. For a
cutter-heavy pitcher those differ: a 96 mph four-seamer measured against a
90 mph cutter baseline falls outside the window and gets scored by the
off-speed sub-model.

Scores the pitcher's season pitches with the server's own engineer_and_score,
once with the stored baseline and once with a baseline built from his fastest
fastball type, then lists other pitchers whose stored type differs.

Usage (on the droplet; read-only, changes nothing):
    /var/www/pitch-plus-api/.venv/bin/python diagnose_fb_baseline.py 656876 \
        [--parquet /var/www/pasttheeyetest.com/pitch_xrv_2026.parquet] [--api-dir /var/www/pitch-plus-api]
"""
import argparse, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd


ROW_COLS = ['pitch_type', 'release_speed', 'release_spin_rate', 'release_extension', 'pfx_x', 'pfx_z',
            'plate_x', 'plate_z', 'release_pos_x', 'release_pos_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
            'spin_axis', 'sz_top', 'sz_bot', 'pitcher', 'stand', 'p_throws']
FB_TYPES = ['FF', 'SI', 'FC']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pitcher_id', type=int)
    ap.add_argument('--parquet', default='/var/www/pasttheeyetest.com/pitch_xrv_2026.parquet')
    ap.add_argument('--api-dir', default=os.path.dirname(os.path.abspath(__file__)),
                    help='pitch-plus-api checkout to import server.py from (default: this file\'s folder)')
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.api_dir))
    pid, key = args.pitcher_id, str(args.pitcher_id)

    print('Loading server (models, baselines)...', flush=True)
    import server
    from build_pitcher_baselines import prepare_fastballs, baseline_record
    thr = server._STUFF_MPH_THRESHOLD

    df = pd.read_parquet(args.parquet)
    me = df[pd.to_numeric(df['pitcher'], errors='coerce') == pid].copy()
    if me.empty:
        sys.exit(f'No pitches for {pid} in {args.parquet}')
    for c in ROW_COLS:
        if c not in ('pitch_type', 'stand', 'p_throws') and c in me.columns:
            me[c] = pd.to_numeric(me[c], errors='coerce')
    me = me[me['pitch_type'].notna() & ~me['pitch_type'].isin(['UN', 'PO'])]

    # ── 1. Stored baseline ──
    stored = server.pitcher_baselines.get(key)
    print(f'\n== Stored baseline for {pid} ==')
    if stored:
        print(f"  fb_type={stored.get('fb_type')}  fb_velo={stored.get('fb_velo', float('nan')):.1f}  "
              f"n={stored.get('_n')}  source={stored.get('_source', '-')}  cold_start={stored.get('_cold_start')}")
    else:
        print('  none stored (server synthesizes one from each request)')

    # ── 2. Pitch mix this season ──
    mix = me.groupby('pitch_type').agg(n=('release_speed', 'size'), velo=('release_speed', 'mean'),
                                       h_brk_in=('pfx_x', lambda s: s.mean() * 12),
                                       ivb_in=('pfx_z', lambda s: s.mean() * 12)).sort_values('n', ascending=False)
    print(f'\n== {len(me):,} pitches in {os.path.basename(args.parquet)} ==')
    print(mix.round(1).to_string())
    fbs = mix[mix.index.isin(FB_TYPES) & (mix['n'] >= 10)]
    most_thrown = fbs['n'].idxmax() if len(fbs) else None
    fastest = fbs['velo'].idxmax() if len(fbs) else None
    print(f'\n  most-thrown fastball type: {most_thrown}   fastest fastball type (model definition): {fastest}')

    # ── 3+4. Score with stored vs fastest-type baseline ──
    rows = me[[c for c in ROW_COLS if c in me.columns]].to_dict('records')

    def score(baseline):
        saved = server.pitcher_baselines.get(key)
        if baseline is not None:
            server.pitcher_baselines[key] = baseline
        try:
            scored = server.engineer_and_score(rows, server.pitch_plus_norm)
            # The baseline the server actually used (stored, or synthesized when none).
            fbv = server._effective_baselines(pd.DataFrame(rows), server.pitcher_baselines,
                                              server._league_fb).get(key, {}).get('fb_velo')
        finally:
            if saved is None:
                server.pitcher_baselines.pop(key, None)
            else:
                server.pitcher_baselines[key] = saved
        xrv, fb_routed = defaultdict(list), defaultdict(list)
        for s in scored:
            pt = s.get('pitch_type_display', s.get('pitch_type'))
            xrv[pt].append(s.get('xRV_stuff', 0.0))
            v = rows[s['index']].get('release_speed')
            ok = fbv is not None and v is not None and not pd.isna(fbv) and not pd.isna(v)
            fb_routed[pt].append(bool(abs(v - fbv) <= thr) if ok else pt in FB_TYPES)
        out = {}
        for pt, vals in xrv.items():
            plus = server._per_type_plus_agg(float(np.mean(vals)), server.PITCH_ALIASES.get(pt, pt), 'stuff')
            out[pt] = (plus, 100.0 * np.mean(fb_routed[pt]))
        return out

    cur = score(None)
    alt = None
    if fastest and (not stored or stored.get('fb_type') != fastest):
        fb = prepare_fastballs(me.copy())
        sub = fb[fb['pitch_type'] == fastest]
        rec = dict(stored or {})
        rec.update(baseline_record(sub, fastest, source='diagnostic'))
        alt = score(rec)

    print(f'\n== Stuff+ by pitch type (FB model = within {thr} mph of baseline velo) ==')
    hdr = f"  {'type':5s} {'n':>5s}   {'stored baseline':>22s}"
    if alt:
        hdr += f"   {('baseline = ' + fastest):>22s}"
    print(hdr)
    for pt in mix.index:
        if pt not in cur:
            continue
        p, f = cur[pt]
        line = f"  {pt:5s} {int(mix.loc[pt, 'n']):5d}   {p:7.1f} Stuff+ {f:4.0f}% FB"
        if alt and pt in alt:
            p2, f2 = alt[pt]
            line += f"   {p2:7.1f} Stuff+ {f2:4.0f}% FB   ({p2 - p:+.1f})"
        print(line)
    if alt is None:
        print('\n  Stored baseline already uses the fastest fastball type — the mismatch is not the cause.')

    # ── 5. Who else is affected ──
    all_fb = df[df['pitch_type'].isin(FB_TYPES)].copy()
    all_fb['release_speed'] = pd.to_numeric(all_fb['release_speed'], errors='coerce')
    g = all_fb.groupby(['pitcher', 'pitch_type'])['release_speed'].agg(['size', 'mean']).reset_index()
    g = g[g['size'] >= 10]
    fastest_by = g.loc[g.groupby('pitcher')['mean'].idxmax()].set_index('pitcher')
    diff = []
    for p_, r in fastest_by.iterrows():
        bl = server.pitcher_baselines.get(str(int(p_)))
        if bl and bl.get('fb_type') and bl['fb_type'] != r['pitch_type']:
            gap = r['mean'] - (bl.get('fb_velo') or np.nan)
            diff.append((int(p_), bl['fb_type'], r['pitch_type'], round(gap, 1), int(r['size'])))
    print(f'\n== Pitchers whose stored baseline type != fastest fastball type: {len(diff)} ==')
    print(f'   (gap > {thr} mph means their fastest fastball is scored by the off-speed model)')
    for p_, b, f_, gap, n in sorted(diff, key=lambda x: -x[3])[:25]:
        flag = '  <-- off-speed model' if gap > thr else ''
        print(f'  {p_:>7d}  stored {b}  fastest {f_}  gap {gap:+.1f} mph  ({n} {f_}){flag}')


if __name__ == '__main__':
    main()
