#!/usr/bin/env python3
"""baseline_fallback.py — synthesize a fastball baseline for pitchers that are
missing from pitcher_baselines.json, so their Stuff+/Tun+ delta features don't
collapse to zero (which makes an in-season debut's grades wonky — see Trey
Gibson, 2026).

Fallback chain, per pitcher:
    1. stored baseline (pitcher_baselines.json)      ← returned unchanged
    2. else: this sample's OWN fastballs             ← self-anchoring
    3. else: league-average fastball, by handedness  ← cold start

Only pitchers ABSENT from the stored file are ever synthesized, so the grades
of every pitcher who already has a baseline are byte-for-byte unchanged.

The synthesized record carries the same keys build_pitcher_baselines.py writes
(fb_velo, fb_pfx_x/z, fb_spin, fb_extension, fb_vaa, fb_release_x/z, and the
fb_tunnel_x/z + fb_plate_x/z anchors) so the existing delta/tunnel code reads
it with no special-casing.
"""
import numpy as np
import pandas as pd

# Same fastball family build_pitcher_baselines.py averages over.
FB_TYPES = {'FF', 'SI', 'FC'}

_STUFF_FIELDS  = ['fb_velo', 'fb_pfx_x', 'fb_pfx_z', 'fb_spin',
                  'fb_extension', 'fb_vaa', 'fb_release_x', 'fb_release_z']
_TUNNEL_FIELDS = ['fb_tunnel_x', 'fb_tunnel_z', 'fb_plate_x', 'fb_plate_z']
_ALL_FIELDS    = _STUFF_FIELDS + _TUNNEL_FIELDS

# Synthesized stuff field → the df column it's averaged from (tunnel anchors are
# left at the league value; a thin in-sample fastball set is a poor anchor).
_SAMPLE_SRC = {
    'fb_velo':      'release_speed',
    'fb_pfx_x':     'pfx_x',
    'fb_pfx_z':     'pfx_z',
    'fb_spin':      'release_spin_rate',
    'fb_extension': 'release_extension',
    'fb_vaa':       'vaa',
    'fb_release_x': 'release_pos_x',
    'fb_release_z': 'release_pos_z',
}


def _hand_of(rec: dict) -> str:
    """Handedness of a stored baseline. Most records predate the p_throws field,
    so fall back to the release-side sign (RHP release from the 3B side → x<0,
    LHP from the 1B side → x>0), which is a reliable proxy for a fastball."""
    pt = rec.get('p_throws')
    if pt in ('L', 'R'):
        return pt
    rx = rec.get('fb_release_x')
    if isinstance(rx, (int, float)):
        return 'L' if rx > 0 else 'R'
    return 'R'


def league_fb_baseline(baselines: dict) -> dict:
    """Mean of every fb_* field across stored baselines, split by handedness.

    Per-hand because pfx_x / release_x flip sign for LHP vs RHP — a pooled mean
    would cancel toward zero. Returns {'L': {...}, 'R': {...}}; 'R' doubles as
    the catch-all when a pitcher's handedness is unknown.
    """
    out = {}
    for hand in ('L', 'R'):
        recs = [d for d in baselines.values() if _hand_of(d) == hand]
        if not recs:
            recs = list(baselines.values())
        out[hand] = {}
        for f in _ALL_FIELDS:
            vals = [r[f] for r in recs if isinstance(r.get(f), (int, float))]
            out[hand][f] = float(np.mean(vals)) if vals else 0.0
    return out


def _sample_mean(series):
    v = pd.to_numeric(series, errors='coerce').mean()
    return float(v) if pd.notna(v) else None


def effective_baselines(df: pd.DataFrame, baselines: dict, league: dict) -> dict:
    """Return {pid_str: baseline_record} for every pitcher present in df.

    Stored record if present (returned unchanged); otherwise the league-average
    record for the pitcher's hand, with the STUFF fields overridden by that
    pitcher's own fastballs in THIS sample when they threw any. df should carry
    a `vaa` column for fb_vaa; if absent, fb_vaa stays at the league value.
    """
    eff = {}
    has_throws = 'p_throws' in df.columns
    has_ptype  = 'pitch_type' in df.columns
    for pid, grp in df.groupby('pitcher'):
        if pd.isna(pid):
            continue
        pid_s = str(int(pid))
        stored = baselines.get(pid_s)
        if stored is not None:
            eff[pid_s] = stored                      # untouched
            continue
        hand = grp['p_throws'].iloc[0] if has_throws else 'R'
        if hand not in ('L', 'R'):
            hand = 'R'
        rec = dict(league.get(hand) or league.get('R') or {})   # cold start
        fb = grp[grp['pitch_type'].isin(FB_TYPES)] if has_ptype else grp.iloc[0:0]
        if len(fb):                                  # self-anchor on own fastballs
            primary = fb['pitch_type'].value_counts().idxmax()
            sub = fb[fb['pitch_type'] == primary]
            for field, col in _SAMPLE_SRC.items():
                if col in sub.columns:
                    m = _sample_mean(sub[col])
                    if m is not None:
                        rec[field] = m
        rec['_synthesized'] = True
        eff[pid_s] = rec
    return eff
