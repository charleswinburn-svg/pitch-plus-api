#!/usr/bin/env python3
"""
score_pitches.py — Production scoring script for contextual pitch grading

Takes a Statcast parquet (pitch_xrv format), runs the three-stage model
(Stuff + Location + Tunnel), applies learned ridge weights, and outputs:

1. Per-game pitch grades:  games/pitch_grades_{game_pk}.json
     keyed by "{gamePk}_{atBatIndex}_{pitchNumber}" for O(1) frontend lookup
     (atBatIndex = at_bat_number - 1, matching MLB Stats API's 0-indexed format)

2. Season pitcher aggregates: season/pitcher_grades_{season}.json
     rolling averages per pitcher and per (pitcher, pitch_type) from all
     scored pitches in the season so far

Usage:
    python score_pitches.py --input /path/to/pitch_xrv_2025.parquet \\
                             --models /path/to/models \\
                             --config /path/to/final_model_config.json \\
                             --output-dir ./output

The script runs the same feature engineering as the training notebooks and
is the single source of truth for nightly scoring. Drop it in your repo and
point a GitHub Actions cron job at it.
"""

import argparse
import json
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb


NUMERIC_COLS = ['release_speed','pfx_x','pfx_z','release_spin_rate','spin_axis',
    'release_extension','release_pos_x','release_pos_z','vx0','vy0','vz0',
    'ax','ay','az','plate_x','plate_z','sz_top','sz_bot']

def _to_float64(df):
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    return df



# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — mirrors the training notebooks exactly
# ═══════════════════════════════════════════════════════════════════════════

def engineer_stuff_features(df):
    """v3 stuff features — matches stuff_model_v3 training notebook."""
    df = df.copy()

    # Cast nullable dtypes
    for col in ['release_spin_rate','release_pos_x','release_pos_z',
                'pfx_x','pfx_z','release_speed','release_extension',
                'spin_axis','arm_angle','ax','ay','az','vx0','vy0','vz0']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

    # ── Handedness-corrected horizontal break ──
    hand_sign = np.where(df['p_throws'] == 'L', -1.0, 1.0).astype('float32')
    df['arm_side_break'] = (df['pfx_x'].values * hand_sign).astype('float32')

    # ── Total movement & spin efficiency ──
    df['total_movement'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    spin = df['release_spin_rate'].values
    df['spin_efficiency'] = np.where(spin > 0, df['total_movement'] / (spin / 1000), np.nan)

    # ── Spin axis circular encoding ──
    if 'spin_axis' in df.columns:
        axis_rad = np.deg2rad(df['spin_axis'].values)
        df['spin_axis_sin'] = np.sin(axis_rad).astype('float32')
        df['spin_axis_cos'] = np.cos(axis_rad).astype('float32')

    # ── Release tilt ──
    if 'vy0' in df.columns and 'vz0' in df.columns:
        df['release_tilt'] = np.arctan2(df['vz0'].values, -df['vy0'].values).astype('float32')

    # VAA/HAA still useful for tunnel features (computed below in their own function)
    if all(c in df.columns for c in ['vy0','vz0','az','ay','vx0','ax']):
        t = np.clip(50.0 / (-df['vy0'].values), 0.35, 0.55)
        df['vaa'] = np.degrees(np.arctan2(df['vz0'].values + df['az'].values * t,
                                           -(df['vy0'].values + df['ay'].values * t)))
        df['haa'] = np.degrees(np.arctan2(df['vx0'].values + df['ax'].values * t,
                                           -(df['vy0'].values + df['ay'].values * t)))

    # ── Fastball baselines ──
    baselines_path = Path(__file__).parent / 'models' / 'pitcher_baselines.json'
    with open(baselines_path) as _f:
        _baselines = json.load(_f)

    for col in ['fb_velo','fb_pfx_x','fb_pfx_z','fb_spin','fb_extension','fb_vaa',
                'fb_rel_x','fb_rel_z']:
        df[col] = np.nan
    for pid, group in df.groupby('pitcher'):
        pid_s = str(int(pid)) if pd.notna(pid) else None
        bl = _baselines.get(pid_s) if pid_s else None
        if not bl: continue
        mask = df['pitcher'] == pid
        df.loc[mask, 'fb_velo'] = bl['fb_velo']
        df.loc[mask, 'fb_pfx_x'] = bl['fb_pfx_x']
        df.loc[mask, 'fb_pfx_z'] = bl['fb_pfx_z']
        df.loc[mask, 'fb_spin'] = bl['fb_spin']
        df.loc[mask, 'fb_extension'] = bl['fb_extension']
        df.loc[mask, 'fb_vaa'] = bl.get('fb_vaa', np.nan)
        df.loc[mask, 'fb_rel_x'] = bl['fb_release_x']
        df.loc[mask, 'fb_rel_z'] = bl['fb_release_z']

    df['delta_velo']          = df['release_speed'] - df['fb_velo']
    df['delta_pfx_x']         = df['pfx_x'] - df['fb_pfx_x']
    df['delta_pfx_z']         = df['pfx_z'] - df['fb_pfx_z']
    df['delta_spin']          = df['release_spin_rate'] - df['fb_spin']
    df['delta_extension']     = df['release_extension'] - df['fb_extension']
    df['delta_vaa']           = df['vaa'] - df['fb_vaa'] if 'vaa' in df.columns else 0.0
    df['movement_separation'] = np.sqrt(df['delta_pfx_x']**2 + df['delta_pfx_z']**2)
    df['release_diff_x']      = df['release_pos_x'] - df['fb_rel_x']
    df['release_diff_z']      = df['release_pos_z'] - df['fb_rel_z']
    df['release_distance']    = np.sqrt(df['release_diff_x']**2 + df['release_diff_z']**2)
    df = df.fillna({c: 0.0 for c in ['delta_velo','delta_pfx_x','delta_pfx_z','delta_spin',
        'delta_extension','delta_vaa','release_diff_x','release_diff_z','release_distance',
        'movement_separation']})

    # ── Categoricals + platoon ──
    df['pitch_type_cat'] = df['pitch_type'].astype('category')
    df['p_throws_cat']   = df['p_throws'].astype('category')
    if 'stand' in df.columns:
        df['stand_cat']  = df['stand'].astype('category')
        df['same_side']  = (df['p_throws'] == df['stand']).astype('int8')

    # ── Arm angle ──
    arm_angles_path = Path(__file__).parent / 'models' / 'pitcher_arm_angles.json'
    slot_reg_path   = Path(__file__).parent / 'models' / 'slot_regression.json'
    heights_path    = Path(__file__).parent / 'models' / 'pitcher_heights.json'
    _arm_angles = {}
    _slot_reg = {}
    _heights = {}
    if arm_angles_path.exists():
        with open(arm_angles_path) as _f:
            _arm_angles = json.load(_f)
    if slot_reg_path.exists():
        with open(slot_reg_path) as _f:
            _slot_reg = json.load(_f)
    if heights_path.exists():
        with open(heights_path) as _f:
            _heights = json.load(_f)

    df['arm_angle_baseline'] = np.nan
    df['arm_angle_std'] = np.nan
    df['arm_angle_dev'] = np.nan

    has_per_pitch_aa = 'arm_angle' in df.columns and pd.to_numeric(df['arm_angle'], errors='coerce').notna().any()
    if has_per_pitch_aa:
        per_pitch_aa = pd.to_numeric(df['arm_angle'], errors='coerce')

    for pid, group in df.groupby('pitcher'):
        pid_s = str(int(pid)) if pd.notna(pid) else None
        aa = _arm_angles.get(pid_s) if pid_s else None
        if not aa:
            continue
        mask = df['pitcher'] == pid
        df.loc[mask, 'arm_angle_baseline'] = aa['arm_angle']
        df.loc[mask, 'arm_angle_std'] = aa['arm_angle_std']
        if has_per_pitch_aa:
            df.loc[mask, 'arm_angle_dev'] = (per_pitch_aa[mask] - aa['arm_angle']).fillna(0)
        else:
            df.loc[mask, 'arm_angle_dev'] = 0.0

    # arm_angle: prefer per-pitch, fall back to baseline
    if has_per_pitch_aa:
        df['arm_angle'] = per_pitch_aa.fillna(df['arm_angle_baseline'])
    else:
        df['arm_angle'] = df['arm_angle_baseline']
    df = df.drop(columns=['arm_angle_baseline'])

    # ── Geometric arm-angle estimation for WBC/AAA pitchers with no measured value ──
    # release_pos_x and release_pos_z are present in every Statcast fetch.
    # Uses cached pitcher height (inches) from pitcher_heights.json; falls back
    # to the MLB average (74") so WBC/AAA pitchers always get a reasonable estimate.
    _nan_aa = df['arm_angle'].isna()
    if _nan_aa.any():
        _DEFAULT_H = 74.0
        _rel_x = pd.to_numeric(df.loc[_nan_aa, 'release_pos_x'], errors='coerce').values
        _rel_z = pd.to_numeric(df.loc[_nan_aa, 'release_pos_z'], errors='coerce').values
        _throws = df.loc[_nan_aa, 'p_throws'].values if 'p_throws' in df.columns else np.full(_nan_aa.sum(), 'R')
        _h = np.array([
            _heights.get(str(int(pid)), _DEFAULT_H) if pd.notna(pid) else _DEFAULT_H
            for pid in df.loc[_nan_aa, 'pitcher']
        ], dtype='float32')
        _shoulder_z = _h * 0.70
        _adj = (_rel_z * 12.0) - _shoulder_z
        _opp = np.abs(_rel_x * 12.0)
        _est = np.degrees(np.arctan2(_opp, _adj))
        _est = np.where(_throws == 'L', -_est, _est)
        df.loc[_nan_aa, 'arm_angle'] = _est.astype('float32')
        df.loc[_nan_aa & df['arm_angle_dev'].isna(), 'arm_angle_dev'] = 0.0

    # ── v3 SLOT REGRESSION: per (pitch_type, p_throws) ──
    df['pfx_x_dev_from_slot'] = np.nan
    df['pfx_z_dev_from_slot'] = np.nan
    slot_coefs = _slot_reg.get('slot', _slot_reg) if _slot_reg else {}
    if slot_coefs:
        for pt in df['pitch_type'].dropna().unique():
            for hand in ['L', 'R']:
                key = f"{pt}_{hand}"
                if key not in slot_coefs:
                    continue
                c = slot_coefs[key]
                mask = (df['pitch_type'] == pt) & (df['p_throws'] == hand) & df['arm_angle'].notna()
                if mask.sum() == 0:
                    continue
                arm = df.loc[mask, 'arm_angle']
                exp_x = c['slope_x'] * arm + c['intercept_x']
                exp_z = c['slope_z'] * arm + c['intercept_z']
                df.loc[mask, 'pfx_x_dev_from_slot'] = (df.loc[mask, 'arm_side_break'] - exp_x).astype(float)
                df.loc[mask, 'pfx_z_dev_from_slot'] = (df.loc[mask, 'pfx_z'] - exp_z).astype(float)

    # ── v3 DRAG RESIDUAL: per pitch_type ──
    df['ay_residual'] = 0.0
    drag_coefs = _slot_reg.get('drag', {}) if _slot_reg else {}
    if drag_coefs and 'ay' in df.columns:
        for pt, c in drag_coefs.items():
            mask = df['pitch_type'] == pt
            if mask.sum() == 0:
                continue
            expected_ay = (c['intercept']
                           + c['slope_velo'] * df.loc[mask, 'release_speed']
                           + c['slope_spin'] * df.loc[mask, 'release_spin_rate'])
            df.loc[mask, 'ay_residual'] = (df.loc[mask, 'ay'] - expected_ay).astype(float)

    return df


def engineer_tunnel_features(df):
    """Trajectory reconstruction + tunnel-point divergence features."""
    df = df.copy()
    PLATE_Y = 17.0 / 12.0
    y_tun = PLATE_Y + 23.0
    y0 = np.full(len(df), 50.0)

    a_ = 0.5 * df['ay'].values
    b_ = df['vy0'].values
    c_ = y0 - y_tun
    with np.errstate(invalid='ignore', divide='ignore'):
        disc = b_**2 - 4 * a_ * c_
        t_tun = np.where(
            disc >= 0,
            (-b_ - np.sqrt(np.maximum(disc, 0))) / (2 * a_),
            np.nan,
        )
        t_tun = np.where((t_tun > 0) & (t_tun < 0.5), t_tun, np.nan)

    df['tunnel_x'] = (df['release_pos_x'].values +
                      df['vx0'].values * t_tun +
                      0.5 * df['ax'].values * t_tun**2)
    df['tunnel_z'] = (df['release_pos_z'].values +
                      df['vz0'].values * t_tun +
                      0.5 * df['az'].values * t_tun**2)

    c_p = y0 - PLATE_Y
    disc_p = b_**2 - 4 * a_ * c_p
    t_p = np.where(
        disc_p >= 0,
        (-b_ - np.sqrt(np.maximum(disc_p, 0))) / (2 * a_),
        np.nan,
    )
    df['time_after_tunnel'] = t_p - t_tun

    # FB tunnel anchors from baseline file (same source as stuff features)
    _baselines_path = Path(__file__).parent / 'models' / 'pitcher_baselines.json'
    with open(_baselines_path) as _f:
        _baselines = json.load(_f)
    for col in ['fb_tunnel_x','fb_tunnel_z','fb_plate_x','fb_plate_z']:
        df[col] = np.nan
    for pid, _ in df.groupby('pitcher'):
        pid_s = str(int(pid)) if pd.notna(pid) else None
        bl = _baselines.get(pid_s) if pid_s else None
        if not bl: continue
        mask = df['pitcher'] == pid
        df.loc[mask, 'fb_tunnel_x'] = bl.get('fb_tunnel_x', np.nan)
        df.loc[mask, 'fb_tunnel_z'] = bl.get('fb_tunnel_z', np.nan)
        df.loc[mask, 'fb_plate_x'] = bl.get('fb_plate_x', np.nan)
        df.loc[mask, 'fb_plate_z'] = bl.get('fb_plate_z', np.nan)

    df['tunnel_diff_x']   = df['tunnel_x'] - df['fb_tunnel_x']
    df['tunnel_diff_z']   = df['tunnel_z'] - df['fb_tunnel_z']
    df['tunnel_distance'] = np.sqrt(df['tunnel_diff_x']**2 + df['tunnel_diff_z']**2) * 12
    df['plate_distance']  = np.sqrt(
        (df['plate_x'] - df['fb_plate_x'])**2 +
        (df['plate_z'] - df['fb_plate_z'])**2
    ) * 12
    df['late_break'] = df['plate_distance'] - df['tunnel_distance']
    df = df.fillna({c: 0.0 for c in ['tunnel_diff_x','tunnel_diff_z','tunnel_distance',
                                       'plate_distance','late_break']})
    return df


def engineer_location_features(df):
    """Per-pitch location features (handedness-aware)."""
    df = df.copy()
    df['plate_x_adj'] = np.where(df['stand'] == 'L', -df['plate_x'], df['plate_x'])
    df['zone_center_dist'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - 2.5)**2)
    x_e = np.maximum(np.abs(df['plate_x']) - 0.83, 0)
    z_t = np.maximum(df['plate_z'] - 3.5, 0)
    z_b = np.maximum(1.5 - df['plate_z'], 0)
    df['out_of_zone_dist'] = np.sqrt(x_e**2 + np.maximum(z_t, z_b)**2)
    df['in_zone'] = (
        (df['plate_x'].abs() <= 0.83) &
        (df['plate_z'] >= 1.5) & (df['plate_z'] <= 3.5)
    ).astype(np.int8)
    df['same_side'] = (df['p_throws'] == df['stand']).astype(np.int8)
    df['stand_cat'] = df['stand'].astype('category')
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_models(model_dir: Path):
    with open(model_dir / 'stuff_model_metadata.json') as f:
        stuff_meta = json.load(f)
    stuff_fb = lgb.Booster(model_file=str(model_dir / stuff_meta['fb_model']['file']))
    stuff_offspeed = lgb.Booster(model_file=str(model_dir / stuff_meta['offspeed_model']['file']))
    stuff_fb_features = stuff_meta['fb_model']['features']
    stuff_offspeed_features = stuff_meta['offspeed_model']['features']
    stuff_family = stuff_meta['family_definition']

    tunnel = lgb.Booster(model_file=str(model_dir / 'tunnel_model_2025.txt'))

    location_models = {}
    for f in sorted((model_dir).glob('location_model_*_2025.txt')):
        pt = f.stem.split('_')[2]  # location_model_FF_2025 → FF
        location_models[pt] = lgb.Booster(model_file=str(f))

    return stuff_fb, stuff_fb_features, stuff_offspeed, stuff_offspeed_features, stuff_family, tunnel, location_models


# ═══════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════

TUNNEL_FEATURES = [
    'tunnel_diff_x', 'tunnel_diff_z', 'tunnel_distance',
    'time_after_tunnel', 'late_break', 'plate_distance',
    'release_diff_x', 'release_diff_z', 'release_distance',
    'pitch_type_cat', 'p_throws_cat', 'release_speed', 'delta_velo',
]
LOCATION_FEATURES = [
    'plate_x', 'plate_z', 'plate_x_adj',
    'zone_center_dist', 'out_of_zone_dist', 'in_zone',
    'pfx_x', 'pfx_z', 'release_speed', 'stand_cat', 'same_side',
]


def score_dataframe(df, stuff_fb, stuff_fb_features, stuff_offspeed, stuff_offspeed_features, stuff_family, tunnel, location_models, weights):
    """Run all three model stages on a dataframe, return df with prediction cols."""
    df = engineer_stuff_features(df)
    df = engineer_tunnel_features(df)
    df = engineer_location_features(df)

    # Stuff: route FB vs offspeed by velocity proximity to fastball baseline
    mph = stuff_family['mph_threshold']
    fb_types = set(stuff_family['fastball_types'])
    fb_mask = (
        df['fb_velo'].notna() & (np.abs(df['release_speed'] - df['fb_velo']) <= mph)
    ) | (df['fb_velo'].isna() & df['pitch_type'].isin(fb_types))
    os_mask = ~fb_mask
    df['xRV_stuff'] = np.float32(0.0)
    if fb_mask.any():
        df.loc[fb_mask, 'xRV_stuff'] = stuff_fb.predict(df.loc[fb_mask, stuff_fb_features]).astype('float32')
    if os_mask.any():
        df.loc[os_mask, 'xRV_stuff'] = stuff_offspeed.predict(df.loc[os_mask, stuff_offspeed_features]).astype('float32')

    # Tunnel
    df['xRV_tunnel'] = np.float32(0.0)
    valid = df[TUNNEL_FEATURES].notna().all(axis=1)
    if valid.any():
        df.loc[valid, 'xRV_tunnel'] = tunnel.predict(
            df.loc[valid, TUNNEL_FEATURES]).astype('float32')

    # Location — route each pitch to its per-type model
    df['xRV_location'] = np.float32(0.0)
    for pt, lmodel in location_models.items():
        mask = (df['pitch_type'] == pt).values
        if mask.sum() == 0:
            continue
        df.loc[mask, 'xRV_location'] = lmodel.predict(
            df.loc[mask, LOCATION_FEATURES]).astype('float32')

    # Weighted combiner
    w = weights['weights']
    df['xRV_final'] = (
        w['stuff']    * df['xRV_stuff'] +
        w['location'] * df['xRV_location'] +
        w['tunnel']   * df['xRV_tunnel'] +
        weights['intercept']
    ).astype('float32')

    return df


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT: PER-GAME JSON + SEASON AGGREGATES
# ═══════════════════════════════════════════════════════════════════════════

def write_per_game_json(df, output_dir: Path, compress=True):
    """
    Write one JSON per game, keyed by "{gamePk}_{atBatIndex}_{pitchNumber}".

    atBatIndex = at_bat_number - 1 to match MLB Stats API's 0-indexed format.
    """
    games_dir = output_dir / 'games'
    games_dir.mkdir(parents=True, exist_ok=True)

    # Round predictions for smaller files
    df = df.copy()
    for col in ['xRV_final', 'xRV_stuff', 'xRV_location', 'xRV_tunnel']:
        df[col] = df[col].round(5)

    written = 0
    for game_pk, game_df in df.groupby('game_pk'):
        grades = {}
        for _, row in game_df.iterrows():
            # atBatIndex is 0-indexed in MLB Stats API; at_bat_number is 1-indexed in Statcast
            key = f"{int(row['game_pk'])}_{int(row['at_bat_number']) - 1}_{int(row['pitch_number'])}"
            grades[key] = {
                'xRV':   float(row['xRV_final']),
                'stuff': float(row['xRV_stuff']),
                'loc':   float(row['xRV_location']),
                'tun':   float(row['xRV_tunnel']),
            }

        out_path = games_dir / f'pitch_grades_{int(game_pk)}.json'
        if compress:
            out_path = out_path.with_suffix('.json.gz')
            with gzip.open(out_path, 'wt', encoding='utf-8') as f:
                json.dump(grades, f, separators=(',', ':'))
        else:
            with open(out_path, 'w') as f:
                json.dump(grades, f, separators=(',', ':'))
        written += 1

    print(f'  Wrote {written} per-game files to {games_dir}/')
    return written


def write_season_aggregates(df, output_dir: Path, season: int, norm_path: Path = None):
    """
    Build season-level pitcher aggregates with correct Pitch+ scaling.

    Bug fix vs. the previous version: the pitcher-overall Plus values
    were computed by z-scoring the pitcher's mean xRV against per-pitch
    league norms. That's wrong — per-pitch std is much larger than the
    spread of pitcher averages, so it compressed every pitcher toward 100.

    Correct approach (matches what /score does for the Summary view):
      1. Compute per-(pitcher, type) Plus values from per-pitch norms.
      2. For pitcher overall, weight-average those per-type Plus values
         by pitch-type usage.

    Algebraically these match what you'd get if you graded every individual
    pitch then averaged: avg(100 - 10z) = 100 - 10*avg(z).
    """
    season_dir = output_dir / 'season'
    season_dir.mkdir(parents=True, exist_ok=True)

    pitch_plus_norm = None
    if norm_path and Path(norm_path).exists():
        with open(norm_path) as f:
            pitch_plus_norm = json.load(f)

    def _per_type_plus(xrv_per100, pitch_type, kind):
        """Grade ONE pitch type for ONE pitcher. xrv_per100 is xRV*100."""
        if pitch_plus_norm is None or pd.isna(xrv_per100):
            return None
        mean_key = {'stuff': 'stuff_mean', 'loc': 'loc_mean',
                    'tun': 'tun_mean',     'pitch': 'mean'}[kind]
        std_key  = {'stuff': 'stuff_std',  'loc': 'loc_std',
                    'tun': 'tun_std',      'pitch': 'std'}[kind]
        n = pitch_plus_norm.get(pitch_type)
        if not n or std_key not in n or n[std_key] <= 0:
            return None
        # Undo the *100 from the agg step to get raw xRV.
        z = max(-4, min(4, (xrv_per100 / 100 - n[mean_key]) / n[std_key]))
        return round(100 - z * 10, 1)

    # ── Pitcher × pitch_type aggregation (compute per-type Plus values first) ──
    pt_agg = df.groupby(['pitcher', 'pitch_type']).agg(
        n=('xRV_final', 'count'),
        xRV=('xRV_final', 'mean'),
        stuff=('xRV_stuff', 'mean'),
        loc=('xRV_location', 'mean'),
        tun=('xRV_tunnel', 'mean'),
    ).reset_index()
    for col in ['xRV', 'stuff', 'loc', 'tun']:
        pt_agg[col] = (pt_agg[col] * 100).round(3)

    pt_out = {}
    for _, row in pt_agg.iterrows():
        pid = str(int(row['pitcher']))
        pt = row['pitch_type']
        pt_out.setdefault(pid, {})[pt] = {
            'n':           int(row['n']),
            'xRV':         float(row['xRV']),
            'stuff':       float(row['stuff']),
            'loc':         float(row['loc']),
            'tun':         float(row['tun']),
            'stuff_plus':  _per_type_plus(row['stuff'], pt, 'stuff'),
            'loc_plus':    _per_type_plus(row['loc'],   pt, 'loc'),
            'tun_plus':    _per_type_plus(row['tun'],   pt, 'tun'),
            'pitch_plus':  _per_type_plus(row['xRV'],   pt, 'pitch'),
        }

    # ── Pitcher overall — weight-average the per-type Plus values by usage ──
    def _weighted_overall(pid_pt_grades, metric_key):
        total = 0
        weighted_sum = 0.0
        for pt, g in pid_pt_grades.items():
            v = g.get(metric_key)
            n = g.get('n', 0)
            if v is not None and n > 0:
                weighted_sum += v * n
                total += n
        return round(weighted_sum / total, 1) if total else None

    pitcher_agg = df.groupby('pitcher').agg(
        n=('xRV_final', 'count'),
        xRV=('xRV_final', 'mean'),
        stuff=('xRV_stuff', 'mean'),
        loc=('xRV_location', 'mean'),
        tun=('xRV_tunnel', 'mean'),
    )
    for col in ['xRV', 'stuff', 'loc', 'tun']:
        pitcher_agg[col] = (pitcher_agg[col] * 100).round(3)

    pitcher_out = {}
    for pid, row in pitcher_agg.iterrows():
        pid_str = str(int(pid))
        types = pt_out.get(pid_str, {})
        pitcher_out[pid_str] = {
            'n':           int(row['n']),
            'xRV':         float(row['xRV']),
            'stuff':       float(row['stuff']),
            'loc':         float(row['loc']),
            'tun':         float(row['tun']),
            'stuff_plus':  _weighted_overall(types, 'stuff_plus'),
            'loc_plus':    _weighted_overall(types, 'loc_plus'),
            'tun_plus':    _weighted_overall(types, 'tun_plus'),
            'pitch_plus':  _weighted_overall(types, 'pitch_plus'),
        }

    pitcher_path = season_dir / f'pitcher_grades_{season}.json'
    pt_path      = season_dir / f'pitcher_pitch_type_grades_{season}.json'

    with open(pitcher_path, 'w') as f:
        json.dump(pitcher_out, f, separators=(',', ':'))
    with open(pt_path, 'w') as f:
        json.dump(pt_out, f, separators=(',', ':'))

    print(f'  Season pitcher aggregates: {len(pitcher_out)} pitchers → {pitcher_path}')
    print(f'  Season pitch-type aggregates: {sum(len(v) for v in pt_out.values())} rows → {pt_path}')




# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Score pitches with the 3-stage model')
    parser.add_argument('--input', required=True, help='Path to Statcast parquet file')
    parser.add_argument('--models', required=True, help='Directory with saved models')
    parser.add_argument('--config', required=True, help='Path to final_model_config.json')
    parser.add_argument('--output-dir', required=True, help='Where to write JSON output')
    parser.add_argument('--season', type=int, default=None,
                        help='Season to tag aggregates (default: infer from data)')
    parser.add_argument('--no-compress', action='store_true',
                        help='Skip gzip compression of per-game files')
    args = parser.parse_args()

    input_path  = Path(args.input)
    model_dir   = Path(args.models)
    output_dir  = Path(args.output_dir)
    config_path = Path(args.config)

    print(f'Loading models from {model_dir}...')
    stuff_fb, stuff_fb_features, stuff_offspeed, stuff_offspeed_features, stuff_family, tunnel, location_models = load_models(model_dir)
    print(f'  Stuff FB: {len(stuff_fb_features)} features, Offspeed: {len(stuff_offspeed_features)} features')
    print(f'  Location: {len(location_models)} per-pitch-type models')
    print(f'  Tunnel: loaded')

    print(f'Loading weights from {config_path}...')
    with open(config_path) as f:
        weights = json.load(f)
    print(f'  Weights: stuff={weights["weights"]["stuff"]:.4f}, '
          f'loc={weights["weights"]["location"]:.4f}, '
          f'tun={weights["weights"]["tunnel"]:.4f}, '
          f'intercept={weights["intercept"]:.5f}')

    print(f'Loading data from {input_path}...')
    df = pd.read_parquet(input_path)

    # Ensure required columns exist
    required = [
        'pitch_type', 'pitcher', 'game_pk', 'at_bat_number', 'pitch_number',
        'stand', 'p_throws', 'plate_x', 'plate_z',
        'release_speed', 'release_spin_rate', 'release_extension',
        'pfx_x', 'pfx_z', 'spin_axis',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing required columns: {missing}')

    # Dedup on natural pitch ID
    df = df.drop_duplicates(
        subset=['game_pk', 'at_bat_number', 'pitch_number'], keep='first'
    ).reset_index(drop=True)
    print(f'  Loaded {len(df):,} unique pitches')

    # Infer season if not provided
    season = args.season if args.season else int(df['season'].iloc[0])

    print('Running models...')
    df = _to_float64(df)
    df = score_dataframe(df, stuff_fb, stuff_fb_features, stuff_offspeed, stuff_offspeed_features, stuff_family, tunnel, location_models, weights)
    print(f'  Scored {len(df):,} pitches')
    print(f'  Mean predicted xRV: {df["xRV_final"].mean():.5f}')
    print(f'  Mean actual xRV:    {df["xRV"].mean():.5f}' if 'xRV' in df.columns else '')

    print('Writing per-game JSON files...')
    write_per_game_json(df, output_dir, compress=not args.no_compress)

    print(f'Writing season {season} aggregates...')
    write_season_aggregates(df, output_dir, season, model_dir / "pitch_plus_norm.json")

    print('Done.')


if __name__ == '__main__':
    main()
