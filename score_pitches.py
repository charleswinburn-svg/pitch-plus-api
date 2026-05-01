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
    """Build the 24-feature stuff set used by the Stuff model."""
    df = df.copy()

    t = np.clip(50.0 / (-df['vy0'].values), 0.35, 0.55)
    df['vaa'] = np.degrees(np.arctan2(
        df['vz0'].values + df['az'].values * t,
        -(df['vy0'].values + df['ay'].values * t)))
    df['haa'] = np.degrees(np.arctan2(
        df['vx0'].values + df['ax'].values * t,
        -(df['vy0'].values + df['ay'].values * t)))
    df['total_movement'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    spin = df['release_spin_rate'].astype('float32').values
    df['spin_efficiency'] = np.where(
        spin > 0, df['total_movement'] / (spin / 1000), np.nan)

    # Fastball baseline per pitcher — loaded from models/pitcher_baselines.json
    # (trailing-window averages, built separately by build_pitcher_baselines.py).
    # Falls back to league-average FF for cold-start pitchers.
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
        df.loc[mask, 'fb_vaa'] = bl['fb_vaa']
        df.loc[mask, 'fb_rel_x'] = bl['fb_release_x']
        df.loc[mask, 'fb_rel_z'] = bl['fb_release_z']

    df['delta_velo']          = df['release_speed'] - df['fb_velo']
    df['delta_pfx_x']         = df['pfx_x'] - df['fb_pfx_x']
    df['delta_pfx_z']         = df['pfx_z'] - df['fb_pfx_z']
    df['delta_spin']          = df['release_spin_rate'] - df['fb_spin']
    df['delta_extension']     = df['release_extension'] - df['fb_extension']
    df['delta_vaa']           = df['vaa'] - df['fb_vaa']
    df['movement_separation'] = np.sqrt(df['delta_pfx_x']**2 + df['delta_pfx_z']**2)
    df['release_diff_x']      = df['release_pos_x'] - df['fb_rel_x']
    df['release_diff_z']      = df['release_pos_z'] - df['fb_rel_z']
    df['release_distance']    = np.sqrt(df['release_diff_x']**2 + df['release_diff_z']**2)
    df = df.fillna({c: 0.0 for c in ['delta_velo','delta_pfx_x','delta_pfx_z','delta_spin',
        'delta_extension','delta_vaa','release_diff_x','release_diff_z','release_distance',
        'movement_separation']})
    df['pitch_type_cat']      = df['pitch_type'].astype('category')
    df['p_throws_cat']        = df['p_throws'].astype('category')

    # ── Arm angle features ──
    # If the parquet has arm_angle (from Statcast), use it directly
    if 'arm_angle' in df.columns:
        df['arm_angle'] = pd.to_numeric(df['arm_angle'], errors='coerce')
        # Per-pitcher stats
        aa_stats = df.groupby('pitcher')['arm_angle'].agg(['mean', 'std']).reset_index()
        aa_stats.columns = ['pitcher', '_aa_mean', '_aa_std']
        aa_stats['_aa_std'] = aa_stats['_aa_std'].fillna(0)
        df = df.merge(aa_stats, on='pitcher', how='left')
        df['arm_angle_std'] = df['_aa_std'].fillna(0)
        df['arm_angle_dev'] = (df['arm_angle'] - df['_aa_mean']).fillna(0)
        df['arm_angle'] = df['arm_angle'].fillna(df['_aa_mean']).fillna(0)
        # pfx deviation from slot: actual pfx minus pitcher's mean pfx
        pfx_stats = df.groupby('pitcher')[['pfx_x', 'pfx_z']].mean().reset_index()
        pfx_stats.columns = ['pitcher', '_pfx_x_slot', '_pfx_z_slot']
        df = df.merge(pfx_stats, on='pitcher', how='left')
        df['pfx_x_dev_from_slot'] = (df['pfx_x'] - df['_pfx_x_slot']).fillna(0)
        df['pfx_z_dev_from_slot'] = (df['pfx_z'] - df['_pfx_z_slot']).fillna(0)
        df = df.drop(columns=['_aa_mean', '_aa_std', '_pfx_x_slot', '_pfx_z_slot'], errors='ignore')
    else:
        # Fall back to pre-computed baselines
        arm_angles_path = Path(__file__).parent / 'models' / 'pitcher_arm_angles.json'
        _arm_angles = {}
        if arm_angles_path.exists():
            with open(arm_angles_path) as _f:
                _arm_angles = json.load(_f)

        # NaN defaults — LightGBM treats missing values via learned default branches,
        # effectively neutralizing the 5 AA features. The other 20 features carry the prediction.
        df['arm_angle'] = np.nan
        df['arm_angle_std'] = np.nan
        df['arm_angle_dev'] = np.nan
        df['pfx_x_dev_from_slot'] = np.nan
        df['pfx_z_dev_from_slot'] = np.nan

        for pid, group in df.groupby('pitcher'):
            pid_s = str(int(pid)) if pd.notna(pid) else None
            aa = _arm_angles.get(pid_s) if pid_s else None
            if not aa:
                continue  # keep NaN defaults
            mask = df['pitcher'] == pid
            df.loc[mask, 'arm_angle'] = aa['arm_angle']
            df.loc[mask, 'arm_angle_std'] = aa['arm_angle_std']
            df.loc[mask, 'arm_angle_dev'] = 0.0
            df.loc[mask, 'pfx_x_dev_from_slot'] = df.loc[mask, 'pfx_x'] - aa['pfx_x_slot']
            df.loc[mask, 'pfx_z_dev_from_slot'] = df.loc[mask, 'pfx_z'] - aa['pfx_z_slot']

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
    stuff = lgb.Booster(model_file=str(model_dir / 'stuff_model_2025.txt'))
    with open(model_dir / 'stuff_model_metadata.json') as f:
        stuff_features = json.load(f)['features']

    tunnel = lgb.Booster(model_file=str(model_dir / 'tunnel_model_2025.txt'))

    location_models = {}
    for f in sorted((model_dir).glob('location_model_*_2025.txt')):
        pt = f.stem.split('_')[2]  # location_model_FF_2025 → FF
        location_models[pt] = lgb.Booster(model_file=str(f))

    return stuff, stuff_features, tunnel, location_models


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


def score_dataframe(df, stuff, stuff_features, tunnel, location_models, weights):
    """Run all three model stages on a dataframe, return df with prediction cols."""
    df = engineer_stuff_features(df)
    df = engineer_tunnel_features(df)
    df = engineer_location_features(df)

    # Stuff
    df['xRV_stuff'] = stuff.predict(df[stuff_features]).astype('float32')

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


def write_season_aggregates(df, output_dir: Path, season: int):
    """
    Build season-level pitcher aggregates (rolling averages across all
    scored pitches for the season so far). Two views:

    - per pitcher:                {pitcher_id: {n, xRV/100, stuff/100, ...}}
    - per pitcher × pitch_type:   {pitcher_id: {pitch_type: {n, xRV/100, ...}}}
    """
    season_dir = output_dir / 'season'
    season_dir.mkdir(parents=True, exist_ok=True)

    # Pitcher-level aggregation
    pitcher_agg = df.groupby('pitcher').agg(
        n=('xRV_final', 'count'),
        xRV=('xRV_final', 'mean'),
        stuff=('xRV_stuff', 'mean'),
        loc=('xRV_location', 'mean'),
        tun=('xRV_tunnel', 'mean'),
    )

    # Scale to per-100 and round
    for col in ['xRV', 'stuff', 'loc', 'tun']:
        pitcher_agg[col] = (pitcher_agg[col] * 100).round(3)

    pitcher_out = {
        str(int(pid)): {
            'n':     int(row['n']),
            'xRV':   float(row['xRV']),
            'stuff': float(row['stuff']),
            'loc':   float(row['loc']),
            'tun':   float(row['tun']),
        }
        for pid, row in pitcher_agg.iterrows()
    }

    # Pitcher × pitch_type aggregation
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
        if pid not in pt_out:
            pt_out[pid] = {}
        pt_out[pid][row['pitch_type']] = {
            'n':     int(row['n']),
            'xRV':   float(row['xRV']),
            'stuff': float(row['stuff']),
            'loc':   float(row['loc']),
            'tun':   float(row['tun']),
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
    stuff, stuff_features, tunnel, location_models = load_models(model_dir)
    print(f'  Stuff: {len(stuff_features)} features')
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
        'season', 'stand', 'p_throws', 'plate_x', 'plate_z',
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
    df = score_dataframe(df, stuff, stuff_features, tunnel, location_models, weights)
    print(f'  Scored {len(df):,} pitches')
    print(f'  Mean predicted xRV: {df["xRV_final"].mean():.5f}')
    print(f'  Mean actual xRV:    {df["xRV"].mean():.5f}' if 'xRV' in df.columns else '')

    print('Writing per-game JSON files...')
    write_per_game_json(df, output_dir, compress=not args.no_compress)

    print(f'Writing season {season} aggregates...')
    write_season_aggregates(df, output_dir, season)

    print('Done.')


if __name__ == '__main__':
    main()
