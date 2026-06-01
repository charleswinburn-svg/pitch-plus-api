"""
Pitch+ live scoring service.

POST /score
  body: {"pitches": [<pitch_event>, ...]}
  where each pitch_event is a single playEvent from MLB Stats API
  with pitchData populated. Optionally include "pitcher_id".

returns: {"scores": [{"index": int, "pitch_plus": float|null, ...}, ...]}

GET /health → {"status": "ok"}
"""
import json
import math
import io
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

# ── Load everything once at startup ──
print("Loading models...")
with open(MODELS_DIR / "stuff_model_metadata.json") as f:
    _stuff_meta = json.load(f)
stuff_fb_model = lgb.Booster(model_file=str(MODELS_DIR / _stuff_meta["fb_model"]["file"]))
stuff_offspeed_model = lgb.Booster(model_file=str(MODELS_DIR / _stuff_meta["offspeed_model"]["file"]))
stuff_fb_features = _stuff_meta["fb_model"]["features"]
stuff_offspeed_features = _stuff_meta["offspeed_model"]["features"]
_STUFF_FB_TYPES = set(_stuff_meta["family_definition"]["fastball_types"])
_STUFF_MPH_THRESHOLD = _stuff_meta["family_definition"]["mph_threshold"]

tunnel_model = lgb.Booster(model_file=str(MODELS_DIR / "tunnel_model_2025.txt"))
with open(MODELS_DIR / "tunnel_model_metadata.json") as f:
    tunnel_features = json.load(f)["features"]

location_models = {}
for f in (MODELS_DIR).glob("location_model_*_2025.txt"):
    pt = f.stem.split("_")[2]
    location_models[pt] = lgb.Booster(model_file=str(f))
with open(MODELS_DIR / "location_model_metadata.json") as f:
    location_features = json.load(f)["features"]

with open(MODELS_DIR / "final_model_config.json") as f:
    config = json.load(f)
weights = config["weights"]
intercept = config["intercept"]

with open(MODELS_DIR / "pitcher_baselines.json") as f:
    pitcher_baselines = json.load(f)

# Authoritative pitcher handedness: baselines take priority, then arm_angles.
# Populated from p_throws field added by build_pitcher_baselines.py /
# build_arm_angle_baselines.py. Overrides whatever the frontend sends.
pitcher_p_throws: dict[str, str] = {
    pid: d["p_throws"]
    for pid, d in pitcher_baselines.items()
    if d.get("p_throws") in ("L", "R")
}

# Arm angle baselines (for new stuff model)
arm_angle_path = MODELS_DIR / "pitcher_arm_angles.json"
if arm_angle_path.exists():
    with open(arm_angle_path) as f:
        pitcher_arm_angles = json.load(f)
    # Fill in any handedness not already in baselines
    for pid, d in pitcher_arm_angles.items():
        if pid not in pitcher_p_throws and d.get("p_throws") in ("L", "R"):
            pitcher_p_throws[pid] = d["p_throws"]
    print(f"Loaded {len(pitcher_arm_angles)} MLB pitcher arm angle baselines")
    print(f"Authoritative handedness: {len(pitcher_p_throws)} pitchers (L/R override from model files)")
else:
    pitcher_arm_angles = {}
    print("No pitcher_arm_angles.json — arm angle features will default to 0")

# Pitcher heights cache for geometric arm-angle estimation (WBC/AAA pitchers not
# in arm_angles baseline files). Built by build_pitcher_heights.py.
_pitcher_heights: dict = {}
_heights_path = MODELS_DIR / "pitcher_heights.json"
if _heights_path.exists():
    with open(_heights_path) as _f:
        _pitcher_heights = json.load(_f)
    print(f"Loaded {len(_pitcher_heights)} pitcher heights for geometric arm-angle fallback")

# AAA arm angle baselines (fallback for prospects/call-ups not yet in MLB data)
aaa_arm_angle_path = MODELS_DIR / "pitcher_arm_angles_aaa.json"
if aaa_arm_angle_path.exists():
    with open(aaa_arm_angle_path) as f:
        aaa_arm_angles = json.load(f)
    # MLB takes precedence — only fill in AAA pitchers not already in MLB data
    added = 0
    for pid, data in aaa_arm_angles.items():
        if pid not in pitcher_arm_angles:
            pitcher_arm_angles[pid] = data
            added += 1
    print(f"Loaded {len(aaa_arm_angles)} AAA pitcher arm angle baselines ({added} new prospects added)")

# Slot regression: per-pitch-type linear coefficients mapping arm_angle → expected pfx_x/z.
# Used to compute pfx_*_dev_from_slot for any pitcher (matches training-time logic).
slot_regression_path = MODELS_DIR / "slot_regression.json"
if slot_regression_path.exists():
    with open(slot_regression_path) as f:
        slot_regression = json.load(f)
    print(f"Loaded slot regression for {len(slot_regression)} pitch types")
else:
    slot_regression = {}
    print("No slot_regression.json — pfx_*_dev_from_slot will use pitcher mean fallback")

# Per-pitch-type Pitch+ league mean/std (computed from 2025 season aggregates)
with open(MODELS_DIR / "pitch_plus_norm.json") as f:
    pitch_plus_norm = json.load(f)  # {pt: {mean, std, stuff_mean, stuff_std, ...}}

# AAA norms (optional — falls back to MLB if not present)
aaa_norm_path = MODELS_DIR / "pitch_plus_norm_aaa.json"
if aaa_norm_path.exists():
    with open(aaa_norm_path) as f:
        pitch_plus_norm_aaa = json.load(f)
    print(f"Loaded AAA norms ({len(pitch_plus_norm_aaa)} pitch types)")
else:
    pitch_plus_norm_aaa = None
    print("No AAA norms found — AAA pitches will use MLB norms")

# ─────────────────────────────────────────────────────────────────────────
# Pitcher grades — produced by score_pitches.py write_season_aggregates.
# ─────────────────────────────────────────────────────────────────────────
import os

PITCHER_GRADES_DIR = Path(os.environ.get("PITCHER_GRADES_DIR", str(ROOT / "season")))
pitcher_grades = {}  # {season: {pid_str: {...}}}

if PITCHER_GRADES_DIR.exists():
    for f in PITCHER_GRADES_DIR.glob("pitcher_grades_*.json"):
        try:
            season = int(f.stem.split("_")[-1])
            with open(f) as fh:
                pitcher_grades[season] = json.load(fh)
            print(f"Loaded {len(pitcher_grades[season])} pitcher grades for {season} <- {f.name}")
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
else:
    print(f"PITCHER_GRADES_DIR not found ({PITCHER_GRADES_DIR}) -- /pitcher_percentiles will return 503")


def _percentile_rank(value, sorted_population):
    """Where does `value` fall within the sorted ascending population?"""
    if value is None or not sorted_population:
        return None
    n = len(sorted_population)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_population[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    count_lt = lo
    count_eq = 0
    while lo < n and sorted_population[lo] == value:
        count_eq += 1
        lo += 1
    rank = (count_lt + count_eq / 2) / n
    return round(rank * 100, 1)


def _qualified_distribution(grades_by_pid, metric, min_n=200):
    """Sorted ascending list of `metric` values across qualified pitchers."""
    vals = [
        g[metric] for g in grades_by_pid.values()
        if g.get(metric) is not None and g.get("n", 0) >= min_n
    ]
    vals.sort()
    return vals





print(f"Loaded {len(location_models)} location models, {len(pitcher_baselines)} baselines")

_tpc = tunnel_model.pandas_categorical or []
PITCH_TYPE_CATS = _tpc[0] if len(_tpc) > 0 else ['CH','CU','FC','FF','FS','KC','SI','SL','ST','SV']
THROWS_CATS     = _tpc[1] if len(_tpc) > 1 else ['L','R']
STAND_CATS      = ['L','R']

# Rare pitch-type aliases applied before model scoring and norm lookup.
# Mirrors PITCH_TYPE_REMAP in score_pitches.py.
PITCH_ALIASES = {'FO': 'FS', 'CS': 'CU', 'SC': 'CH', 'KN': 'FS'}

_GRAVITY = 32.174        # ft/s²
_Y0_REF  = 50.0          # Statcast kinematic reference y (ft from back of plate)
_Y_PLATE = 17.0 / 12.0  # front of plate (ft)


def _pfx_from_kinematics(ax, ay, az, vy0):
    """
    Derive Statcast-style pfx in feet from kinematic components (MLB Stats API).
    pfx = aerodynamic deviation from gravity-only trajectory at front of plate.
    ax, ay, az in ft/s²; vy0 in ft/s. Same coordinate system as Statcast.
    Bypasses MLB Stats API pfxX/pfxZ which use opposite sign convention from Statcast.
    """
    if any(v is None for v in [ax, ay, az, vy0]):
        return None, None
    dy   = _Y0_REF - _Y_PLATE          # 48.583 ft
    disc = vy0 * vy0 - 2.0 * ay * dy
    if disc < 0:
        return None, None
    t = (-vy0 - math.sqrt(disc)) / ay  # vy0<0, ay>0 → positive flight time
    if t <= 0:
        return None, None
    pfx_x = 0.5 * ax * t * t
    pfx_z = 0.5 * (az + _GRAVITY) * t * t  # subtract gravity's contribution
    return pfx_x, pfx_z


def map_pitch(evt: dict, pitcher_id: Optional[int] = None) -> Optional[dict]:
    """Convert MLB Stats API playEvent → Statcast-style row."""
    pd_ = evt.get("pitchData") or {}
    if not pd_:
        return None
    coords = pd_.get("coordinates") or {}
    breaks = pd_.get("breaks") or {}
    type_code = (evt.get("details") or {}).get("type", {}).get("code")
    if not type_code:
        return None
    if evt.get("_pfx_direct"):
        pfx_x = coords.get("pfxX")
        pfx_z = coords.get("pfxZ")
    else:
        pfx_x, pfx_z = _pfx_from_kinematics(
            coords.get("aX"), coords.get("aY"), coords.get("aZ"), coords.get("vY0"),
        )
    return {
        "pitch_type": type_code,
        "release_speed": pd_.get("startSpeed"),
        "release_spin_rate": breaks.get("spinRate"),
        "release_extension": pd_.get("extension"),
        "pfx_x": pfx_x,
        "pfx_z": pfx_z,
        "plate_x": coords.get("pX"),
        "plate_z": coords.get("pZ"),
        "release_pos_x": coords.get("x0"),
        "release_pos_z": coords.get("z0"),
        "vx0": coords.get("vX0"),
        "vy0": coords.get("vY0"),
        "vz0": coords.get("vZ0"),
        "ax": coords.get("aX"),
        "ay": coords.get("aY"),
        "az": coords.get("aZ"),
        "spin_axis": breaks.get("spinDirection"),
        "sz_top": pd_.get("strikeZoneTop"),
        "sz_bot": pd_.get("strikeZoneBottom"),
        "pitcher": pitcher_id,
        "stand": evt.get("_stand", "R"),
        "p_throws": evt.get("_p_throws", "R"),
    }


def engineer_and_score(rows: list[dict], norm_dict: dict = None) -> list[dict]:
    """Run feature engineering + 3-stage model on a list of mapped pitches."""
    if norm_dict is None:
        norm_dict = pitch_plus_norm
    df = pd.DataFrame(rows)
    if df.empty:
        return []

    # Alias rare pitch types to their closest equivalent for model scoring
    # FO (forkball) → FS (splitter): same grip family, same model treatment
    df['pitch_type_display'] = df['pitch_type'].copy()  # preserve original for output
    df['pitch_type'] = df['pitch_type'].replace(PITCH_ALIASES)

    # Cast to float64
    for c in ['release_speed','pfx_x','pfx_z','release_spin_rate','spin_axis',
              'release_extension','release_pos_x','release_pos_z','vx0','vy0','vz0',
              'ax','ay','az','plate_x','plate_z','sz_top','sz_bot']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')

    # ── v3 STUFF FEATURES ──
    # Match stuff_model_v3 training notebook exactly.

    # Override p_throws with authoritative handedness for known pitchers.
    # This guards against the frontend sending the wrong handedness (default "R")
    # for LHP with unusual pfx profiles (e.g. extreme arm-slot pitchers).
    if pitcher_p_throws:
        for pid_str, pt in pitcher_p_throws.items():
            try:
                pid_int = int(pid_str)
            except (ValueError, TypeError):
                continue
            mask = df['pitcher'] == pid_int
            if mask.any():
                df.loc[mask, 'p_throws'] = pt

    # Handedness-corrected horizontal break (positive = arm side)
    hand_sign = np.where(df['p_throws'] == 'L', -1.0, 1.0)
    df['arm_side_break'] = (df['pfx_x'].values * hand_sign).astype(float)

    # Total movement & spin efficiency
    df['total_movement'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    spin = df['release_spin_rate'].values
    df['spin_efficiency'] = np.where(spin > 0, df['total_movement'] / (spin / 1000), np.nan)

    # Spin axis circular encoding
    if 'spin_axis' in df.columns:
        axis_rad = np.deg2rad(df['spin_axis'].values)
        df['spin_axis_sin'] = np.sin(axis_rad)
        df['spin_axis_cos'] = np.cos(axis_rad)

    # Release tilt
    df['release_tilt'] = np.arctan2(df['vz0'].values, -df['vy0'].values)

    # VAA / HAA still computed for tunnel features below
    t = np.clip(50.0 / (-df['vy0'].values), 0.35, 0.55)
    df['vaa'] = np.degrees(np.arctan2(df['vz0'].values + df['az'].values * t,
                                       -(df['vy0'].values + df['ay'].values * t)))
    df['haa'] = np.degrees(np.arctan2(df['vx0'].values + df['ax'].values * t,
                                       -(df['vy0'].values + df['ay'].values * t)))

    # Deltas vs pitcher's fastball baseline (now includes fb_velo as direct feature)
    df['fb_velo'] = np.nan
    for col in ['delta_velo','delta_pfx_x','delta_pfx_z','delta_spin','delta_extension','delta_vaa',
                'release_diff_x','release_diff_z','release_distance','movement_separation']:
        df[col] = 0.0

    for i, row in df.iterrows():
        pid = str(int(row['pitcher'])) if pd.notna(row.get('pitcher')) else None
        bl = pitcher_baselines.get(pid) if pid else None
        if bl:
            df.at[i,'fb_velo'] = bl['fb_velo']
            df.at[i,'delta_velo'] = row['release_speed'] - bl['fb_velo']
            df.at[i,'delta_pfx_x'] = row['pfx_x'] - bl['fb_pfx_x']
            df.at[i,'delta_pfx_z'] = row['pfx_z'] - bl['fb_pfx_z']
            df.at[i,'delta_spin'] = row['release_spin_rate'] - bl['fb_spin']
            df.at[i,'delta_extension'] = row['release_extension'] - bl['fb_extension']
            df.at[i,'delta_vaa'] = row['vaa'] - bl['fb_vaa']
            df.at[i,'release_diff_x'] = row['release_pos_x'] - bl['fb_release_x']
            df.at[i,'release_diff_z'] = row['release_pos_z'] - bl['fb_release_z']
            df.at[i,'release_distance'] = np.sqrt(df.at[i,'release_diff_x']**2 + df.at[i,'release_diff_z']**2)
            df.at[i,'movement_separation'] = np.sqrt((row['pfx_x']-bl['fb_pfx_x'])**2 + (row['pfx_z']-bl['fb_pfx_z'])**2)

    df['pitch_type_cat'] = pd.Categorical(df['pitch_type'], categories=PITCH_TYPE_CATS)
    df['p_throws_cat'] = pd.Categorical(df['p_throws'], categories=THROWS_CATS)
    df['stand_cat'] = pd.Categorical(df['stand'], categories=STAND_CATS)
    df['same_side'] = (df['p_throws'] == df['stand']).astype('int8')

    # ── Arm angle features ──
    df['arm_angle'] = np.nan
    df['arm_angle_std'] = np.nan
    df['arm_angle_dev'] = np.nan

    for pid, group in df.groupby('pitcher'):
        pid_s = str(int(pid)) if pd.notna(pid) else None
        aa = pitcher_arm_angles.get(pid_s) if pid_s else None
        if not aa:
            continue
        mask = df['pitcher'] == pid
        df.loc[mask, 'arm_angle'] = aa['arm_angle']
        df.loc[mask, 'arm_angle_std'] = aa['arm_angle_std']
        df.loc[mask, 'arm_angle_dev'] = 0.0  # no per-pitch AA from MLB Stats API

    # Geometric fallback for pitchers with no arm_angle baseline (WBC/AAA).
    # Estimates angle from release position and cached pitcher height.
    _DEFAULT_H = 74.0  # 6'2" average MLB pitcher height
    _nan_aa = df['arm_angle'].isna()
    if _nan_aa.any():
        _rel_x = pd.to_numeric(df.loc[_nan_aa, 'release_pos_x'], errors='coerce').values
        _rel_z = pd.to_numeric(df.loc[_nan_aa, 'release_pos_z'], errors='coerce').values
        _throws = df.loc[_nan_aa, 'p_throws'].values if 'p_throws' in df.columns else np.full(_nan_aa.sum(), 'R')
        _h = np.array([
            _pitcher_heights.get(str(int(pid)), _DEFAULT_H) if pd.notna(pid) else _DEFAULT_H
            for pid in df.loc[_nan_aa, 'pitcher']
        ], dtype='float64')
        _shoulder_z = _h * 0.70
        _adj = (_rel_z * 12.0) - _shoulder_z
        _opp = np.abs(_rel_x * 12.0)
        _est = np.degrees(np.arctan2(_opp, _adj))
        _est = np.where(_throws == 'L', -_est, _est)
        df.loc[_nan_aa, 'arm_angle'] = _est
        df.loc[_nan_aa & df['arm_angle_dev'].isna(), 'arm_angle_dev'] = 0.0

    # ── v3 SLOT REGRESSION: per (pitch_type, p_throws) ──
    # arm_side_break - (slope * arm_angle + intercept)
    # pfx_z          - (slope * arm_angle + intercept)
    df['pfx_x_dev_from_slot'] = np.nan
    df['pfx_z_dev_from_slot'] = np.nan
    slot_coefs = slot_regression.get('slot', slot_regression) if slot_regression else {}
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
    # ay - (intercept + slope_velo*velo + slope_spin*spin)
    df['ay_residual'] = 0.0
    drag_coefs = slot_regression.get('drag', {}) if slot_regression else {}
    if drag_coefs:
        for pt, c in drag_coefs.items():
            mask = df['pitch_type'] == pt
            if mask.sum() == 0:
                continue
            expected_ay = (c['intercept']
                           + c['slope_velo'] * df.loc[mask, 'release_speed']
                           + c['slope_spin'] * df.loc[mask, 'release_spin_rate'])
            df.loc[mask, 'ay_residual'] = (df.loc[mask, 'ay'] - expected_ay).astype(float)

    # ── Stuff prediction: route FB vs offspeed by velocity proximity to fastball baseline ──
    fb_mask = (
        df['fb_velo'].notna() & (np.abs(df['release_speed'] - df['fb_velo']) <= _STUFF_MPH_THRESHOLD)
    ) | (df['fb_velo'].isna() & df['pitch_type'].isin(_STUFF_FB_TYPES))
    os_mask = ~fb_mask
    df['xRV_stuff'] = 0.0
    if fb_mask.any():
        df.loc[fb_mask, 'xRV_stuff'] = stuff_fb_model.predict(df.loc[fb_mask, stuff_fb_features])
    if os_mask.any():
        df.loc[os_mask, 'xRV_stuff'] = stuff_offspeed_model.predict(df.loc[os_mask, stuff_offspeed_features])

    # ── Location features ──
    df['plate_x_adj'] = np.where(df['stand']=='L', -df['plate_x'], df['plate_x'])
    df['zone_center_dist'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - 2.5)**2)
    df['in_zone'] = ((df['plate_x'].abs() <= 0.83) &
                     (df['plate_z'] >= 1.5) & (df['plate_z'] <= 3.5)).astype(int)
    # Euclidean distance from zone edge (matches training script)
    x_e = np.maximum(df['plate_x'].abs() - 0.83, 0)
    z_t = np.maximum(df['plate_z'] - 3.5, 0)
    z_b = np.maximum(1.5 - df['plate_z'], 0)
    df['out_of_zone_dist'] = np.sqrt(x_e**2 + np.maximum(z_t, z_b)**2)

    df['xRV_location'] = 0.0
    for pt, model in location_models.items():
        mask = df['pitch_type'] == pt
        if mask.sum() > 0:
            X = df.loc[mask, location_features]
            df.loc[mask, 'xRV_location'] = model.predict(X)

    # ── Tunnel features: real trajectory math ──
    PLATE_Y = 17.0 / 12.0
    y_tun = PLATE_Y + 23.0
    y0 = 50.0

    a_ = 0.5 * df['ay'].values
    b_ = df['vy0'].values
    c_ = y0 - y_tun
    with np.errstate(invalid='ignore', divide='ignore'):
        disc = b_**2 - 4 * a_ * c_
        t_tun = np.where(disc >= 0,
                         (-b_ - np.sqrt(np.maximum(disc, 0))) / (2 * a_),
                         np.nan)
        t_tun = np.where((t_tun > 0) & (t_tun < 0.5), t_tun, np.nan)

    # Trajectory at tunnel point — needs vx0 and ax which may be missing
    # If missing, fall back to release_pos (no x deflection)
    vx0 = df['vx0'].fillna(0).values if 'vx0' in df.columns else np.zeros(len(df))
    ax  = df['ax'].fillna(0).values if 'ax' in df.columns else np.zeros(len(df))
    df['tunnel_x'] = (df['release_pos_x'].values + vx0 * t_tun + 0.5 * ax * t_tun**2)
    df['tunnel_z'] = (df['release_pos_z'].values + df['vz0'].values * t_tun
                      + 0.5 * df['az'].values * t_tun**2)

    c_p = y0 - PLATE_Y
    disc_p = b_**2 - 4 * a_ * c_p
    with np.errstate(invalid='ignore', divide='ignore'):
        t_p = np.where(disc_p >= 0,
                       (-b_ - np.sqrt(np.maximum(disc_p, 0))) / (2 * a_),
                       np.nan)
    df['time_after_tunnel'] = t_p - t_tun

    # Diffs vs pitcher's fastball tunnel anchor (from baselines)
    df['tunnel_diff_x'] = 0.0
    df['tunnel_diff_z'] = 0.0
    df['plate_distance'] = 0.0
    for i, row in df.iterrows():
        pid = str(int(row['pitcher'])) if pd.notna(row.get('pitcher')) else None
        bl = pitcher_baselines.get(pid) if pid else None
        if bl and 'fb_tunnel_x' in bl:
            df.at[i,'tunnel_diff_x'] = (row['tunnel_x'] - bl['fb_tunnel_x']) if pd.notna(row['tunnel_x']) else 0.0
            df.at[i,'tunnel_diff_z'] = (row['tunnel_z'] - bl['fb_tunnel_z']) if pd.notna(row['tunnel_z']) else 0.0
            df.at[i,'plate_distance'] = np.sqrt(
                (row['plate_x'] - bl['fb_plate_x'])**2 +
                (row['plate_z'] - bl['fb_plate_z'])**2
            ) * 12

    df['tunnel_distance'] = np.sqrt(df['tunnel_diff_x']**2 + df['tunnel_diff_z']**2) * 12
    df['late_break'] = df['plate_distance'] - df['tunnel_distance']

    # Fill any remaining NaNs with 0 so LightGBM doesn't choke
    for col in ['tunnel_diff_x','tunnel_diff_z','tunnel_distance','time_after_tunnel',
                'late_break','plate_distance']:
        df[col] = df[col].fillna(0)

    X_tunnel = df[tunnel_features]
    df['xRV_tunnel'] = tunnel_model.predict(X_tunnel)

    # ── Final ridge combination ──
    df['xRV_final'] = (weights['stuff'] * df['xRV_stuff'] +
                       weights['location'] * df['xRV_location'] +
                       weights['tunnel'] * df['xRV_tunnel'] +
                       intercept)

    # ── Convert to Pitch+ (100 = avg, ±10 per std, higher = better) ──
    out = []
    for i, row in df.iterrows():
        pt = row['pitch_type']  # aliased (FO→FS) for norm lookup
        pt_display = row['pitch_type_display']  # original for output
        n = norm_dict.get(pt)

        def _grade(val, mean_key, std_key):
            if not n or std_key not in n or n[std_key] <= 0:
                return None
            z = max(-4, min(4, (val - n[mean_key]) / n[std_key]))
            return round(100 - z * 10, 1)

        out.append({
            "index": int(i),
            "pitch_type": pt_display,
            "stuff_plus": _grade(row['xRV_stuff'], 'stuff_mean', 'stuff_std'),
            "loc_plus": _grade(row['xRV_location'], 'loc_mean', 'loc_std'),
            "tunnel_plus": _grade(row['xRV_tunnel'], 'tun_mean', 'tun_std'),
            "pitch_plus": _grade(row['xRV_final'], 'mean', 'std'),
            "xRV_stuff": float(row['xRV_stuff']),
            "xRV_location": float(row['xRV_location']),
            "xRV_tunnel": float(row['xRV_tunnel']),
            "xRV_final": float(row['xRV_final']),
        })
    return out


# ── FastAPI app ──
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down to your domain in prod
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class PitchRequest(BaseModel):
    pitches: list[dict]
    is_aaa: bool = False
    start_date: Optional[str] = None  # "YYYY-MM-DD" — filter pitches carrying a game_date field
    end_date:   Optional[str] = None


def _date_in_range(d, start: Optional[str], end: Optional[str]) -> bool:
    if not d:
        return True
    s = str(d)
    if start and s < start:
        return False
    if end and s > end:
        return False
    return True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": len(location_models),
        "baselines": len(pitcher_baselines),
        "arm_angles": len(pitcher_arm_angles),
        "has_aaa_norms": pitch_plus_norm_aaa is not None,
        "pitch_type_cats": PITCH_TYPE_CATS,
    }


@app.post("/score")
def score(req: PitchRequest):
    rows = []
    for i, evt in enumerate(req.pitches):
        pid = evt.get("pitcher_id") or evt.get("_pitcher_id")
        mapped = map_pitch(evt, pid)
        if mapped:
            mapped['_orig_index'] = i
            rows.append(mapped)
    if not rows:
        return {"scores": []}
    # Use AAA norms when requested and available
    norm = (pitch_plus_norm_aaa if req.is_aaa and pitch_plus_norm_aaa else pitch_plus_norm)
    scored = engineer_and_score(rows, norm)
    return {"scores": scored}


def _per_type_plus_agg(xrv_mean: float, pitch_type: str, kind: str):
    """Convert mean xRV to pitcher×pitch-type Plus scale.

    Identical formula to score_pitches.py write_season_aggregates() so that
    /score_aggregate on a full season matches /leaderboard exactly.
    Stuff+ is additionally rescaled to global mean=100, stdev=10 using the
    season params stored in pitch_plus_norm['_stuff_plus_rescale'].
    """
    n = pitch_plus_norm.get(pitch_type)
    if not n:
        return None
    mean_key = {'stuff': 'stuff_mean', 'loc': 'loc_mean', 'tun': 'tun_mean', 'pitch': 'mean'}[kind]
    std_key  = {'stuff': 'stuff_std',  'loc': 'loc_std',  'tun': 'tun_std',  'pitch': 'std'}[kind]
    if std_key not in n or n[std_key] <= 0:
        return None
    z = max(-4.0, min(4.0, (xrv_mean - n[mean_key]) / n[std_key]))
    raw = 100.0 - z * 10.0
    if kind == 'stuff':
        r = pitch_plus_norm.get('_stuff_plus_rescale', {})
        if isinstance(r, dict) and r.get('stdev', 0) > 0:
            raw = 100.0 + (raw - r['mean']) / r['stdev'] * 10.0
    return round(raw, 1)


def _weighted_overall_agg(by_pt: dict) -> dict:
    """Usage-weighted average of per-pitch-type Plus values."""
    totals = {k: 0.0 for k in ('stuff', 'loc', 'tun', 'pitch')}
    counts = {k: 0   for k in ('stuff', 'loc', 'tun', 'pitch')}
    n_total = 0
    for g in by_pt.values():
        n = g.get('n', 0)
        n_total += n
        for k in ('stuff', 'loc', 'tun', 'pitch'):
            v = g.get(k)
            if v is not None and n > 0:
                totals[k] += v * n
                counts[k] += n
    out = {'n': n_total}
    for k in ('stuff', 'loc', 'tun', 'pitch'):
        out[k] = round(totals[k] / counts[k], 1) if counts[k] else None
    return out


@app.post("/score_aggregate")
def score_aggregate(req: PitchRequest):
    """Score pitches and aggregate to the pitcher×pitch-type Plus scale.

    Same payload as /score. Response by_pitch_type shape matches /leaderboard
    so single-game and season numbers are directly comparable.
    """
    rows = []
    pitcher_ids = []
    for evt in req.pitches:
        pid = evt.get("pitcher_id") or evt.get("_pitcher_id")
        mapped = map_pitch(evt, pid)
        if mapped:
            rows.append(mapped)
            pitcher_ids.append(pid)

    if req.start_date or req.end_date:
        paired = [(r, pid) for r, pid in zip(rows, pitcher_ids)
                  if _date_in_range(r.get("game_date"), req.start_date, req.end_date)]
        if paired:
            rows, pitcher_ids = zip(*paired)
            rows, pitcher_ids = list(rows), list(pitcher_ids)
        else:
            rows, pitcher_ids = [], []

    if not rows:
        return {"by_pitch_type": {}, "overall": {"stuff": None, "loc": None, "tun": None, "pitch": None, "n": 0}}

    norm = (pitch_plus_norm_aaa if req.is_aaa and pitch_plus_norm_aaa else pitch_plus_norm)
    scored = engineer_and_score(rows, norm)

    def _new_bucket():
        return {"stuff": 0.0, "loc": 0.0, "tun": 0.0, "pitch": 0.0, "n": 0}

    cross = defaultdict(_new_bucket)
    per_pid = defaultdict(lambda: defaultdict(_new_bucket))
    distinct_pids = set()

    for i, s in enumerate(scored):
        pid = pitcher_ids[i]
        pt = s.get("pitch_type_display", s.get("pitch_type"))  # original pre-alias type
        for b in (cross[pt], per_pid[pid][pt]):
            b["stuff"] += s.get("xRV_stuff",    0.0)
            b["loc"]   += s.get("xRV_location", 0.0)
            b["tun"]   += s.get("xRV_tunnel",   0.0)
            b["pitch"] += s.get("xRV_final",    0.0)
            b["n"]     += 1
        distinct_pids.add(pid)

    def _bucket_to_plus(pt_display, b):
        n = b["n"]
        if n == 0:
            return None
        pt_norm = PITCH_ALIASES.get(pt_display, pt_display)
        return {
            "stuff": _per_type_plus_agg(b["stuff"] / n, pt_norm, "stuff"),
            "loc":   _per_type_plus_agg(b["loc"]   / n, pt_norm, "loc"),
            "tun":   _per_type_plus_agg(b["tun"]   / n, pt_norm, "tun"),
            "pitch": _per_type_plus_agg(b["pitch"] / n, pt_norm, "pitch"),
            "n": n,
        }

    by_pt = {pt: g for pt, b in cross.items() if (g := _bucket_to_plus(pt, b)) is not None}
    resp = {"by_pitch_type": by_pt, "overall": _weighted_overall_agg(by_pt)}

    if len(distinct_pids) > 1:
        resp["by_pitcher"] = {}
        for pid, pt_map in per_pid.items():
            pid_by_pt = {pt: g for pt, b in pt_map.items() if (g := _bucket_to_plus(pt, b)) is not None}
            resp["by_pitcher"][str(pid)] = {
                "by_pitch_type": pid_by_pt,
                "overall": _weighted_overall_agg(pid_by_pt),
            }

    return resp


@app.get("/pitcher_percentiles/{pitcher_id}")
def pitcher_percentiles(pitcher_id: int, season: int = 2026, min_n: int = 200):
    """
    Return Stuff+/Loc+/Tun+/Pitch+ values for one pitcher AND their
    percentile rank within that season's qualified-pitcher distribution.

    Qualified = pitchers with at least `min_n` scored pitches in the season.
    Default 200 pitches ≈ 1-2 starts for a starter, several appearances for
    a reliever — low enough to include most active arms without including
    one-batter cameos.
    """
    if season not in pitcher_grades:
        return {
            "error": f"No grades loaded for season {season}",
            "available_seasons": list(pitcher_grades.keys()),
        }

    grades = pitcher_grades[season]
    pid_str = str(pitcher_id)
    me = grades.get(pid_str)
    if not me:
        return {"error": f"Pitcher {pitcher_id} not found in {season} grades"}

    metrics = ["stuff_plus", "loc_plus", "tun_plus", "pitch_plus"]
    out = {
        "pitcher_id": pitcher_id,
        "season": season,
        "n_pitches": me.get("n"),
        "qualified": me.get("n", 0) >= min_n,
        "qualified_threshold": min_n,
    }
    for m in metrics:
        dist = _qualified_distribution(grades, m, min_n=min_n)
        out[m] = {
            "value": me.get(m),
            "percentile": _percentile_rank(me.get(m), dist),
            "population_size": len(dist),
        }
    return out


@app.get("/pitcher_grades/{pitcher_id}")
def pitcher_grade_lookup(pitcher_id: int, season: int = 2026):
    """Raw aggregate for a single pitcher (no percentile)."""
    if season not in pitcher_grades:
        return {"error": f"No grades loaded for season {season}"}
    pid_str = str(pitcher_id)
    me = pitcher_grades[season].get(pid_str)
    if not me:
        return {"error": f"Pitcher {pitcher_id} not found"}
    return {"pitcher_id": pitcher_id, "season": season, **me}


# ─────────────────────────────────────────────────────────────────────────
# Add to server.py near the other pitcher_* endpoints. Returns the full
# pitcher_grades dict for a season so the frontend can rank a live-computed
# value against an arbitrary subset (e.g. percentile-card-eligible pitchers).
# ─────────────────────────────────────────────────────────────────────────

@app.get("/pitcher_grades_distribution")
def pitcher_grades_distribution(season: int = 2026):
    """
    Return all cached pitcher grades for a season as { pitcher_id: {...} }.
    Used by the PitcherCard's live-scoring path: it computes Stuff+/Loc+/...
    fresh from MLB Stats API, then ranks the result against this distribution
    (filtered client-side to whichever set of eligible pitchers it cares about).
    """
    if season not in pitcher_grades:
        return {"error": f"No grades loaded for season {season}",
                "available_seasons": list(pitcher_grades.keys())}
    return {
        "season": season,
        "grades": pitcher_grades[season],
    }


MIN_N_OVERALL = 200
MIN_N_PER_PITCH_TYPE = 30


@lru_cache(maxsize=8)
def _build_leaderboard(season: int) -> dict:
    overall_path = PITCHER_GRADES_DIR / f"pitcher_grades_{season}.json"
    pt_path      = PITCHER_GRADES_DIR / f"pitcher_pitch_type_grades_{season}.json"
    if not overall_path.exists():
        return {"season": season, "pitchers": [], "error": "aggregates not found"}

    overall_raw = json.loads(overall_path.read_text())
    pt_raw      = json.loads(pt_path.read_text()) if pt_path.exists() else {}

    result = []
    for pid, g in overall_raw.items():
        if g.get("n", 0) < MIN_N_OVERALL:
            continue
        overall = {
            "stuff": g.get("stuff_plus"),
            "loc":   g.get("loc_plus"),
            "tun":   g.get("tun_plus"),
            "pitch": g.get("pitch_plus"),
            "n":     int(g["n"]),
        }
        by_pt = {}
        for pt, pg in pt_raw.get(pid, {}).items():
            if pg.get("n", 0) < MIN_N_PER_PITCH_TYPE:
                continue
            by_pt[pt] = {
                "stuff": pg.get("stuff_plus"),
                "loc":   pg.get("loc_plus"),
                "tun":   pg.get("tun_plus"),
                "pitch": pg.get("pitch_plus"),
                "n":     int(pg["n"]),
            }
        result.append({
            "player_id":     int(pid),
            "overall":       overall,
            "by_pitch_type": by_pt,
        })

    return {"season": season, "pitchers": result}


@app.get("/leaderboard")
def leaderboard(season: int = 2026):
    if season < 2015 or season > 2100:
        raise HTTPException(status_code=400, detail=f"season out of range: {season}")
    return _build_leaderboard(season)


# ─────────────────────────────────────────────────────────────────────────────
# /player_stats/{player_id} — date-range stats for hitter/pitcher cards
# ─────────────────────────────────────────────────────────────────────────────

_FG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,*/*",
    "Referer": "https://www.fangraphs.com/",
}
_SAVANT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/csv,*/*",
}

# iSwing+ lazy state
ISWING_DIR = Path(os.environ.get("ISWING_MODELS_DIR", str(ROOT.parent)))
_iswing_loaded: bool = False
_iswing_models = None  # (model_a, model_b, scaler_a, scaler_b, config) or None
_iswing_norm: dict = {}  # {year: {log_mean, log_std}}

_SWING_DESCS = {
    'hit_into_play', 'swinging_strike', 'swinging_strike_blocked',
    'foul', 'foul_tip', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip',
    'hit_into_play_no_out', 'hit_into_play_score',
}


def _fetch_fg_rows(url: str) -> list:
    """Fetch FanGraphs API data rows, trying requests then cloudscraper."""
    if not _HAS_REQUESTS:
        return []
    try:
        r = _requests.get(url, headers=_FG_HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json().get("data", [])
    except Exception:
        pass
    try:
        import cloudscraper
        r = cloudscraper.create_scraper().get(url, headers=_FG_HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json().get("data", [])
    except Exception:
        pass
    return []


def _fg_find_player(df: pd.DataFrame, player_id: int) -> Optional[pd.Series]:
    mlbam_col = next((c for c in df.columns if c.lower() in ("xmlbamid", "mlbamid")), None)
    if mlbam_col is None:
        return None
    matches = df[pd.to_numeric(df[mlbam_col], errors="coerce") == player_id]
    return matches.iloc[0] if not matches.empty else None


def _fetch_fg_siera(player_id: int, season: int, start: str, end: str) -> Optional[float]:
    url = (
        "https://www.fangraphs.com/api/leaders/major-league/data"
        f"?pos=all&stats=pit&lg=all&qual=1&type=8"
        f"&season={season}&month=0&season1={season}&ind=0"
        f"&team=0&rost=0&age=0&filter=&players=0"
        f"&startdate={start}&enddate={end}&pageitems=2000&page=1"
    )
    rows = _fetch_fg_rows(url)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    row = _fg_find_player(df, player_id)
    if row is None:
        return None
    siera_col = next((c for c in df.columns if c.upper() == "SIERA"), None)
    if siera_col is None:
        return None
    val = pd.to_numeric(row.get(siera_col), errors="coerce")
    return float(val) if not pd.isna(val) else None


def _fetch_savant_bat_tracking(player_id: int, season: int, start: str, end: str) -> Optional[float]:
    """Return Blasts/Contact% (0–100) from Savant bat tracking leaderboard."""
    if not _HAS_REQUESTS:
        return None
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        f"?attackZone=&batSide=&contactType=&count="
        f"&dateStart={start}&dateEnd={end}"
        f"&minSwings=0&type=details&csv=true"
    )
    try:
        r = _requests.get(url, headers=_SAVANT_HEADERS, timeout=60)
        if not r.ok or len(r.text) < 50:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        pid_col = next((c for c in df.columns if c.lower() in ("player_id", "batter", "mlbamid", "xmlbamid")), None)
        if pid_col is None:
            return None
        row = df[pd.to_numeric(df[pid_col], errors="coerce") == player_id]
        if row.empty:
            return None
        blast_col = next((c for c in df.columns if "blast" in c.lower()), None)
        if blast_col is None:
            return None
        val = pd.to_numeric(row.iloc[0][blast_col], errors="coerce")
        if pd.isna(val):
            return None
        # Savant stores as decimal (0.083); return as percentage (8.3)
        return float(val) * 100 if val <= 1.0 else float(val)
    except Exception:
        return None


def _iswing_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features required by the iSwing+ model (mirrors iswing_update.py)."""
    if "attack_angle" in df.columns:
        df["ideal_attack_angle"] = df["attack_angle"].between(5, 20).fillna(False).astype(int)

    CONTACT = {"hit_into_play", "foul", "hit_into_play_no_out", "hit_into_play_score", "foul_bunt"}
    df["made_contact"] = df["description"].isin(CONTACT).astype(int)

    if all(c in df.columns for c in ["plate_x", "plate_z", "sz_top", "sz_bot"]):
        df["in_zone"] = (
            (df["plate_x"].abs() <= 0.83) &
            (df["plate_z"] >= df["sz_bot"]) &
            (df["plate_z"] <= df["sz_top"])
        ).fillna(False).astype(int)
    elif all(c in df.columns for c in ["plate_x", "plate_z"]):
        df["in_zone"] = (
            (df["plate_x"].abs() <= 0.83) & df["plate_z"].between(1.5, 3.5)
        ).fillna(False).astype(int)

    if all(c in df.columns for c in ["plate_x", "plate_z"]):
        df["location_difficulty"] = np.sqrt(df["plate_x"]**2 + (df["plate_z"] - 2.5)**2)

    pitch_families = {
        "FF":"fastball","SI":"fastball","FC":"fastball","FA":"fastball",
        "SL":"breaking","CU":"breaking","KC":"breaking","SV":"breaking","CS":"breaking","ST":"breaking",
        "CH":"offspeed","FS":"offspeed","FO":"offspeed","SC":"offspeed","KN":"offspeed","EP":"offspeed",
    }
    if "pitch_type" in df.columns:
        pf = df["pitch_type"].map(pitch_families).fillna("other")
        df["is_fastball"] = (pf == "fastball").astype(int)
        df["is_breaking"] = (pf == "breaking").astype(int)
        df["is_offspeed"] = (pf == "offspeed").astype(int)

    if all(c in df.columns for c in ["pfx_x", "pfx_z"]):
        df["total_movement"] = np.sqrt(df["pfx_x"]**2 + df["pfx_z"]**2)
    if all(c in df.columns for c in ["balls", "strikes"]):
        df["count_leverage"] = df["balls"] - df["strikes"]
    if all(c in df.columns for c in ["attack_direction", "plate_x"]):
        df["directional_match"] = -df["attack_direction"] * df["plate_x"]
    if all(c in df.columns for c in ["bat_speed", "release_speed"]):
        df["speed_differential"] = df["bat_speed"] - (df["release_speed"] * 0.7)

    if all(c in df.columns for c in ["bat_speed", "plate_x"]):
        df["plate_x_bin"] = pd.cut(df["plate_x"], bins=20, labels=False)
        exp_speed = df.groupby("plate_x_bin")["bat_speed"].transform("mean")
        df["speed_over_expected"] = (df["bat_speed"] - exp_speed).fillna(0)
        df.drop(columns=["plate_x_bin"], inplace=True)

    if all(c in df.columns for c in ["bat_speed", "location_difficulty"]):
        df["speed_vs_location"] = df["bat_speed"] * (1 + 0.3 * df["location_difficulty"])

    if all(c in df.columns for c in ["launch_speed", "bat_speed", "release_speed"]):
        df["theoretical_max_ev"] = 1.23 * df["bat_speed"] + 0.23 * df["release_speed"]
        df["squared_up_rate"] = df["launch_speed"] / df["theoretical_max_ev"].replace(0, np.nan)

    if "estimated_woba_using_speedangle" in df.columns:
        contact_mask = df["launch_speed"].notna() & df["launch_angle"].notna()
        df["xwOBAcon"] = np.where(contact_mask, df["estimated_woba_using_speedangle"], np.nan)

    return df


def _iswing_score_df(df: pd.DataFrame, model_a, model_b, scaler_a, scaler_b, config) -> pd.DataFrame:
    feat_a = config["features_a"]
    feat_b = config["features_b"]
    meds_a = config["feature_medians_a"]
    meds_b = config["feature_medians_b"]
    avail_a = [f for f in feat_a if f in df.columns]
    avail_b = [f for f in feat_b if f in df.columns]
    Xa = df[avail_a].copy()
    Xb = df[avail_b].copy()
    for c in Xa.columns: Xa[c] = Xa[c].fillna(meds_a.get(c, Xa[c].median()))
    for c in Xb.columns: Xb[c] = Xb[c].fillna(meds_b.get(c, Xb[c].median()))
    Xa_sc = pd.DataFrame(scaler_a.transform(Xa), columns=avail_a, index=df.index)
    Xb_sc = pd.DataFrame(scaler_b.transform(Xb), columns=avail_b, index=df.index)
    q_exp = config.get("quality_exponent", 1.5)
    c_exp = config.get("contact_exponent", 0.3)
    df = df.copy()
    df["pred_quality"] = np.maximum(model_a.predict(Xa_sc), 0.001)
    df["contact_prob"] = np.maximum(model_b.predict_proba(Xb_sc)[:, 1], 0.001)
    df["raw_value"] = df["pred_quality"]**q_exp * df["contact_prob"]**c_exp
    return df


def _ensure_iswing():
    """Lazy-load iSwing+ models and precompute year-level normalization."""
    global _iswing_loaded, _iswing_models, _iswing_norm
    if _iswing_loaded:
        return
    _iswing_loaded = True
    try:
        import joblib
        model_a  = joblib.load(str(ISWING_DIR / "iswing_model_a.pkl"))
        model_b  = joblib.load(str(ISWING_DIR / "iswing_model_b.pkl"))
        scaler_a = joblib.load(str(ISWING_DIR / "iswing_scaler_a.pkl"))
        scaler_b = joblib.load(str(ISWING_DIR / "iswing_scaler_b.pkl"))
        with open(str(ISWING_DIR / "iswing_config.json")) as f:
            config = json.load(f)
        _iswing_models = (model_a, model_b, scaler_a, scaler_b, config)
        print(f"Loaded iSwing+ models from {ISWING_DIR}")

        # Precompute per-year normalization params from full swings CSV
        csv_path = ISWING_DIR / "competitive_swings_2023_2026.csv"
        if csv_path.exists():
            try:
                raw = pd.read_csv(str(csv_path))
                raw = _iswing_enrich(raw)
                raw = raw.dropna(subset=["bat_speed"])
                raw = _iswing_score_df(raw, *_iswing_models)
                raw["year"] = pd.to_datetime(raw["game_date"], errors="coerce").dt.year
                for yr in raw["year"].dropna().unique():
                    yr = int(yr)
                    yr_agg = raw[raw["year"] == yr].groupby("batter")["raw_value"].mean()
                    if len(yr_agg) < 10:
                        continue
                    log_vals = np.log(yr_agg.clip(lower=1e-10))
                    _iswing_norm[yr] = {"log_mean": float(log_vals.mean()), "log_std": float(log_vals.std())}
                print(f"iSwing+ norms computed for years: {list(_iswing_norm.keys())}")
            except Exception as e:
                print(f"iSwing+ norm precompute failed: {e}")
    except Exception as e:
        print(f"iSwing+ models unavailable: {e}")


def _score_iswing_daterange(player_id: int, season: int, start: str, end: str) -> Optional[float]:
    """Score a player's iSwing+ for a date range and return the normalized 100-scale value."""
    _ensure_iswing()
    if _iswing_models is None or not _HAS_REQUESTS:
        return None

    url = (
        f"https://baseballsavant.mlb.com/statcast_search/csv"
        f"?all=true&type=details&hfSea={season}%7C&hfGT=R%7C"
        f"&player_type=batter&batters_lookup%5B%5D={player_id}"
    )
    if start:
        url += f"&start_date={start}"
    if end:
        url += f"&end_date={end}"

    try:
        r = _requests.get(url, headers=_SAVANT_HEADERS, timeout=60)
        if not r.ok or len(r.text) < 100:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        if "description" not in df.columns or "bat_speed" not in df.columns:
            return None
        df = df[df["description"].isin(_SWING_DESCS)].copy()
        df = df.dropna(subset=["bat_speed"])
        if len(df) < 5:
            return None

        # Filter to competitive swings (per-batter 10th percentile threshold)
        thresh = df["bat_speed"].quantile(0.10)
        df = df[(df["bat_speed"] >= thresh) | ((df["bat_speed"] >= 60) & (df.get("launch_speed", pd.Series(dtype=float)) >= 90))]
        if len(df) < 5:
            return None

        df = _iswing_enrich(df)
        df = _iswing_score_df(df, *_iswing_models)

        mean_raw = float(df["raw_value"].mean())
        norm = _iswing_norm.get(season)
        if norm is None:
            # Fall back to current year if available, else skip
            norm = next(iter(_iswing_norm.values()), None) if _iswing_norm else None
        if norm is None:
            return None

        log_mean = norm["log_mean"]
        log_std  = norm["log_std"]
        if log_std <= 0:
            return None
        score = 100.0 + 15.0 * (np.log(max(mean_raw, 1e-10)) - log_mean) / log_std
        return round(float(score), 1)
    except Exception as e:
        print(f"iSwing+ scoring error for {player_id}: {e}")
        return None


@app.get("/player_stats/{player_id}")
def player_stats(
    player_id: int,
    season: int = 2026,
    start: str = "",
    end: str = "",
    group: str = "hitting",
):
    """
    Return date-range stats for a hitter or pitcher card.

    Provides stats that can't be computed client-side:
    - hitting: Blasts/Contact (Savant bat tracking), iSwing+ (ML model)
    - pitching: SIERA (FanGraphs type=8)

    Returns partial data (HTTP 200) if some sources fail.
    """
    stats: dict = {}

    if group == "pitching":
        siera = _fetch_fg_siera(player_id, season, start, end)
        if siera is not None:
            stats["SIERA"] = siera

    else:  # hitting
        blasts = _fetch_savant_bat_tracking(player_id, season, start, end)
        if blasts is not None:
            stats["Blasts/Contact"] = round(blasts, 1)

        iswing = _score_iswing_daterange(player_id, season, start, end)
        if iswing is not None:
            stats["iSwing+"] = iswing

    return {"stats": stats}
