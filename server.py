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
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional

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

# Arm angle baselines (for new stuff model)
arm_angle_path = MODELS_DIR / "pitcher_arm_angles.json"
if arm_angle_path.exists():
    with open(arm_angle_path) as f:
        pitcher_arm_angles = json.load(f)
    print(f"Loaded {len(pitcher_arm_angles)} MLB pitcher arm angle baselines")
else:
    pitcher_arm_angles = {}
    print("No pitcher_arm_angles.json — arm angle features will default to 0")

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

PITCH_TYPE_CATS = ['CH','CU','FC','FF','FS','KC','SI','SL','ST','SV']
THROWS_CATS = ['L','R']
STAND_CATS = ['L','R']


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
    return {
        "pitch_type": type_code,
        "release_speed": pd_.get("startSpeed"),
        "release_spin_rate": breaks.get("spinRate"),
        "release_extension": pd_.get("extension"),
        "pfx_x": coords.get("pfxX"),
        "pfx_z": coords.get("pfxZ"),
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
    PITCH_ALIASES = {'FO': 'FS'}
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

