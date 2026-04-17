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
stuff_model = lgb.Booster(model_file=str(MODELS_DIR / "stuff_model_2025.txt"))
with open(MODELS_DIR / "stuff_model_metadata.json") as f:
    stuff_features = json.load(f)["features"]

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

# Per-pitch-type Pitch+ league mean/std (computed from 2025 season aggregates)
with open(MODELS_DIR / "pitch_plus_norm.json") as f:
    pitch_plus_norm = json.load(f)  # {pt: {mean, std}}

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


def engineer_and_score(rows: list[dict]) -> list[dict]:
    """Run feature engineering + 3-stage model on a list of mapped pitches."""
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

    # ── Stuff features ──
    t = np.clip(50.0 / (-df['vy0'].values), 0.35, 0.55)
    df['vaa'] = np.degrees(np.arctan2(df['vz0'].values + df['az'].values * t,
                                       -(df['vy0'].values + df['ay'].values * t)))
    df['haa'] = np.degrees(np.arctan2(df['vx0'].values + df['ax'].values * t,
                                       -(df['vy0'].values + df['ay'].values * t)))
    df['total_movement'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    df['spin_efficiency'] = df['total_movement'] / df['release_spin_rate'].replace(0, np.nan) * 1000

    # Deltas vs pitcher's fastball baseline
    for col in ['delta_velo','delta_pfx_x','delta_pfx_z','delta_spin','delta_extension','delta_vaa',
                'release_diff_x','release_diff_z','release_distance','movement_separation']:
        df[col] = 0.0

    for i, row in df.iterrows():
        pid = str(int(row['pitcher'])) if pd.notna(row.get('pitcher')) else None
        bl = pitcher_baselines.get(pid) if pid else None
        if bl:
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

    df['pitch_type_cat'] = pd.Categorical(df['pitch_type'], categories=PITCH_TYPE_CATS).codes
    df['p_throws_cat'] = pd.Categorical(df['p_throws'], categories=THROWS_CATS).codes
    df['stand_cat'] = pd.Categorical(df['stand'], categories=STAND_CATS).codes
    df['same_side'] = (df['p_throws'] == df['stand']).astype(int)

    # ── Stuff prediction ──
    X_stuff = df[stuff_features].values
    df['xRV_stuff'] = stuff_model.predict(X_stuff)

    # ── Location features ──
    df['plate_x_adj'] = np.where(df['stand']=='L', -df['plate_x'], df['plate_x'])
    df['zone_center_dist'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - 2.5)**2)
    df['in_zone'] = ((df['plate_x'].abs() <= 0.83) &
                     (df['plate_z'] >= 1.5) & (df['plate_z'] <= 3.5)).astype(int)
    df['out_of_zone_dist'] = np.where(df['in_zone']==1, 0.0,
                                       np.maximum(0, df['plate_x'].abs() - 0.83) +
                                       np.maximum(0, 1.5 - df['plate_z']) +
                                       np.maximum(0, df['plate_z'] - 3.5))

    df['xRV_location'] = 0.0
    for pt, model in location_models.items():
        mask = df['pitch_type'] == pt
        if mask.sum() > 0:
            X = df.loc[mask, location_features].values
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

    X_tunnel = df[tunnel_features].values
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
        norm = pitch_plus_norm.get(pt)
        if norm and norm['std'] > 0:
            z = (row['xRV_final'] - norm['mean']) / norm['std']
            z = max(-4, min(4, z))  # clamp extreme outliers
            pitch_plus = round(100 - z * 10, 1)
        else:
            pitch_plus = None
        out.append({
            "index": int(i),
            "pitch_type": pt_display,
            "pitch_plus": pitch_plus,
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": len(location_models),
        "baselines": len(pitcher_baselines),
        "pitch_type_cats": PITCH_TYPE_CATS,  # debug: confirm which ordering is deployed
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
    scored = engineer_and_score(rows)
    return {"scores": scored}
