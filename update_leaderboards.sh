#!/usr/bin/env bash
# update_leaderboards.sh — Rebuild pitch modeling leaderboards and team plots.
#
# Usage:
#   ./update_leaderboards.sh [--year YYYY] [--parquet /path/to/file.parquet] [--fetch] [--no-restart]
#
#   --year YYYY          Season to score (default: 2026)
#   --parquet PATH       Existing Statcast parquet; if omitted uses default path
#   --fetch              Re-download Statcast data even if parquet already exists
#   --no-restart         Skip restarting pitch-plus-api.service
#
# What it does:
#   1. Optionally fetches fresh Statcast data via pybaseball
#   2. Runs score_pitches.py → updates season/pitcher_grades_{year}.json
#   3. Restarts pitch-plus-api.service so the API reloads pitcher grades
#   4. Runs build_team_plus.py → updates public/team_plus_{year}.json

set -euo pipefail

# ── Configurable paths ────────────────────────────────────────────────────────
API_DIR=/var/www/pitch-plus-api
FRONTEND_DIR=/var/www/pasttheeyetest.com
# Score against the API's OWN models dir so the leaderboard build and the running
# server share one pitch_plus_norm.json (incl. _stuff_plus_rescale). Otherwise the
# leaderboard rescales Stuff+ to mean=100 while live /score_aggregate reads an
# un-rescaled API copy → live grades come out systematically lower.
MODELS_DIR=$API_DIR/models
CONFIG=$MODELS_DIR/final_model_config.json
# ─────────────────────────────────────────────────────────────────────────────

YEAR=2026
PARQUET=""
FETCH=0
NO_RESTART=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --year)      YEAR="$2"; shift 2 ;;
        --parquet)   PARQUET="$2"; shift 2 ;;
        --fetch)     FETCH=1; shift ;;
        --no-restart) NO_RESTART=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$PARQUET" ]]; then
    PARQUET="$FRONTEND_DIR/pitch_xrv_${YEAR}.parquet"
fi

# ── Resolve Python interpreter (prefer a project venv with lightgbm/pandas) ───
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python3" ]]; then
    PY="$VIRTUAL_ENV/bin/python3"
else
    PY=""
    for cand in "$API_DIR/.venv/bin/python3" "$API_DIR/venv/bin/python3" \
                "$FRONTEND_DIR/.venv/bin/python3" "$FRONTEND_DIR/venv/bin/python3"; do
        if [[ -x "$cand" ]]; then PY="$cand"; break; fi
    done
    [[ -z "$PY" ]] && PY="python3"
fi
echo "Using Python: $PY"
if ! "$PY" -c "import lightgbm" 2>/dev/null; then
    echo "ERROR: lightgbm not available in $PY." >&2
    echo "       Activate the venv (source .venv/bin/activate) or install deps." >&2
    exit 1
fi

# ── Step 0: Fetch Statcast if needed ─────────────────────────────────────────
if [[ ! -f "$PARQUET" ]] || [[ "$FETCH" -eq 1 ]]; then
    echo "=== Fetching Statcast ${YEAR} ==="
    "$PY" - <<PYEOF
import sys
try:
    from pybaseball import statcast
except ImportError:
    print("ERROR: pybaseball not installed. Run: pip install pybaseball", file=sys.stderr)
    sys.exit(1)
import pandas as pd
from pathlib import Path

year = int("$YEAR")
parquet = Path("$PARQUET")
parquet.parent.mkdir(parents=True, exist_ok=True)

print(f"  Fetching {year}-03-15 → {year}-11-15 ...")
df = statcast(start_dt=f"{year}-03-15", end_dt=f"{year}-11-15")
if df is None or len(df) == 0:
    print(f"ERROR: No Statcast data returned for {year}.", file=sys.stderr)
    sys.exit(1)
print(f"  {len(df):,} pitches fetched")
df.to_parquet(parquet, index=False)
print(f"  Wrote {parquet}")
PYEOF
else
    echo "=== Using cached parquet: $PARQUET ==="
fi

# ── Step 1: Score pitches → update pitcher leaderboard ───────────────────────
echo ""
echo "=== Step 1: Scoring pitches for ${YEAR} ==="
"$PY" "$API_DIR/score_pitches.py" \
    --input    "$PARQUET" \
    --models   "$MODELS_DIR" \
    --config   "$CONFIG" \
    --output-dir "$API_DIR" \
    --season   "$YEAR"

echo "  Leaderboard : $API_DIR/season/pitcher_grades_${YEAR}.json"
echo "  Pitch types : $API_DIR/season/pitcher_pitch_type_grades_${YEAR}.json"

# ── Step 2: Restart API to reload season grades ───────────────────────────────
if [[ "$NO_RESTART" -eq 0 ]]; then
    echo ""
    echo "=== Step 2: Restarting pitch-plus-api.service ==="
    if systemctl is-active --quiet pitch-plus-api.service 2>/dev/null; then
        systemctl restart pitch-plus-api.service
        echo "  Service restarted."
    else
        echo "  Service not running — skipping restart."
    fi
else
    echo ""
    echo "=== Step 2: Skipping service restart (--no-restart) ==="
fi

# ── Step 3: Build team plots ──────────────────────────────────────────────────
echo ""
echo "=== Step 3: Building team plots for ${YEAR} ==="
# PYTHONPATH=API_DIR so build_team_plus.py imports the v3.4 score_pitches.py
(cd "$FRONTEND_DIR" && PYTHONPATH="$API_DIR:${PYTHONPATH:-}" "$PY" build_team_plus.py --year "$YEAR" --parquet "$PARQUET")

echo "  Team plots  : $FRONTEND_DIR/public/team_plus_${YEAR}.json"

# ── Step 4: Build hitter xRV/600 PA card data ────────────────────────────────
# The hitter xRV model (models/xrv_model_2022_2025.pkl) is cloudpickled Python
# 3.12 bytecode + scikit-learn 1.6.1 and SEGFAULTS under the lightgbm venv's
# Python. It needs its OWN interpreter. Resolve one (env XRV_PY wins), else a
# .venv-xrv alongside the repos. If none is found, warn and skip — this step is
# optional and must never fail the pitcher pipeline.
echo ""
echo "=== Step 4: Building hitter xRV for ${YEAR} ==="
XRV_PY="${XRV_PY:-}"
if [[ -z "$XRV_PY" ]]; then
    for cand in "$FRONTEND_DIR/.venv-xrv/bin/python" "$API_DIR/.venv-xrv/bin/python"; do
        if [[ -x "$cand" ]]; then XRV_PY="$cand"; break; fi
    done
fi
if [[ -z "$XRV_PY" || ! -x "$XRV_PY" ]]; then
    echo "  ⚠ No xRV interpreter found (set XRV_PY or create $FRONTEND_DIR/.venv-xrv)."
    echo "    Skipping hitter xRV. To enable:"
    echo "      uv venv --python 3.12 $FRONTEND_DIR/.venv-xrv"
    echo "      $FRONTEND_DIR/.venv-xrv/bin/python -m pip install scikit-learn==1.6.1 cloudpickle 'numpy>=2,<3' 'pandas>=2,<3' pyarrow pybaseball"
elif ! "$XRV_PY" -c "import sklearn, cloudpickle" 2>/dev/null; then
    echo "  ⚠ $XRV_PY is missing scikit-learn/cloudpickle — skipping hitter xRV."
else
    (cd "$FRONTEND_DIR" && "$XRV_PY" build_hitter_xrv.py --season "$YEAR" --parquet "$PARQUET")
    echo "  Hitter xRV  : $FRONTEND_DIR/public/hitter_xrv_${YEAR}.json"
    echo "  xRV (games) : $FRONTEND_DIR/public/hitter_xrv_games_${YEAR}.json"
fi

echo ""
echo "=== Done. ==="
