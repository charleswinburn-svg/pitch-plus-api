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
MODELS_DIR=$FRONTEND_DIR/models
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

# ── Step 0: Fetch Statcast if needed ─────────────────────────────────────────
if [[ ! -f "$PARQUET" ]] || [[ "$FETCH" -eq 1 ]]; then
    echo "=== Fetching Statcast ${YEAR} ==="
    python3 - <<PYEOF
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
python3 "$API_DIR/score_pitches.py" \
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
# Run from FRONTEND_DIR so build_team_plus.py imports the local score_pitches.py
(cd "$FRONTEND_DIR" && python3 build_team_plus.py --year "$YEAR" --parquet "$PARQUET")

echo "  Team plots  : $FRONTEND_DIR/public/team_plus_${YEAR}.json"

echo ""
echo "=== Done. ==="
