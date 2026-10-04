#!/usr/bin/env bash
# update_aaa_pitchers.sh — Build the AAA pitcher-card data: Stuff+/Loc+/Tun+/Pitch+
# season grades, arsenal panel and grade distributions, from AAA Statcast.
#
# AAA pitches are graded on the MLB scale (MLB models + pitch_plus_norm.json and its
# stored rescale, all READ-ONLY) but kept completely separate from MLB: they live in
# pitch_aaa_{year}.parquet and only ever produce *_aaa_* files. Every MLB file is
# checksummed before and after the run; any change aborts with an error.
#
# Usage:
#   ./update_aaa_pitchers.sh [--year YYYY] [--full] [--no-fetch] [--no-restart]
#
#   --year YYYY    Season (default: current year)
#   --full         Re-pull the whole AAA season from Savant (default: new days only)
#   --no-fetch     Skip the Savant pull; rebuild from the existing AAA parquet
#   --no-restart   Skip restarting pitch-plus-api.service (use during a multi-year backfill)
#
# Writes:
#   $API_DIR/season/pitcher_grades_aaa_{year}.json            (served by /pitcher_percentiles?level=aaa)
#   $API_DIR/season/pitcher_pitch_type_grades_aaa_{year}.json
#   $FRONTEND_DIR/public/pitcher_arsenal_aaa_{year}.json
#   $FRONTEND_DIR/public/pitcher_grade_dist_aaa_{year}.json
#
# One-time backfill:
#   for y in 2023 2024 2025 2026; do ./update_aaa_pitchers.sh --year $y --full --no-restart; done
#   systemctl restart pitch-plus-api.service

set -euo pipefail

API_DIR="${API_DIR:-/var/www/pitch-plus-api}"
FRONTEND_DIR="${FRONTEND_DIR:-/var/www/pasttheeyetest.com}"
MODELS_DIR=$API_DIR/models
CONFIG=$MODELS_DIR/final_model_config.json

YEAR=$(date +%Y)
FULL=0
NO_FETCH=0
NO_RESTART=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --year)       YEAR="$2"; shift 2 ;;
        --full)       FULL=1; shift ;;
        --no-fetch)   NO_FETCH=1; shift ;;
        --no-restart) NO_RESTART=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

PARQUET="$FRONTEND_DIR/pitch_aaa_${YEAR}.parquet"
OUT_GRADES="$API_DIR/season/pitcher_grades_aaa_${YEAR}.json"
OUT_ARSENAL="$FRONTEND_DIR/public/pitcher_arsenal_aaa_${YEAR}.json"
OUT_DIST="$FRONTEND_DIR/public/pitcher_grade_dist_aaa_${YEAR}.json"

# ── Resolve Python interpreter (same rule as update_leaderboards.sh) ──────────
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
if ! "$PY" -c "import lightgbm, pandas, pyarrow, requests" 2>/dev/null; then
    echo "ERROR: $PY needs lightgbm, pandas, pyarrow and requests." >&2
    exit 1
fi

# ── Separation guard: checksum every MLB file this pipeline could conceivably touch ─
mlb_files() {
    ls "$MODELS_DIR"/pitch_plus_norm.json "$MODELS_DIR"/pitcher_baselines.json \
       "$MODELS_DIR"/pitcher_arm_angles.json 2>/dev/null
    ls "$API_DIR"/season/pitcher_grades_[0-9]*.json "$API_DIR"/season/pitcher_pitch_type_grades_[0-9]*.json 2>/dev/null
    ls "$FRONTEND_DIR"/public/pitcher_arsenal_[0-9]*.json "$FRONTEND_DIR"/public/pitcher_grade_dist_[0-9]*.json 2>/dev/null
}
MLB_SUMS=$(mlb_files | xargs -r md5sum)

# ── Step 1: AAA Statcast (incremental) ───────────────────────────────────────
BEFORE=$(stat -c %Y "$PARQUET" 2>/dev/null || echo 0)
if [[ "$NO_FETCH" -eq 0 ]]; then
    echo "=== Step 1: Fetching AAA Statcast ${YEAR} ==="
    FULL_FLAG=()
    [[ "$FULL" -eq 1 ]] && FULL_FLAG=(--full)
    (cd "$FRONTEND_DIR" && "$PY" fetch_statcast_aaa.py --year "$YEAR" --out "$PARQUET" ${FULL_FLAG[@]+"${FULL_FLAG[@]}"})
fi
if [[ ! -f "$PARQUET" ]]; then
    echo "No AAA parquet for ${YEAR} ($PARQUET) — nothing to build."
    exit 0
fi
AFTER=$(stat -c %Y "$PARQUET")
if [[ "$NO_FETCH" -eq 0 && "$BEFORE" == "$AFTER" && -f "$OUT_GRADES" && -f "$OUT_ARSENAL" && -f "$OUT_DIST" ]]; then
    echo "No new AAA pitches since the last build — outputs are current."
    exit 0
fi

# ── Step 2: AAA season grades on the MLB scale (separate *_aaa_* files) ──────
echo ""
echo "=== Step 2: Scoring AAA pitches (MLB scale) ==="
"$PY" "$API_DIR/score_pitches.py" \
    --input "$PARQUET" --models "$MODELS_DIR" --config "$CONFIG" \
    --output-dir "$API_DIR" --season "$YEAR" --level aaa

# ── Step 3: AAA arsenal panel ────────────────────────────────────────────────
echo ""
echo "=== Step 3: AAA arsenal ==="
(cd "$FRONTEND_DIR" && "$PY" build_pitcher_arsenal.py --season "$YEAR" --level aaa \
    --parquet "$PARQUET" --output-dir ./public)

# ── Step 4: AAA grade distributions (API code + models, MLB scale) ───────────
echo ""
echo "=== Step 4: AAA grade distributions ==="
(cd "$FRONTEND_DIR" && PITCH_PLUS_API_DIR="$API_DIR" "$PY" build_pitcher_grade_dist.py \
    --season "$YEAR" --level aaa --parquet "$PARQUET" --models "$MODELS_DIR" --output-dir ./public)

# ── Separation check: no MLB file may have changed ───────────────────────────
if [[ -n "$MLB_SUMS" ]] && ! echo "$MLB_SUMS" | md5sum -c --quiet - ; then
    echo "ERROR: an MLB file changed during the AAA build — AAA must never touch MLB data." >&2
    exit 1
fi
echo ""
echo "MLB files unchanged ($(echo "$MLB_SUMS" | grep -c .) checked)."

# ── Step 5: restart the API so /pitcher_percentiles?level=aaa serves the new grades ─
if [[ "$NO_RESTART" -eq 0 ]]; then
    if systemctl is-active --quiet pitch-plus-api.service 2>/dev/null; then
        systemctl restart pitch-plus-api.service
        echo "Restarted pitch-plus-api.service"
    else
        echo "pitch-plus-api.service not running — skipping restart"
    fi
fi

echo ""
echo "=== Done: AAA ${YEAR} ==="
echo "  $OUT_GRADES"
echo "  $OUT_ARSENAL"
echo "  $OUT_DIST"
