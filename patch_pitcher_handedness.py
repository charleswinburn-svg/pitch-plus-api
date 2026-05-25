#!/usr/bin/env python3
"""
patch_pitcher_handedness.py — One-time script to add p_throws to model files.

Queries the MLB Stats API for pitchHand.code for all pitchers in
pitcher_baselines.json and pitcher_arm_angles.json, then writes the
p_throws field into both files so the server's pitcher_p_throws dict
is populated on next startup.

Usage:
    python3 patch_pitcher_handedness.py
    python3 patch_pitcher_handedness.py --dry-run   # print counts, don't write
"""
import argparse, json, time
from pathlib import Path
import requests

BASELINES_PATH  = Path("models/pitcher_baselines.json")
ARM_ANGLES_PATH = Path("models/pitcher_arm_angles.json")
MLB_PEOPLE_URL  = "https://statsapi.mlb.com/api/v1/people"
BATCH_SIZE      = 200


def fetch_handedness(pids: list[int]) -> dict[int, str]:
    """Return {pitcher_id: "L" or "R"} for a batch of pids."""
    ids_str = ",".join(str(p) for p in pids)
    try:
        r = requests.get(
            MLB_PEOPLE_URL,
            params={"personIds": ids_str, "fields": "people,id,pitchHand,code"},
            timeout=30,
        )
        r.raise_for_status()
        out = {}
        for person in r.json().get("people", []):
            pid = person.get("id")
            code = (person.get("pitchHand") or {}).get("code")
            if pid and code in ("L", "R"):
                out[int(pid)] = code
        return out
    except Exception as e:
        print(f"  WARNING: batch failed ({e})")
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bl  = json.loads(BASELINES_PATH.read_text())
    aa  = json.loads(ARM_ANGLES_PATH.read_text())

    all_pids = sorted({int(k) for k in list(bl.keys()) + list(aa.keys())})
    print(f"Fetching handedness for {len(all_pids)} pitchers in {(len(all_pids)+BATCH_SIZE-1)//BATCH_SIZE} batches...")

    handedness: dict[int, str] = {}
    for i in range(0, len(all_pids), BATCH_SIZE):
        batch = all_pids[i : i + BATCH_SIZE]
        result = fetch_handedness(batch)
        handedness.update(result)
        print(f"  [{i+len(batch)}/{len(all_pids)}] resolved {len(result)}/{len(batch)}")
        if i + BATCH_SIZE < len(all_pids):
            time.sleep(0.5)

    print(f"\nResolved {len(handedness)}/{len(all_pids)} pitchers")
    lhp = sum(1 for v in handedness.values() if v == "L")
    rhp = sum(1 for v in handedness.values() if v == "R")
    print(f"  LHP: {lhp}  RHP: {rhp}")

    if args.dry_run:
        print("\n--dry-run: not writing files")
        return

    # Patch baselines
    n_bl = 0
    for pid_str, rec in bl.items():
        pt = handedness.get(int(pid_str))
        if pt:
            rec["p_throws"] = pt
            n_bl += 1
    BASELINES_PATH.write_text(json.dumps(bl, indent=2))
    print(f"\nWrote {n_bl}/{len(bl)} p_throws entries to {BASELINES_PATH}")

    # Patch arm angles
    n_aa = 0
    for pid_str, rec in aa.items():
        pt = handedness.get(int(pid_str))
        if pt:
            rec["p_throws"] = pt
            n_aa += 1
    ARM_ANGLES_PATH.write_text(json.dumps(aa, separators=(",", ":")))
    print(f"Wrote {n_aa}/{len(aa)} p_throws entries to {ARM_ANGLES_PATH}")

    # Summary of unresolved
    missing = [p for p in all_pids if p not in handedness]
    if missing:
        print(f"\n{len(missing)} unresolved (not found in MLB API): {missing[:10]}{'...' if len(missing)>10 else ''}")


if __name__ == "__main__":
    main()
