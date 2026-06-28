#!/usr/bin/env python3
"""
diagnose_live_vs_savant.py — quantify the live-API vs day-after-Savant Stuff+ gap.

Both the live game card and the nightly Savant grades run the SAME model through
the SAME code (server.engineer_and_score). The only difference is the input data:

  * live card     — MLB Stats API GUMBO feed, pfx DERIVED from kinematics
                    (_pfx_from_kinematics), spin_axis from MLB spinDirection.
  * day-after     — Savant pfx_x/pfx_z, spin_axis (the _pfx_direct path / batch).

This harness pulls BOTH sources for the same pitches of a game, scores each with
the production code, and diffs every stuff-model input + the resulting Stuff+,
grouped by pitch type. It changes nothing — read-only diagnostic.

The headline it answers (Case A vs B):
  A) GUMBO trajectory params (aX/aZ/vY0/x0/z0/extension) ≈ Savant's, but the
     derived pfx ≠ Savant pfx  → a formula/source problem, fixable in code.
  B) The GUMBO params themselves differ from Savant  → live feed is preliminary
     and revised before Savant; a code fix can't fully close it.

Run on a host with outbound access to statsapi.mlb.com AND baseballsavant.mlb.com
(the droplet), inside the API venv so `import server` loads the models:

    python3 diagnose_live_vs_savant.py --game-pk 776543 [--game-pk ...] \
        [--out live_vs_savant.csv]
"""
import argparse
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Production scoring — the exact code both the live card and the batch run through.
import server  # noqa: E402  (loads models at import)
from server import map_pitch, engineer_and_score, _pfx_from_kinematics, _per_type_plus_agg  # noqa: E402

HDRS = {"User-Agent": "Mozilla/5.0 (ptet-diagnostic)"}
STATSAPI = "https://statsapi.mlb.com"
SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"

# Fastball family (for the Case A/B verdict, per the stuff-model family definition).
FB_TYPES = {"FF", "SI", "FC"}


# ─────────────────────────────────────────────────────────────────────────────
# Data pulls
# ─────────────────────────────────────────────────────────────────────────────
def fetch_gumbo(game_pk):
    """Live GUMBO feed → list of pitch dicts keyed by (atBatIndex, pitchNumber).
    Mirrors the field extraction in the frontend's mlbApi.js."""
    url = f"{STATSAPI}/api/v1.1/game/{game_pk}/feed/live"
    data = requests.get(url, headers=HDRS, timeout=60).json()
    date = (data.get("gameData", {}).get("datetime", {}) or {}).get("officialDate")
    out = []
    for play in data.get("liveData", {}).get("plays", {}).get("allPlays", []):
        abi = play.get("atBatIndex")
        matchup = play.get("matchup", {}) or {}
        pitcher_id = (matchup.get("pitcher") or {}).get("id")
        p_throws = (matchup.get("pitchHand") or {}).get("code") or "R"
        bat_side = (matchup.get("batSide") or {}).get("code") or "R"
        for ev in play.get("playEvents", []):
            if not ev.get("isPitch") or not ev.get("pitchData"):
                continue
            pd_ = ev["pitchData"]
            co = pd_.get("coordinates", {}) or {}
            br = pd_.get("breaks", {}) or {}
            out.append({
                "key": (abi, ev.get("pitchNumber")),
                "pitcher_id": pitcher_id, "p_throws": p_throws, "stand": bat_side,
                "pitch_type": (ev.get("details", {}).get("type", {}) or {}).get("code"),
                "velo": pd_.get("startSpeed"), "extension": pd_.get("extension"),
                "szTop": pd_.get("strikeZoneTop"), "szBot": pd_.get("strikeZoneBottom"),
                "pfxX": co.get("pfxX"), "pfxZ": co.get("pfxZ"),
                "pX": co.get("pX"), "pZ": co.get("pZ"),
                "x0": co.get("x0"), "z0": co.get("z0"),
                "vX0": co.get("vX0"), "vY0": co.get("vY0"), "vZ0": co.get("vZ0"),
                "aX": co.get("aX"), "aY": co.get("aY"), "aZ": co.get("aZ"),
                "spin": br.get("spinRate"), "spinDirection": br.get("spinDirection"),
                "breakHorizontal": br.get("breakHorizontal"),
                "breakVerticalInduced": br.get("breakVerticalInduced"),
            })
    return date, out


def fetch_savant(date, game_pk):
    """Statcast rows for one date, filtered to game_pk → keyed by (atBatIndex, pitchNumber).
    atBatIndex = at_bat_number - 1 (MLB Stats API 0-indexed)."""
    params = {
        "all": "true", "hfGT": "R|", "hfSea": str(pd.Timestamp(date).year),
        "player_type": "pitcher", "game_date_gt": date, "game_date_lt": date,
        "type": "details", "csv": "true", "min_pitches": "0", "min_results": "0", "min_pas": "0",
    }
    r = requests.get(SAVANT_CSV, params=params, headers=HDRS, timeout=120)
    df = pd.read_csv(StringIO(r.text)) if r.status_code == 200 and len(r.text) > 200 else pd.DataFrame()
    if df.empty:
        return {}
    df = df[df["game_pk"] == int(game_pk)].copy()
    out = {}
    for _, r in df.iterrows():
        out[(int(r["at_bat_number"]) - 1, int(r["pitch_number"]))] = r
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Payload builders (feed the production map_pitch / engineer_and_score)
# ─────────────────────────────────────────────────────────────────────────────
def live_evt(g):
    """Live payload — mirrors the production frontend makePitch (no _pfx_direct).
    Includes breakHorizontal/breakVerticalInduced so the server's break-preferred
    pfx path is exercised; re-running post-fix should drive Δ to ~0."""
    return {
        "pitcher_id": g["pitcher_id"], "_stand": g["stand"], "_p_throws": g["p_throws"],
        "details": {"type": {"code": g["pitch_type"]}},
        "pitchData": {
            "startSpeed": g["velo"], "extension": g["extension"],
            "strikeZoneTop": g["szTop"], "strikeZoneBottom": g["szBot"],
            "coordinates": {k: g[k] for k in ("pfxX", "pfxZ", "pX", "pZ", "x0", "z0",
                                              "vX0", "vY0", "vZ0", "aX", "aY", "aZ")},
            "breaks": {"spinRate": g["spin"], "spinDirection": g["spinDirection"],
                       "breakHorizontal": g["breakHorizontal"],
                       "breakVerticalInduced": g["breakVerticalInduced"]},
        },
    }


def _f(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def savant_evt(s, pitcher_id, p_throws):
    """Savant payload: _pfx_direct → server uses Savant pfx_x/pfx_z directly."""
    return {
        "pitcher_id": pitcher_id, "_stand": s.get("stand") or "R",
        "_p_throws": s.get("p_throws") or p_throws, "_pfx_direct": True,
        "details": {"type": {"code": s.get("pitch_type")}},
        "pitchData": {
            "startSpeed": _f(s.get("release_speed")), "extension": _f(s.get("release_extension")),
            "strikeZoneTop": _f(s.get("sz_top")), "strikeZoneBottom": _f(s.get("sz_bot")),
            "coordinates": {
                "pfxX": _f(s.get("pfx_x")), "pfxZ": _f(s.get("pfx_z")),
                "pX": _f(s.get("plate_x")), "pZ": _f(s.get("plate_z")),
                "x0": _f(s.get("release_pos_x")), "z0": _f(s.get("release_pos_z")),
                "vX0": _f(s.get("vx0")), "vY0": _f(s.get("vy0")), "vZ0": _f(s.get("vz0")),
                "aX": _f(s.get("ax")), "aY": _f(s.get("ay")), "aZ": _f(s.get("az")),
            },
            "breaks": {"spinRate": _f(s.get("release_spin_rate")), "spinDirection": _f(s.get("spin_axis"))},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────
def compare_game(game_pk):
    date, gumbo = fetch_gumbo(game_pk)
    if not date or not gumbo:
        print(f"  ! no GUMBO pitches for {game_pk}")
        return []
    savant = fetch_savant(date, game_pk)
    if not savant:
        print(f"  ! no Savant rows for {game_pk} ({date})")
        return []

    paired = [(g, savant[g["key"]]) for g in gumbo if g["key"] in savant]
    if not paired:
        print(f"  ! 0 pitches joined for {game_pk}")
        return []

    # Map both sides in paired order, then keep only pitches that mapped on BOTH
    # sides — so live[i] / savant[i] always refer to the same pitch.
    live_mapped = [map_pitch(live_evt(g), g["pitcher_id"]) for g, _ in paired]
    sav_mapped = [map_pitch(savant_evt(s, g["pitcher_id"], g["p_throws"]), g["pitcher_id"])
                  for g, s in paired]
    keep = [i for i in range(len(paired)) if live_mapped[i] and sav_mapped[i]]
    if len(keep) < len(paired):
        print(f"    ({len(paired) - len(keep)} pitches dropped — unmapped on one side)")
    paired = [paired[i] for i in keep]
    live_rows = [live_mapped[i] for i in keep]
    sav_rows = [sav_mapped[i] for i in keep]
    if not paired:
        return []

    # engineer_and_score returns one row per input, tagged with "index"; sort by it
    # to guarantee positional alignment with live_rows / sav_rows.
    live_scored = sorted(engineer_and_score(live_rows), key=lambda r: r["index"])
    sav_scored = sorted(engineer_and_score(sav_rows), key=lambda r: r["index"])

    recs = []
    for i in range(len(paired)):
        g, s = paired[i]
        lr, sr = live_rows[i], sav_rows[i]
        lo, so = live_scored[i], sav_scored[i]
        pt = lo.get("pitch_type")
        sp_live = _per_type_plus_agg(lo["xRV_stuff"], server.PITCH_ALIASES.get(pt, pt), "stuff")
        sp_sav = _per_type_plus_agg(so["xRV_stuff"], server.PITCH_ALIASES.get(pt, pt), "stuff")
        # kinematic candidate computed directly (independent of what production
        # now uses) so the source ranking stays a true before/after comparison.
        kpx, kpz = _pfx_from_kinematics(g["aX"], g["aY"], g["aZ"], g["vY0"])
        recs.append({
            "game_pk": game_pk, "pitch_type": pt, "is_fb": pt in FB_TYPES,
            # raw model inputs (live = kinematic/MLB, sav = Savant)
            "pfx_x_live": lr.get("pfx_x"), "pfx_x_sav": sr.get("pfx_x"),
            "pfx_z_live": lr.get("pfx_z"), "pfx_z_sav": sr.get("pfx_z"),
            "ext_live": lr.get("release_extension"), "ext_sav": sr.get("release_extension"),
            "relx_live": lr.get("release_pos_x"), "relx_sav": sr.get("release_pos_x"),
            "relz_live": lr.get("release_pos_z"), "relz_sav": sr.get("release_pos_z"),
            "velo_live": lr.get("release_speed"), "velo_sav": sr.get("release_speed"),
            "spin_live": lr.get("release_spin_rate"), "spin_sav": sr.get("release_spin_rate"),
            "axis_live": lr.get("spin_axis"), "axis_sav": sr.get("spin_axis"),
            # raw trajectory params (Case A vs B): GUMBO vs Savant
            "aZ_g": g["aZ"], "aZ_s": _f(s.get("az")), "aX_g": g["aX"], "aX_s": _f(s.get("ax")),
            "vY0_g": g["vY0"], "vY0_s": _f(s.get("vy0")), "ext_g": g["extension"], "ext_s": _f(s.get("release_extension")),
            # candidate live pfx sources vs Savant (inches): which best reproduces Savant pfx_z?
            "savant_ivb_in": (sr.get("pfx_z") or np.nan) * 12,
            "kinematic_ivb_in": (kpz if kpz is not None else np.nan) * 12,
            "mlb_pfxZ_in": g["pfxZ"], "mlb_breakIVB_in": g["breakVerticalInduced"],
            "savant_hb_in": (sr.get("pfx_x") or np.nan) * 12,
            "kinematic_hb_in": (kpx if kpx is not None else np.nan) * 12,
            "mlb_pfxX_in": g["pfxX"], "mlb_breakHB_in": g["breakHorizontal"],
            # outputs
            "xrv_stuff_live": lo["xRV_stuff"], "xrv_stuff_sav": so["xRV_stuff"],
            "stuff_plus_live": sp_live, "stuff_plus_sav": sp_sav,
        })
    print(f"  {game_pk} ({date}): {len(recs)} pitches compared")
    return recs


def report(df):
    if df.empty:
        print("\nNo pitches compared — check connectivity / game_pk.")
        return
    d = df.copy()
    # numeric coercion — _per_type_plus_agg may return None, and missing inputs
    # arrive as None; coerce so the delta subtractions don't hit object dtype.
    for c in d.columns:
        if c not in ("pitch_type", "is_fb"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # deltas (live − savant)
    for a in ["pfx_x", "pfx_z", "ext", "relx", "relz", "velo", "spin", "axis",
              "xrv_stuff", "stuff_plus"]:
        d[f"d_{a}"] = d[f"{a}_live"] - d[f"{a}_sav"]
    d["d_axis"] = ((d["d_axis"] + 180) % 360) - 180  # circular degrees

    print("\n" + "=" * 78)
    print("PER-PITCH-TYPE Δ (live − savant)   [pfx in ft, Stuff+ in points]")
    print("=" * 78)
    agg = d.groupby("pitch_type").agg(
        n=("d_pfx_z", "size"),
        d_pfx_x=("d_pfx_x", "mean"), d_pfx_z=("d_pfx_z", "mean"),
        d_ext=("d_ext", "mean"), d_relx=("d_relx", "mean"), d_relz=("d_relz", "mean"),
        d_velo=("d_velo", "mean"), d_spin=("d_spin", "mean"), d_axis=("d_axis", "mean"),
        d_stuff_plus=("d_stuff_plus", "mean"),
    ).round(3).sort_values("n", ascending=False)
    print(agg.to_string())
    print("\n  (negative d_stuff_plus = live grades LOWER than Savant = 'jumps up day-after')")

    # Case A vs B — do the GUMBO trajectory params match Savant?
    print("\n" + "=" * 78)
    print("CASE A/B — |GUMBO − Savant| on the trajectory params (should be ~0 for Case A)")
    print("=" * 78)
    for label, mask in [("Fastballs (FF/SI/FC)", d["is_fb"]), ("Offspeed", ~d["is_fb"])]:
        sub = d[mask]
        if sub.empty:
            continue
        print(f"  {label}: n={len(sub)}  "
              f"|ΔaZ|={(sub.aZ_g - sub.aZ_s).abs().mean():.3f}  "
              f"|ΔaX|={(sub.aX_g - sub.aX_s).abs().mean():.3f}  "
              f"|ΔvY0|={(sub.vY0_g - sub.vY0_s).abs().mean():.3f}  "
              f"|Δext|={(sub.ext_g - sub.ext_s).abs().mean():.3f}")
    print("  → all ~0 ⇒ Case A (formula/source: fixable in code).  "
          "materially nonzero ⇒ Case B (live feed revised).")

    # Which live pfx source best reproduces Savant pfx_z (inches MAE)?
    print("\n" + "=" * 78)
    print("BEST LIVE pfx_z SOURCE — MAE vs Savant IVB (inches), fastballs only")
    print("=" * 78)
    fb = d[d["is_fb"]]
    for name, col in [("kinematic (current live)", "kinematic_ivb_in"),
                      ("MLB coords.pfxZ", "mlb_pfxZ_in"),
                      ("MLB breaks.breakVerticalInduced", "mlb_breakIVB_in")]:
        mae = (fb[col] - fb["savant_ivb_in"]).abs().mean()
        bias = (fb[col] - fb["savant_ivb_in"]).mean()
        print(f"  {name:34s}  MAE={mae:6.2f}\"  bias={bias:+6.2f}\"")
    print("  (lowest MAE / smallest bias = the source the live card should use)")

    # spin_axis drift (drives offspeed)
    off = d[~d["is_fb"]]
    if not off.empty:
        print(f"\nspin_axis drift (offspeed): mean|Δ|={off['d_axis'].abs().mean():.1f}°  "
              f"mean Δstuff_plus={off['d_stuff_plus'].mean():+.2f}")

    # worst offenders
    print("\nWorst 8 pitches by |Δstuff_plus|:")
    cols = ["game_pk", "pitch_type", "pfx_z_live", "pfx_z_sav", "d_stuff_plus"]
    print(d.reindex(d["d_stuff_plus"].abs().sort_values(ascending=False).index)[cols].head(8).round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game-pk", type=int, action="append", required=True,
                    help="MLB gamePk (repeatable)")
    ap.add_argument("--out", default=None, help="Write the per-pitch comparison CSV here")
    args = ap.parse_args()

    all_recs = []
    for gpk in args.game_pk:
        print(f"Game {gpk}:")
        all_recs.extend(compare_game(gpk))

    df = pd.DataFrame(all_recs)
    report(df)
    if args.out and not df.empty:
        df.to_csv(args.out, index=False)
        print(f"\nWrote per-pitch comparison → {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
