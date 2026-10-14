"""Experiment L: SICKLE's recalled yields against the state's own records.

    python src/official_records.py

Executes experiment_l of configs/extension_protocol.yaml. OBSERVED, not
simulated: every number here is either a SICKLE label or a figure the Tamil
Nadu Agriculture Department published (data/official/, provenance in
data/official/SOURCES.yaml).

Every result in this project rests on farmer-recalled yield. Nothing so far
checked that label against anything independent. Two checks do that here:

  L1  LEVEL   Is SICKLE's recalled yield at the level of the state's
              crop-cutting yields for the delta, once the units agree? The
              state reports paddy "in terms of rice", and its own conversion
              table puts cleaned rice at 2/3 of paddy.
  L2  EVENT   In 2020-21 the state paid crop-damage relief across every core
              delta district (cyclone Burevi, December 2020, and the
              "unprecedented heavy rain" of January 2021). Every SICKLE
              Samba/Thaladi crop of that season was in the field in January
              2021. Did recalled yields at the same plots fall relative to
              2019-20, when relief in the delta was small? And did the
              satellite model's estimates fall with them?

Neither check can say whether a given SICKLE plot was damaged: relief is
paid per district and the plots carry no district (INSURANCE_ASSUMPTIONS.md).

Writes results/metrics/official_records.json.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import farmer_report as fr  # noqa: E402

PROTOCOL = PROJ / "configs" / "extension_protocol.yaml"
OFF = PROJ / "data" / "official"
OUT = PROJ / "results" / "metrics" / "official_records.json"

ACRE_HA = 0.404686                  # hectares per acre
RICE_FROM_PADDY = 2.0 / 3.0         # TN AGRISNET Table XXVI.9
CORE_2019_20 = ["Thanjavur", "Thiruvarur", "Nagapattinam"]
MIN_CELL = 20
ALIAS = {"Tiruvarur": "Thiruvarur", "Cuddallore": "Cuddalore",
         "Tiruchirapalli": "Trichy", "Pudukottai": "Pudukkottai"}
CORE_ALL = ["Thanjavur", "Thiruvarur", "Nagapattinam", "Mayiladuthurai"]


def load_protocol():
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf8"))
    if not p.get("frozen_before_results"):
        raise SystemExit("extension protocol is not marked frozen")
    L = p["experiment_l"]
    if L["label"] != "OBSERVED":
        raise SystemExit("experiment L must be labelled OBSERVED")
    if "2/3" not in L["official_data"]["conversion"]:
        raise SystemExit("protocol conversion factor is not 2/3")
    if "fewer than 20" not in L["l1_level"]["cells"]:
        raise SystemExit("minimum cell size differs from this script")
    for d in CORE_2019_20:
        if d not in L["l1_level"]["official_band"]:
            raise SystemExit(f"{d} missing from the protocol's official band")
    return p


# ------------------------------------------------------------------ data
def official():
    rice = {}
    with (OFF / "tn_rice_district_season_2019_20.csv").open(encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            rice[ALIAS.get(r["district"], r["district"])] = r
    relief = []
    with (OFF / "tn_sdrf_crop_relief_2019_2021.csv").open(encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            r["district"] = ALIAS.get(r["district"], r["district"])
            relief.append(r)
    return rice, relief


def f_or_none(x):
    return float(x) if str(x).strip() not in ("", "-") else None


def plot_boot(values: np.ndarray, plots: np.ndarray, nb: int, seed=0):
    """Mean and 95 % interval, resampling plots, not field-seasons."""
    uniq = np.unique(plots)
    pos = {g: np.where(plots == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(nb)
    for b in range(nb):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        out[b] = values[np.concatenate([pos[g] for g in pick])].mean()
    return float(values.mean()), [float(np.percentile(out, 2.5)),
                                  float(np.percentile(out, 97.5))], out


# ------------------------------------------------------------------- L1
def level_check(P, feats, rice):
    L = P["experiment_l"]
    nb = P["bootstrap"]["replicates"]
    smap = L["season_map"]
    col = {"Kar/Kuruvai/Sornavari": "yield_kuruvai_kg_ha",
           "Samba/Thaladi/Pishanam": "yield_samba_thaladi_kg_ha",
           "Navarai/Kodai": "yield_navarai_kg_ha"}
    y = np.array([r["yield"] for r in feats])
    plots = np.array([r["plot"] for r in feats])
    season = np.array([r["season"] for r in feats])
    year = np.array([int(float(r["year"])) for r in feats])
    delta_all = [d for d, r in rice.items()
                 if r["delta_status"] in ("core", "partial")]

    out = {"unit_note": "SICKLE kg/acre paddy -> kg/ha (/0.404686) -> "
                        "rice-equivalent (x 2/3); official kg/ha rice",
           "cells": {}, "not_judged": {}}
    print("=" * 86)
    print("L1  LEVEL: SICKLE recalled yield against the state's crop-cutting "
          "yield, 2019-20")
    print("=" * 86)
    for official_season, sickle_seasons in smap.items():
        sel = np.isin(season, sickle_seasons) & (year == 2019)
        n = int(sel.sum())
        key = f"{official_season} 2019-20"
        if n < MIN_CELL:
            out["not_judged"][key] = {"n_field_seasons": n,
                                      "reason": f"fewer than {MIN_CELL}"}
            print(f"  {key:<34s} n={n:<4d} not judged (fewer than {MIN_CELL})")
            continue
        paddy_ha = y[sel] / ACRE_HA
        res = {"n_field_seasons": n,
               "n_plots": int(len(np.unique(plots[sel]))),
               "sickle_seasons": sickle_seasons}
        core_vals = {d: float(rice[d][col[official_season]])
                     for d in CORE_2019_20}
        # production-weighted official yield for the core districts
        a_col = col[official_season].replace("yield_", "area_").replace(
            "_kg_ha", "_ha")
        p_col = col[official_season].replace("yield_", "prod_").replace(
            "_kg_ha", "_t")
        A = sum(float(rice[d][a_col]) for d in CORE_2019_20)
        Pt = sum(float(rice[d][p_col]) for d in CORE_2019_20)
        core_weighted = 1000.0 * Pt / A
        all_vals = {d: f_or_none(rice[d][col[official_season]])
                    for d in delta_all}
        all_vals = {d: v for d, v in all_vals.items() if v is not None}
        res["official_core_districts"] = core_vals
        res["official_core_band"] = [min(core_vals.values()),
                                     max(core_vals.values())]
        res["official_core_area_weighted"] = core_weighted
        res["official_all_delta_band"] = [min(all_vals.values()),
                                          max(all_vals.values())]
        res["by_conversion"] = {}
        for lab, fac in (("2/3 (official)", RICE_FROM_PADDY),
                         ("0.62", 0.62), ("0.70", 0.70)):
            m, ci, _ = plot_boot(paddy_ha * fac, plots[sel], nb)
            lo_b, hi_b = res["official_core_band"]
            verdict = ("CONSISTENT" if ci[1] >= lo_b and ci[0] <= hi_b
                       else "LOWER" if ci[1] < lo_b else "HIGHER")
            lo_a, hi_a = res["official_all_delta_band"]
            v_all = ("CONSISTENT" if ci[1] >= lo_a and ci[0] <= hi_a
                     else "LOWER" if ci[1] < lo_a else "HIGHER")
            res["by_conversion"][lab] = {
                "sickle_mean_rice_eq_kg_ha": m, "ci95": ci,
                "ratio_to_core_weighted": m / core_weighted,
                "verdict_core_band": verdict,
                "verdict_all_delta_band": v_all}
        res["sickle_mean_paddy_kg_ha"] = float(paddy_ha.mean())
        res["verdict"] = res["by_conversion"]["2/3 (official)"][
            "verdict_core_band"]
        out["cells"][key] = res
        r0 = res["by_conversion"]["2/3 (official)"]
        print(f"  {key:<34s} n={n:<4d} SICKLE {r0['sickle_mean_rice_eq_kg_ha']:6.0f} "
              f"[{r0['ci95'][0]:.0f}, {r0['ci95'][1]:.0f}] kg/ha rice-eq   "
              f"official core {res['official_core_band'][0]:.0f}-"
              f"{res['official_core_band'][1]:.0f}  ratio "
              f"{r0['ratio_to_core_weighted']:.2f}  -> {res['verdict']}")
        for lab in ("0.62", "0.70"):
            rr = res["by_conversion"][lab]
            print(f"  {'':<34s}       sensitivity x{lab}: "
                  f"{rr['sickle_mean_rice_eq_kg_ha']:6.0f} -> "
                  f"{rr['verdict_core_band']}; all-delta band "
                  f"{res['official_all_delta_band'][0]:.0f}-"
                  f"{res['official_all_delta_band'][1]:.0f} -> "
                  f"{rr['verdict_all_delta_band']}")
    print()
    return out


# ------------------------------------------------------------------- L2
def relief_summary(relief):
    """Agriculture relief in the core delta districts, by season."""
    s = {"2019-20 Samba (north-east monsoon 2019)": {},
         "2020-21 Samba (Burevi Dec 2020 + January 2021 rain)": {}}
    for r in relief:
        if r["district"] not in CORE_ALL:
            continue
        amt = float(r["agri_allocated_or_sanctioned_lakh"] or 0)
        if r["table"] == "XXII.7":
            s["2019-20 Samba (north-east monsoon 2019)"][r["district"]] = amt
        elif r["table"] in ("XXII.10", "XXII.11"):
            k = "2020-21 Samba (Burevi Dec 2020 + January 2021 rain)"
            s[k][r["district"]] = s[k].get(r["district"], 0.0) + amt
    tot = {k: float(sum(v.values())) for k, v in s.items()}
    return {"agri_relief_lakh_by_district": s, "total_lakh": tot,
            "districts_with_relief_2020_21": sorted(
                s["2020-21 Samba (Burevi Dec 2020 + January 2021 rain)"]),
            "note": "2019-20 is amount sanctioned (XXII.7); 2020-21 is fund "
                    "allocation (XXII.10, XXII.11). Agriculture column only."}


def paired_change(feats, values, seasons):
    """Per plot: mean in 2020 over mean in 2019, minus one."""
    acc = defaultdict(lambda: {2019: [], 2020: []})
    for i, r in enumerate(feats):
        yr = int(float(r["year"]))
        if r["season"] in seasons and yr in (2019, 2020):
            acc[r["plot"]][yr].append(values[i])
    plots, ch = [], []
    for p, d in sorted(acc.items()):
        if d[2019] and d[2020]:
            plots.append(p)
            ch.append(np.mean(d[2020]) / np.mean(d[2019]) - 1.0)
    return np.array(plots), np.array(ch)


def event_check(P, feats, oof, relief):
    nb = P["bootstrap"]["replicates"]
    smap = P["experiment_l"]["season_map"]
    samba = smap["Samba/Thaladi/Pishanam"]
    kuru = smap["Kar/Kuruvai/Sornavari"]
    y = np.array([r["yield"] for r in feats])

    # exposure: Samba/Thaladi 2020-21 crops standing in January 2021
    jan1, jan31 = datetime(2021, 1, 1), datetime(2021, 1, 31)
    s20 = [r for r in feats if r["season"] in samba
           and int(float(r["year"])) == 2020]
    standing = sum(1 for r in s20 if r["harv"] >= jan1 and r["sow"] <= jan31)

    out = {"relief": relief_summary(relief),
           "exposure": {"n_samba_2020_21": len(s20),
                        "n_standing_in_january_2021": standing,
                        "harvest_range": [min(r["harv"] for r in s20).date().isoformat(),
                                          max(r["harv"] for r in s20).date().isoformat()]}}
    print("=" * 86)
    print("L2  EVENT: the season the state paid crop-damage relief across the "
          "delta")
    print("=" * 86)
    for k, v in out["relief"]["total_lakh"].items():
        print(f"  official agriculture relief, core delta districts, {k}: "
              f"Rs {v:,.0f} lakh")
    print(f"  SICKLE Samba/Thaladi 2020-21 field-seasons standing in January "
          f"2021: {standing} of {len(s20)}")

    res = {}
    rng_seed = {"recalled": 0, "model": 1, "control": 2}
    for lab, vals, seas in (("recalled", y, samba), ("model", oof, samba),
                            ("control", y, kuru)):
        pl, ch = paired_change(feats, vals, seas)
        m, ci, boots = plot_boot(ch, pl, nb, seed=rng_seed[lab])
        res[lab] = {"n_plots": int(len(pl)), "mean_change": m, "ci95": ci,
                    "_boots": boots}
    # contrast: Samba minus Kuruvai, independent bootstraps
    d = res["recalled"]["_boots"] - res["control"]["_boots"]
    contrast = {"delta": res["recalled"]["mean_change"]
                - res["control"]["mean_change"],
                "ci95": [float(np.percentile(d, 2.5)),
                         float(np.percentile(d, 97.5))]}
    # recalled minus model, paired on the same plots
    pl_r, ch_r = paired_change(feats, y, samba)
    pl_m, ch_m = paired_change(feats, oof, samba)
    assert np.array_equal(pl_r, pl_m)
    gm, gci, _ = plot_boot(ch_r - ch_m, pl_r, nb, seed=3)
    gap = {"delta": gm, "ci95": gci}
    for v in res.values():
        v.pop("_boots")

    verdict = {
        "recalled_registers_damage_season": bool(res["recalled"]["ci95"][1] < 0),
        "decline_is_event_specific": bool(contrast["ci95"][1] < 0),
        "model_registers_damage_season": bool(res["model"]["ci95"][1] < 0)}
    out.update({"samba_recalled": res["recalled"], "samba_model": res["model"],
                "kuruvai_control": res["control"],
                "contrast_samba_minus_kuruvai": contrast,
                "recalled_minus_model": gap, "verdict": verdict})
    print(f"  recalled Samba change, {res['recalled']['n_plots']} plots: "
          f"{res['recalled']['mean_change']:+.3f} "
          f"[{res['recalled']['ci95'][0]:+.3f}, {res['recalled']['ci95'][1]:+.3f}]"
          f"  -> {'REGISTERS' if verdict['recalled_registers_damage_season'] else 'NOT DEMONSTRATED'}")
    print(f"  Kuruvai control, {res['control']['n_plots']} plots: "
          f"{res['control']['mean_change']:+.3f} "
          f"[{res['control']['ci95'][0]:+.3f}, {res['control']['ci95'][1]:+.3f}]")
    print(f"  Samba minus Kuruvai: {contrast['delta']:+.3f} "
          f"[{contrast['ci95'][0]:+.3f}, {contrast['ci95'][1]:+.3f}]  -> "
          f"{'EVENT-SPECIFIC' if verdict['decline_is_event_specific'] else 'NOT DEMONSTRATED'}")
    print(f"  model (out-of-fold) Samba change: "
          f"{res['model']['mean_change']:+.3f} "
          f"[{res['model']['ci95'][0]:+.3f}, {res['model']['ci95'][1]:+.3f}]"
          f"  -> {'REGISTERS' if verdict['model_registers_damage_season'] else 'NOT DEMONSTRATED'}")
    print(f"  recalled minus model, paired: {gap['delta']:+.3f} "
          f"[{gap['ci95'][0]:+.3f}, {gap['ci95'][1]:+.3f}]")
    print()
    return out


def main() -> int:
    P = load_protocol()
    feats = fr.load()
    oof = np.load(PROJ / "results/predictions/oof_pred.npy")
    if len(oof) != len(feats):
        raise SystemExit("oof_pred.npy is misaligned with the feature table")
    rice, relief = official()
    print("OBSERVED. SICKLE labels are farmer-recalled paddy yield; official "
          "yields are crop-cutting")
    print("estimates in terms of cleaned rice. Neither check can say whether "
          "a given plot was damaged.")
    print()
    res = {"label": "OBSERVED",
           "protocol_version": P["version"],
           "protocol_frozen": str(P["frozen_on"]),
           "seen_before_freezing": P["seen_before_freezing"]["L"],
           "conversion_rice_from_paddy": RICE_FROM_PADDY,
           "acre_in_ha": ACRE_HA,
           "l1_level": level_check(P, feats, rice),
           "l2_event": event_check(P, feats, oof, relief)}
    # flat keys for the claims registry, whose paths cannot hold the "/"
    # in the official season names
    cells = res["l1_level"]["cells"]
    ev = res["l2_event"]
    res["headline"] = {
        "kuruvai_ratio_to_official": cells["Kar/Kuruvai/Sornavari 2019-20"][
            "by_conversion"]["2/3 (official)"]["ratio_to_core_weighted"],
        "samba_ratio_to_official": cells["Samba/Thaladi/Pishanam 2019-20"][
            "by_conversion"]["2/3 (official)"]["ratio_to_core_weighted"],
        "samba_recalled_change": ev["samba_recalled"]["mean_change"],
        "samba_model_change": ev["samba_model"]["mean_change"],
        "samba_minus_kuruvai": ev["contrast_samba_minus_kuruvai"]["delta"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf8")
    print(f"wrote {OUT.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
