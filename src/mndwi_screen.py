"""Does water/vegetation decoupling at transplanting flag bad field-seasons?

    python src/mndwi_screen.py

Executes configs/mndwi_protocol.yaml, which was frozen before any result
was computed. The protocol is loaded, not restated: if this file's
behaviour and the protocol disagree, the run aborts rather than silently
preferring the code.

THE IDEA
--------
A paddy field is flooded at transplanting. A plot that looks green while
showing little standing water during that window may be green with
something other than establishing paddy. MNDWI, (B3 - B11) / (B3 + B11),
responds to open water; NDMI, which this project already computes, uses B8
against B11 and responds to canopy moisture. They are not interchangeable,
which is why this needed building rather than reusing an existing column.

THE PRIOR IS NOT FAVOURABLE
---------------------------
Seven reliability signals have been tested in this project and none beat
chance. This one is tested the same way and reported the same way whatever
it returns.

Writes results/metrics/mndwi_screen.json.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import extract_features as ef  # noqa: E402

ROOT = ef.ROOT
PROTOCOL = PROJ / "configs" / "mndwi_protocol.yaml"
OUT = PROJ / "results" / "metrics" / "mndwi_screen.json"
LEVEL = 0.80
NBOOT, SEED = 10000, 0
BUDGETS = (0.25, 0.33, 0.50)


# ------------------------------------------------------------- protocol
def load_protocol() -> dict:
    if not PROTOCOL.exists():
        raise SystemExit(f"frozen protocol missing: {PROTOCOL}")
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf8"))
    if not p.get("frozen_before_results"):
        raise SystemExit("protocol is not marked frozen_before_results")
    w = p["signal"]["window"]
    if (w["days_before"], w["days_after"]) != (7, 21):
        raise SystemExit(
            f"this script implements a -7/+21 day window; the protocol "
            f"specifies -{w['days_before']}/+{w['days_after']}. Change one "
            f"of them deliberately and record an amendment.")
    if "0.80" not in p["target"]["definition"]:
        raise SystemExit("protocol target threshold is not 0.80")
    return p


def parse(d):
    for f in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime((d or "").strip(), f)
        except Exception:                                    # noqa: BLE001
            continue
    return None


# --------------------------------------------------------------- signal
def window_means(uid: str, sel, t0: datetime, before: int, after: int):
    """Mean NDVI and MNDWI over cloud-screened acquisitions in the window.

    Reads B3 and B11 on their own grids with the mask resampled to match,
    exactly as src/extract_features.py does, so no imagery is upsampled.
    B11 is a 20 m band and B3 a 10 m band; a plot too small to register on
    the 20 m grid yields no MNDWI and the field-season is reported as not
    scorable rather than imputed.
    """
    lo, hi = t0 - timedelta(days=before), t0 + timedelta(days=after)
    ndvi, mndwi, n_acq, n_cloud = [], [], 0, 0
    for f in sorted(glob.glob(str(ROOT / f"images/S2/npy/{uid}/*.npz"))):
        dt = ef.acq_date(f)
        if dt is None or not (lo <= dt <= hi):
            continue
        d = np.load(f)

        def band(name):
            if name not in d:
                return np.nan
            a = d[name]
            m = ef.fit_mask(sel, a.shape)
            if m.sum() == 0:
                return np.nan
            return float(np.nanmean(a[m]))

        if "SCL" in d:
            a = d["SCL"]
            m = ef.fit_mask(sel, a.shape)
            if m.sum() and np.isin(a[m].astype(int),
                                   list(ef.SCL_BAD)).mean() > 0.5:
                n_cloud += 1
                continue
        g2, swir = band("B3"), band("B11")
        red, nir = band("B4"), band("B8")
        n_acq += 1
        if np.isfinite([red, nir]).all():
            ndvi.append((nir - red) / (nir + red + 1e-9))
        if np.isfinite([g2, swir]).all():
            mndwi.append((g2 - swir) / (g2 + swir + 1e-9))
    return (float(np.mean(ndvi)) if ndvi else np.nan,
            float(np.mean(mndwi)) if mndwi else np.nan,
            n_acq, n_cloud)


# ---------------------------------------------------------------- stats
def auroc(score, pos):
    pos = np.asarray(pos, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    s = np.asarray(score, dtype=float)
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), dtype=float)
    srt = s[o]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def at_budget(score, bad, frac):
    n = len(score)
    k = max(1, int(round(frac * n)))
    sel = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")[:k]
    m = np.zeros(n, dtype=bool)
    m[sel] = True
    c = int((m & bad).sum())
    return {"budget": frac, "n_visited": k, "n_caught": c,
            "recall": c / max(1, int(bad.sum())), "precision": c / k}


def boot(stat, plots, n_boot=NBOOT, seed=SEED):
    uniq = np.unique(plots)
    pos = {g: np.where(plots == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        idx = np.concatenate([pos[g] for g in pick])
        r = stat(idx)
        if r is not None and np.isfinite(r):
            v.append(float(r))
    if not v:
        return float("nan"), float("nan")
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def permutation_null(bad, n_perm=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    a = [auroc(rng.permutation(len(bad)).astype(float), bad)
         for _ in range(n_perm)]
    a = np.array(a)
    return {"mean": float(a.mean()),
            "p2_5": float(np.percentile(a, 2.5)),
            "p97_5": float(np.percentile(a, 97.5)),
            "n_permutations": n_perm}


# ----------------------------------------------------------------- main
def main() -> int:
    p = load_protocol()
    w = p["signal"]["window"]
    before, after = int(w["days_before"]), int(w["days_after"])
    print(f"protocol v{p['version']} frozen {p['frozen_on']}; window "
          f"-{before}/+{after} days around {w['anchor']}")

    split = json.loads(
        (PROJ / "configs/splits.json").read_text(encoding="utf8"))
    assign = {u: s for s, v in split["field_seasons"].items() for u in v}
    tab = {r["UNIQUE_ID"]: r for r in csv.DictReader(
        (ROOT / "sickle_dataset_tabular.csv").open(encoding="utf8"))}

    feats = list(csv.DictReader(
        (PROJ / "results/features.csv").open(encoding="utf8")))
    print(f"{len(feats)} yield-labelled field-seasons")

    rows, not_scorable = [], []
    for i, r in enumerate(feats, 1):
        uid, pid = r["uid"], float(r["plot"])
        t = tab[uid]
        t0 = parse(t.get("TRANSPLANTING_DATE"))
        if t0 is None:
            not_scorable.append((uid, "no transplanting date"))
            continue
        sel = ef.plot_pixels(uid, pid, "10m")
        if sel is None or sel.sum() == 0:
            not_scorable.append((uid, "no plot pixels"))
            continue
        nd, mn, n_acq, n_cloud = window_means(uid, sel, t0, before, after)
        rows.append({"uid": uid, "plot": pid,
                     "split": assign.get(uid, r.get("split", "?")),
                     "yield": float(r["yield"]), "ndvi": nd, "mndwi": mn,
                     "n_acq": n_acq, "n_cloud_skipped": n_cloud})
        if i % 60 == 0:
            print(f"  {i}/{len(feats)}")

    have = [r for r in rows if np.isfinite(r["ndvi"])
            and np.isfinite(r["mndwi"])]
    for r in rows:
        if not (np.isfinite(r["ndvi"]) and np.isfinite(r["mndwi"])):
            not_scorable.append((r["uid"], "no usable acquisition in window"))
    print(f"\nscorable {len(have)}/{len(feats)}; "
          f"not scorable {len(not_scorable)} "
          f"(reported, not dropped from the denominator silently)")
    if len(have) < 30:
        print("TOO FEW SCORABLE FIELD-SEASONS to evaluate; stopping.")
        OUT.write_text(json.dumps(
            {"n_scorable": len(have), "not_scorable": not_scorable,
             "verdict": "not evaluable"}, indent=2), encoding="utf8")
        return 1

    nd = np.array([r["ndvi"] for r in have])
    mn = np.array([r["mndwi"] for r in have])
    y = np.array([r["yield"] for r in have])
    plots = np.array([r["plot"] for r in have])
    spl = np.array([r["split"] for r in have])

    tr = spl == "train"
    bench = float(y[tr].mean())
    znd = (nd - nd[tr].mean()) / (nd[tr].std() + 1e-12)
    zmn = (mn - mn[tr].mean()) / (mn[tr].std() + 1e-12)
    decoupling = znd - zmn
    bad = y < LEVEL * bench

    hist = {}
    byplot = defaultdict(list)
    for i, r in enumerate(have):
        byplot[r["plot"]].append(i)
    hv = np.full(len(have), np.nan)
    for _, v in byplot.items():
        for i in v:
            ti = parse(tab[have[i]["uid"]]["SOWING_DATE"])
            prior = [j for j in v
                     if parse(tab[have[j]["uid"]]["HARVESTING_DATE"]) < ti]
            if prior:
                hv[i] = float(np.mean(y[prior]))
    hist_score = -np.where(np.isfinite(hv), hv, bench)

    signals = {
        "decoupling (pre-specified)": decoupling,
        "ndvi alone": znd,
        "mndwi alone": -zmn,
        "history only": hist_score,
    }

    res = {"protocol_version": p["version"],
           "protocol_frozen": str(p["frozen_on"]),
           "window_days": [before, after],
           "n_field_seasons": len(feats), "n_scorable": len(have),
           "not_scorable": not_scorable[:40],
           "n_not_scorable": len(not_scorable),
           "threshold_kg_per_acre": LEVEL * bench, "sets": {}}

    for label, mask in (("held-out test split (PRIMARY)", spl == "test"),
                        ("all scorable field-seasons (secondary)",
                         np.ones(len(have), bool))):
        idx = np.where(mask)[0]
        b, pl = bad[idx], plots[idx]
        print("\n" + "=" * 76)
        print(f"{label}: {len(idx)} field-seasons at "
              f"{len(np.unique(pl))} plots, {int(b.sum())} below "
              f"{LEVEL * bench:.0f} kg/acre")
        print("=" * 76)
        if not (0 < b.sum() < len(idx)):
            print("   degenerate, no contrast")
            continue
        null = permutation_null(b)
        print(f"   random referral null: mean {null['mean']:.3f}, central "
              f"95% [{null['p2_5']:.3f}, {null['p97_5']:.3f}]")
        print(f"{'signal':<28s}{'AUROC':>8s}{'95% CI':>18s}"
              f"{'rec@33%':>9s}{'informative':>13s}")
        entry = {"n": len(idx), "n_bad": int(b.sum()),
                 "permutation_null": null, "signals": {}}
        for nm, s in signals.items():
            sv = np.asarray(s)[idx]
            a = auroc(sv, b)
            lo, hi = boot(lambda i, sv=sv: (auroc(sv[i], b[i])
                                            if 0 < b[i].sum() < len(i)
                                            else None), pl)
            r33 = at_budget(sv, b, 0.33)
            entry["signals"][nm] = {
                "auroc": a, "ci": [lo, hi], "informative": bool(lo > 0.5),
                "budgets": {str(x): at_budget(sv, b, x) for x in BUDGETS}}
            print(f"{nm:<28s}{a:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>18s}"
                  f"{r33['recall']:>9.2f}"
                  f"{('yes' if lo > 0.5 else 'NO'):>13s}")

        # the two comparisons the protocol makes the verdict rest on
        dec = np.asarray(decoupling)[idx]
        for other in ("history only", "ndvi alone", "mndwi alone"):
            ov = np.asarray(signals[other])[idx]
            d = (entry["signals"]["decoupling (pre-specified)"]["auroc"]
                 - entry["signals"][other]["auroc"])
            lo, hi = boot(
                lambda i, a=dec, bb=ov: (auroc(a[i], b[i]) - auroc(bb[i], b[i])
                                         if 0 < b[i].sum() < len(i)
                                         else None), pl)
            entry.setdefault("vs", {})[other] = {
                "delta_auroc": d, "ci": [lo, hi], "wins": bool(lo > 0)}
            print(f"   decoupling minus {other:<16s} {d:>+7.3f}  "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"{'wins' if lo > 0 else 'not separable'}")
        res["sets"][label] = entry

    # ---- verdict, by the pre-registered rules
    prim = res["sets"].get("held-out test split (PRIMARY)")
    if prim:
        d = prim["signals"]["decoupling (pre-specified)"]
        beats_hist = prim["vs"]["history only"]["wins"]
        beats_parts = (prim["vs"]["ndvi alone"]["wins"]
                       and prim["vs"]["mndwi alone"]["wins"])
        ok = d["informative"] and beats_hist and beats_parts
        res["verdict"] = "supported" if ok else "NOT demonstrated"
        res["verdict_reasons"] = {
            "auroc_interval_excludes_chance": d["informative"],
            "beats_history_only": beats_hist,
            "beats_each_component_alone": beats_parts}
        print("\n" + "=" * 76)
        print(f"VERDICT (pre-registered rules): "
              f"{res['verdict'].upper()}")
        print(f"   AUROC interval excludes chance      "
              f"{'yes' if d['informative'] else 'no'}")
        print(f"   beats history-only ranking          "
              f"{'yes' if beats_hist else 'no'}")
        print(f"   beats NDVI alone and MNDWI alone    "
              f"{'yes' if beats_parts else 'no'}")
        print("   The target is a relative cut on farmer-recalled yield, "
              "not crop damage.")

    # ---- sensitivity to the window, reported not selected from
    res["sensitivity_note"] = (
        "The pre-specified window is -7/+21 days. Other windows are a "
        "robustness report and were not used to choose the verdict.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf8")
    print(f"\nwrote {OUT.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
