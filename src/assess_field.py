r"""Field-by-field assessment: the practical pipeline.

    python src/assess_field.py --uid 2066
    python src/assess_field.py --split test --top 15 --csv out.csv

THE DESIGN FOLLOWS A MEASURED RESULT, NOT A PREFERENCE
-------------------------------------------------------
src/causal_history.py compared, under grouped cross-validation over 131
plots, a field's own history -- the mean yield of its seasons HARVESTED
BEFORE this season was SOWN, never a later one -- with the satellite model:

    field with history (188 of 319)   own history, no imagery   MAE 201.6
                                      satellite model           MAE 290.7
    brand-new field (131 of 319)      satellite model           MAE 259.5
                                      area average              MAE 273.8

Where history exists it beats the satellite model by 89.1 kg/acre, 95% CI
[-129.5, -50.7]. For a brand-new field the satellite model is NOT shown to
beat the area average: -14.3 kg/acre, 95% CI [-30.4, +3.6]
(src/does_satellite_add.py). So this tool uses history whenever it exists,
falls back to the satellite model only for a field with no record, and says
that this estimate is not reliable.

Until 2026-10-16 this tool averaged a field's OTHER seasons, later ones
included: the leak src/causal_history.py was written to remove. It gave a
history-based estimate to fields that had no earlier harvest. It now shares
farmer_report's loader and history, and tests/test_assess_field_history.py
requires the two to agree on every field-season.

WHAT IT CANNOT DO
-----------------
Detect a bad season. On plots that are sometimes above and sometimes below
the threshold, ranking falls to AUROC 0.626, 95% CI [0.386, 0.887], which
includes chance (src/label_validity.py).

It does not decide eligibility, compute a payout, or produce an
Insurance Unit figure. The dataset has no Gram Panchayat identity, no
Threshold Yield and no Sum Insured. A flag means "a person should look
at this field", never "this claim is approved or denied".

EVIDENCE GRADES CARRIED ON EVERY OUTPUT
---------------------------------------
  level, field with history   MODERATE. Own-history MAE 201.6 kg/acre.
  level, cold start           WEAK. Not shown to beat the area average.
  between-field ranking       MODERATE. Out-of-fold AUROC 0.824
                              [0.745, 0.891]; printed from the artefact.
  within-field season change  NOT DEMONSTRATED. Do not use for this.
  prediction interval         CALIBRATED BUT UNINFORMATIVE. 90%
                              coverage, but near-constant width (7
                              distinct values over 62 fields) and
                              correlated with realised error at only
                              r = +0.10.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import BACKEND, make_rf, tree_predictions   # noqa: E402
from reliability import cols_all, matrix                # noqa: E402
import farmer_report as fr                              # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
FEATS = PROJ / "results" / "features.csv"
SEEDS = [0, 1, 2, 3, 4]


def load_rows():
    """The report's own loader: the feature table plus sowing and
    harvest dates, which the causal history needs."""
    return fr.load()


def plot_history(rows):
    """A field-season's history: the mean yield of its seasons HARVESTED
    BEFORE it was SOWN, exactly as the report computes it. NaN where there
    is no such season, the genuine cold start. Never a later season."""
    return fr.causal_history(rows)


def _year(y):
    """The data stores the year as 2019.0."""
    try:
        return f"{int(float(y))}"
    except (TypeError, ValueError):
        return y


def _metric(name):
    return json.loads((PROJ / "results" / "metrics" / name)
                      .read_text(encoding="utf8"))


def build(rows, cols, fit_splits=("train", "val")):
    """Fit on the permitted splits; return a scorer for any rows."""
    tr = [r for r in rows if r["split"] in fit_splits]
    Xtr = matrix(tr, cols)
    med = np.nanmedian(Xtr, 0)
    med[~np.isfinite(med)] = 0.0
    i = np.where(~np.isfinite(Xtr))
    Xtr[i] = np.take(med, i[1])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Z = (Xtr - mu) / sd
    ytr = np.array([r["yield"] for r in tr])

    forests = [make_rf(s).fit(Z, ytr) for s in SEEDS]
    qlo = qhi = None
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        qlo = GradientBoostingRegressor(loss="quantile", alpha=0.05,
                                        n_estimators=200, max_depth=2,
                                        learning_rate=0.05,
                                        random_state=0).fit(Z, ytr)
        qhi = GradientBoostingRegressor(loss="quantile", alpha=0.95,
                                        n_estimators=200, max_depth=2,
                                        learning_rate=0.05,
                                        random_state=0).fit(Z, ytr)
    except Exception:
        pass

    # Calibration and thresholds are fitted on the FIT SPLITS only.
    bench = float(np.mean([r["yield"] for r in rows if r["split"] == "train"]))

    def score(target_rows):
        X = matrix(target_rows, cols)
        j = np.where(~np.isfinite(X))
        X[j] = np.take(med, j[1])
        Z2 = (X - mu) / sd
        P = np.vstack([f.predict(Z2) for f in forests])
        pred = P.mean(0)
        trees = tree_predictions(forests[0], Z2)
        spread = trees.std(axis=0, ddof=1) / np.abs(pred)
        lo = qlo.predict(Z2) if qlo is not None else pred - 1.64 * trees.std(0)
        hi = qhi.predict(Z2) if qhi is not None else pred + 1.64 * trees.std(0)
        return pred, lo, hi, spread

    # reference distribution of the dispersion signal, from the fit splits
    _, _, _, sp_fit = score(tr)
    return score, bench, sp_fit


def grade(spread, sp_fit, n_clear, n_px):
    """Reliability flag with the reason, from label-free inputs only."""
    reasons = []
    q = float((sp_fit < spread).mean())        # percentile within fit split
    if q < 0.33:
        lvl = "lower"
    elif q < 0.67:
        lvl = "typical"
    else:
        lvl = "HIGHER"
        reasons.append(f"model disagreement in the top {100*(1-q):.0f}% "
                       f"of training fields")
    if n_clear < 15:
        reasons.append(f"only {n_clear} cloud-free Sentinel-2 acquisitions")
    if n_px < 20:
        reasons.append(f"field covers only {n_px} Sentinel-2 pixels")
    return lvl, reasons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--uid", help="assess one field-season by UNIQUE_ID")
    ap.add_argument("--split", default="test",
                    help="assess a whole split (default test)")
    ap.add_argument("--budget", type=float, default=0.33,
                    help="verification budget as a fraction")
    ap.add_argument("--top", type=int, default=10,
                    help="rows to print when listing")
    ap.add_argument("--csv", type=Path, help="also write the full table here")
    args = ap.parse_args()

    rows = load_rows()
    cols = cols_all()
    score, bench, sp_fit = build(rows, cols)

    if args.uid:
        tgt = [r for r in rows if r["uid"] == args.uid]
        if not tgt:
            raise SystemExit(f"no field-season with UNIQUE_ID {args.uid}")
    else:
        tgt = [r for r in rows if r["split"] == args.split]
        if not tgt:
            raise SystemExit(f"no rows in split {args.split!r}")

    # This tool's own forests give the interval and the disagreement only.
    # The satellite estimate itself is the report's (farmer_report.satellite),
    # so a field gets the same number from both tools.
    _, lo, hi, spread = score(tgt)
    pos_of = {r["uid"]: i for i, r in enumerate(rows)}
    sat_all = fr.satellite(rows)
    pred_sat = np.array([sat_all[pos_of[r["uid"]]] for r in tgt])
    hist_all = plot_history(rows)
    hist = np.array([hist_all[pos_of[r["uid"]]] for r in tgt])

    # History where it exists; satellite only for a cold-start field.
    pred = np.where(np.isfinite(hist), hist, pred_sat)
    source = np.where(np.isfinite(hist), "plot-history", "satellite")
    order = np.argsort(pred)
    k = max(1, int(round(args.budget * len(tgt))))
    refer = set(order[:k].tolist())

    out = []
    for i, r in enumerate(tgt):
        lvl, why = grade(spread[i], sp_fit, int(float(r["s2_n_clear"])),
                         int(float(r["n_px"])))
        rank = int(np.where(order == i)[0][0]) + 1
        out.append({
            "uid": r["uid"], "plot": f"{r['plot']:.0f}",
            "season": r["season"], "year": _year(r["year"]),
            "predicted_yield_kg_per_acre": round(float(pred[i]), 1),
            "estimate_source": str(source[i]),
            "satellite_estimate": round(float(pred_sat[i]), 1),
            "plot_history": (round(float(hist[i]), 1)
                             if np.isfinite(hist[i]) else ""),
            "interval_90_lo": round(float(lo[i]), 1),
            "interval_90_hi": round(float(hi[i]), 1),
            "pct_of_benchmark": round(100 * float(pred[i]) / bench, 1),
            "rank_lowest_first": rank, "of": len(tgt),
            "disagreement": round(float(spread[i]), 4),
            "reliability": lvl,
            "verify": i in refer,
            "reasons": "; ".join(why) if why else "no quality flags",
            "reference_yield_kg_per_acre": round(r["yield"], 1),
        })

    if args.uid:
        o = out[0]
        print(f"\nFIELD-SEASON {o['uid']}   plot {o['plot']}   "
              f"{o['season']} {o['year']}")
        print("-" * 62)
        print(f"  estimated yield      "
              f"{o['predicted_yield_kg_per_acre']:.0f} kg/acre   "
              f"[source: {o['estimate_source']}]")
        if o["estimate_source"] == "plot-history":
            print("                       this field's own past "
                  "harvests beat the satellite model;")
            print(f"                       the model would have said "
                  f"{o['satellite_estimate']:.0f} kg/acre")
        else:
            print("                       no earlier season on file, so "
                  "the satellite model is used;")
            print("                       it is NOT shown to beat the area "
                  "average: do not act on the number")
        half = (o['interval_90_hi'] - o['interval_90_lo']) / 2
        print(f"  90% interval         {o['interval_90_lo']:.0f} to "
              f"{o['interval_90_hi']:.0f}   (+/-{half:.0f})")
        print("                       near-constant across fields; NOT "
              "specific to this field")
        print(f"  vs benchmark         {o['pct_of_benchmark']:.0f}% of "
              f"{bench:.0f} kg/acre")
        print(f"  rank, lowest first   {o['rank_lowest_first']} of {o['of']}")
        print(f"  reliability          {o['reliability']} model disagreement")
        print(f"  reasons              {o['reasons']}")
        print(f"  VERIFY IN FIELD?     "
              f"{'YES' if o['verify'] else 'not under this budget'}")
        print("")
        ev = _metric("label_validity.json")["ranking_by_subset"][
            "EVENT plots only (the real case)"]
        rk = _metric("does_satellite_add.json")["ranking"][
            "hybrid = SICKLE-Guard"]
        print("  SEASONAL DETERIORATION: not assessed. On plots that are "
              "sometimes above")
        print(f"  and sometimes below the threshold, ranking AUROC is "
              f"{ev['auroc']:.3f} [{ev['ci'][0]:.3f}, {ev['ci'][1]:.3f}],")
        print("  which includes chance. This tool cannot tell you whether "
              "THIS season was bad.")
        print(f"\n  evidence grade: ranking MODERATE (AUROC {rk['auroc']:.3f} "
              f"out-of-fold); magnitude WEAK;\n  within-season detection NOT "
              f"DEMONSTRATED. This is a prioritisation aid,\n  not a yield "
              f"measurement and not an insurance decision.")
        print(f"  [held-out reference yield was "
              f"{o['reference_yield_kg_per_acre']:.0f} kg/acre]")
    else:
        print(f"\nFIELD-BY-FIELD ASSESSMENT  split={args.split}  "
              f"n={len(tgt)}  backend={BACKEND}")
        print(f"benchmark {bench:.0f} kg/acre (train mean) | "
              f"verification budget {args.budget:.0%} -> {k} fields\n")
        print(f"{'rank':>4s} {'uid':>6s} {'plot':>6s} {'est':>7s} {'src':>5s} "
              f"{'90% interval':>16s} {'%bench':>7s} {'reliab':>8s} "
              f"{'verify':>7s}  reason")
        for o in sorted(out, key=lambda z: z["rank_lowest_first"])[:args.top]:
            print(f"{o['rank_lowest_first']:>4d} {o['uid']:>6s} "
                  f"{o['plot']:>6s} {o['predicted_yield_kg_per_acre']:>7.0f} "
                  f"{('hist' if o['estimate_source'] == 'plot-history' else 'sat'):>5s} "
                  f"{f'{o[chr(105)+chr(110)+chr(116)+chr(101)+chr(114)+chr(118)+chr(97)+chr(108)+chr(95)+chr(57)+chr(48)+chr(95)+chr(108)+chr(111)]:.0f}-{o[chr(105)+chr(110)+chr(116)+chr(101)+chr(114)+chr(118)+chr(97)+chr(108)+chr(95)+chr(57)+chr(48)+chr(95)+chr(104)+chr(105)]:.0f}':>16s} "
                  f"{o['pct_of_benchmark']:>7.0f} {o['reliability']:>8s} "
                  f"{('YES' if o['verify'] else '-'):>7s}  {o['reasons'][:46]}")
        nver = sum(1 for o in out if o["verify"])
        print(f"\n{nver} of {len(out)} field-seasons recommended for "
              f"verification under a {args.budget:.0%} budget.")
        print("Ranking is the supported use. Magnitudes are weak and "
              "within-season\nchange is not demonstrated -- see DISCOVERY.md.")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
