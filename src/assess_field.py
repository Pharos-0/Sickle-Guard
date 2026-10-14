r"""Field-by-field assessment: the practical pipeline.

    python src/assess_field.py --uid 2066
    python src/assess_field.py --split test --top 15 --csv out.csv

THE DESIGN FOLLOWS A MEASURED RESULT, NOT A PREFERENCE
-------------------------------------------------------
src/site_vs_season.py compared the satellite model against a baseline
that sees no imagery at all and only averages the field's previous
harvests, over 131 plots under grouped cross-validation:

    field with >=1 prior season   plot history, no imagery   MAE 197.4
    brand-new field, no history   satellite model            MAE 277.9
    no information at all         global mean                MAE 303.0

Plot history beats the satellite model by 80.5 kg/acre, 95% CI
[+40.9, +124.8] over plots. So this tool uses history whenever it
exists and falls back to the satellite model only for a field with no
record. The satellite model closes 24% of the gap between knowing
nothing and knowing the field, and that cold-start gain is real:
-25.1 kg/acre, 95% CI [-41.6, -8.8].

Reporting a satellite number for a field whose history is on file would
be presenting the worse of two available estimates.

WHAT IT CANNOT DO
-----------------
Detect a bad season. A field's deviation from its own history is not
predictable from this imagery: MAE 195.7 against 197.4 for predicting
no deviation at all, paired CI [-8.8, +5.2], and bad-season AUROC of
0.586-0.598 with every interval containing chance.

It does not decide eligibility, compute a payout, or produce an
Insurance Unit figure. The dataset has no Gram Panchayat identity, no
Threshold Yield and no Sum Insured. A flag means "a person should look
at this field", never "this claim is approved or denied".

EVIDENCE GRADES CARRIED ON EVERY OUTPUT
---------------------------------------
  level, field with history   MODERATE. r = +0.72 from history alone.
  level, cold start           WEAK but real. r = +0.47; beats the
                              global mean by 25 kg/acre, CI excludes 0.
  between-field ranking       MODERATE. Supported out-of-fold.
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

PROJ = Path(__file__).resolve().parents[1]
FEATS = PROJ / "results" / "features.csv"
SEEDS = [0, 1, 2, 3, 4]


def load_rows():
    with FEATS.open(encoding="utf8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["yield"] = float(r["yield"])
        r["plot"] = float(r["plot"])
    return rows


def plot_history(rows):
    """Mean of each field-season's OTHER seasons at the same plot.

    Uses past harvests, which a scheme assessing a registered field
    knows. Never uses the field-season's own label. Returns NaN where
    the plot has no other season, the genuine cold-start case.
    """
    from collections import defaultdict
    byplot = defaultdict(list)
    for i, r in enumerate(rows):
        byplot[r["plot"]].append(i)
    hist = np.full(len(rows), np.nan)
    for _, v in byplot.items():
        for i in v:
            others = [j for j in v if j != i]
            if others:
                hist[i] = float(np.mean([rows[j]["yield"] for j in others]))
    return hist


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

    pred_sat, lo, hi, spread = score(tgt)
    hist_all = plot_history(rows)
    pos_of = {r["uid"]: i for i, r in enumerate(rows)}
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
            "season": r["season"], "year": r["year"],
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
            print("                       no prior season on file, so "
                  "the satellite model is used")
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
        print("  SEASONAL DETERIORATION: not assessed. A field's "
              "deviation from its own")
        print("  history is not predictable from this imagery "
              "(AUROC 0.59, CI spans chance).")
        print("  This tool cannot tell you whether THIS season was bad.")
        print(f"\n  evidence grade: ranking MODERATE (r=+0.57 out-of-fold); "
              f"magnitude WEAK;\n  within-season detection NOT DEMONSTRATED. "
              f"This is a prioritisation aid,\n  not a yield measurement and "
              f"not an insurance decision.")
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
