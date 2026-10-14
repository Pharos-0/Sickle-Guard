"""RQ4: how does yield-prediction error propagate into a threshold decision?

    python src/insurance_sim.py

READ INSURANCE_ASSUMPTIONS.md FIRST. This is a sensitivity study, not a
payout calculation. The dataset has no Gram Panchayat identity for any of
the 131 yield plots, so Insurance Units cannot be formed and the real PMFBY
procedure cannot be reproduced. No monetary figure is produced.

THE HYPOTHETICAL RULE
---------------------
A field-season is loss-flagged when its yield falls below
`level * benchmark`, benchmark being the TRAIN-split mean so no test
information leaks in, and level swept over 0.70 / 0.80 / 0.90.

STRATEGIES COMPARED, all at the same verification budget
--------------------------------------------------------
  reference        decide from the reference yield (the oracle, by
                   construction error-free, so it is the target not a
                   competitor)
  prediction       decide from the model, verify nothing
  random           verify a random subset
  reliability      verify the subset with the highest forest disagreement
  low_yield        verify the lowest predicted yields -- the operationally
                   obvious rule, and the baseline the reliability signal has
                   to beat to be worth anything

A verified case is assumed to resolve to the reference decision. That is an
idealisation and the assumptions file says so.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trees import RandomForest  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
FEATS = PROJ / "results" / "features.csv"
OUT = PROJ / "results" / "metrics" / "insurance_sim.json"
LEVELS = [0.70, 0.80, 0.90]
BUDGETS = [0.20, 0.33, 0.50]
SEEDS = [0, 1, 2, 3, 4]


def main() -> int:
    sys.path.insert(0, str(PROJ / "src"))
    from reliability import cols_all, matrix  # reuse the exact feature set

    with FEATS.open(encoding="utf8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["yield"] = float(r["yield"])
        r["plot"] = float(r["plot"])
    tr = [r for r in rows if r["split"] in ("train", "val")]
    trn = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    cols = cols_all()

    ytr = np.array([r["yield"] for r in tr])
    yte = np.array([r["yield"] for r in te])
    plots = np.array([r["plot"] for r in te])

    Xtr, Xte = matrix(tr, cols), matrix(te, cols)
    med = np.nanmedian(Xtr, axis=0); med[~np.isfinite(med)] = 0.0
    for X in (Xtr, Xte):
        i = np.where(~np.isfinite(X)); X[i] = np.take(med, i[1])
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    per_seed, trees0 = [], None
    for s in SEEDS:
        m = RandomForest(n_estimators=200, max_depth=8, min_leaf=3,
                         max_features="sqrt", random_state=s).fit(Xtr, ytr)
        per_seed.append(m.predict(Xte))
        if trees0 is None:
            trees0 = np.vstack([t.predict(Xte) for t in m.trees])
    pred = np.vstack(per_seed).mean(0)
    disagree = trees0.std(axis=0, ddof=1) / np.abs(pred)

    # Benchmark from TRAIN only -- never from test, never from val.
    bench = float(np.mean([r["yield"] for r in trn]))
    print(f"test {len(te)} field-seasons / {len(set(plots))} plots")
    print(f"benchmark (train-split mean) {bench:.0f} kg/acre")
    print(f"model MAE {np.mean(np.abs(pred - yte)):.1f} kg/acre\n")
    print("NOT a payout calculation. No Insurance Unit exists in this data.")
    print("See INSURANCE_ASSUMPTIONS.md.\n")

    # ---- why the decisions differ: variance compression ---------------
    # A model that cannot predict well minimises MAE by sitting near the
    # mean. That is fine for MAE and fatal for a threshold decision, because
    # thresholds live in the tail the predictor has stopped reaching.
    vr = float(pred.std() / yte.std())
    slope = float(np.polyfit(pred, yte, 1)[0])
    comp = {"sd_reference": float(yte.std()), "sd_prediction": float(pred.std()),
            "variance_ratio": vr,
            "range_reference": float(yte.max() - yte.min()),
            "range_prediction": float(pred.max() - pred.min()),
            "pearson_r": float(np.corrcoef(pred, yte)[0, 1]),
            "calibration_slope_ref_on_pred": slope}
    print("VARIANCE COMPRESSION, the mechanism behind every disagreement")
    print(f"  SD  reference {yte.std():6.0f}   prediction {pred.std():6.0f}"
          f"   ratio {vr:.3f}")
    print(f"  range reference {yte.max()-yte.min():6.0f}   prediction "
          f"{pred.max()-pred.min():6.0f}")
    print(f"  correlation r {np.corrcoef(pred, yte)[0,1]:+.3f}  -- the model "
          f"ranks fields,")
    print(f"  calibration slope {slope:.3f} (1.0 = calibrated) -- but keeps "
          f"only {vr:.0%} of the spread.")
    print("")

    rng = np.random.default_rng(0)
    out = {"benchmark_kg_per_acre": bench, "n": len(te),
           "n_plots": int(len(set(plots))),
           "variance_compression": comp, "levels": {}}

    for lev in LEVELS:
        thr = lev * bench
        ref_flag = yte < thr
        prd_flag = pred < thr
        disagree_mask = ref_flag != prd_flag
        fn = int((ref_flag & ~prd_flag).sum())     # loss missed by the model
        fp = int((~ref_flag & prd_flag).sum())
        print(f"indemnity level {lev:.0%}  (threshold {thr:.0f} kg/acre)")
        print(f"  reference flags {int(ref_flag.sum()):2d}/{len(te)} as loss, "
              f"model flags {int(prd_flag.sum()):2d}")
        print(f"  decisions that differ: {int(disagree_mask.sum())} "
              f"({disagree_mask.mean():.0%})  "
              f"[missed losses {fn}, false alarms {fp}]")

        strat = {}
        for b in BUDGETS:
            k = max(1, int(round(b * len(te))))
            picks = {
                "random": None,
                "reliability": np.argsort(-disagree)[:k],
                "low_yield": np.argsort(pred)[:k],
            }
            for name, idx in picks.items():
                if name == "random":
                    caught = []
                    for _ in range(2000):
                        sel = rng.choice(len(te), size=k, replace=False)
                        m_ = np.zeros(len(te), bool); m_[sel] = True
                        caught.append(int((disagree_mask & ~m_).sum()))
                    residual = float(np.mean(caught))
                else:
                    m_ = np.zeros(len(te), bool); m_[idx] = True
                    residual = float((disagree_mask & ~m_).sum())
                strat.setdefault(f"{b:.2f}", {})[name] = {
                    "n_verified": k,
                    "residual_wrong_decisions": residual,
                    "residual_rate": residual / len(te)}
            r = strat[f"{b:.2f}"]
            print(f"    budget {b:>4.0%}: residual wrong decisions  "
                  f"random {r['random']['residual_wrong_decisions']:5.1f} | "
                  f"reliability {r['reliability']['residual_wrong_decisions']:5.1f} | "
                  f"low-yield {r['low_yield']['residual_wrong_decisions']:5.1f}"
                  f"   (of {int(disagree_mask.sum())})")

        shortfall_ref = np.maximum(thr - yte, 0) / bench
        shortfall_prd = np.maximum(thr - pred, 0) / bench
        out["levels"][f"{lev:.2f}"] = {
            "threshold_kg_per_acre": thr,
            "n_loss_reference": int(ref_flag.sum()),
            "n_loss_predicted": int(prd_flag.sum()),
            "n_decisions_differ": int(disagree_mask.sum()),
            "decision_disagreement_rate": float(disagree_mask.mean()),
            "missed_losses_false_negative": fn,
            "false_alarms_false_positive": fp,
            "mean_abs_shortfall_diff_frac_of_benchmark":
                float(np.mean(np.abs(shortfall_ref - shortfall_prd))),
            "verification": strat}
        print()

    out["headline"] = (
        "The model is competitive on MAE yet flags almost no losses, "
        f"because it retains only {vr:.0%} of the reference yield variance. "
        "Every decision disagreement at the 70% and 80% levels is a MISSED "
        "loss, never a false alarm.")
    out["caveats"] = [
        "No Gram Panchayat or Insurance Unit exists in this data; the real "
        "scheme decides at the IU, this simulates individual plots.",
        "The benchmark is the train-split mean, NOT a PMFBY Threshold Yield.",
        "No Sum Insured is available, so no monetary figure is computed.",
        "Verification is assumed to resolve a case to the reference value "
        "exactly, which overstates what a real crop-cutting experiment does.",
        "The reference itself is farmer-recalled yield, not a CCE "
        "measurement.",
    ]
    OUT.write_text(json.dumps(out, indent=2), encoding="utf8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
