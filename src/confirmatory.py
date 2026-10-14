"""Confirmatory evaluation against the frozen acceptance criteria.

    python src/confirmatory.py

Scores the upgraded system ONCE on the 27 held-out test plots and reports
pass/fail for each criterion in ACCEPTANCE_CRITERIA.md. Criteria were frozen
from validation evidence before this ran.

THE UPGRADED SYSTEM
-------------------
1. Random forest over the full feature set, five seeds averaged. Unchanged
   from the previous study, because ROOT_CAUSE_REPORT.md showed the problem
   is not the point predictor.
2. Variance calibration toward the MSE-optimal SD ratio, with the inflation
   factor fitted on VALIDATION ONLY. Applied as hygiene; it cannot change
   the ranking and is not claimed to fix the decision problem.
3. The decision layer is budgeted verification of the lowest-predicted
   fields, NOT a threshold on the point prediction. This is the change the
   root-cause analysis argued for.

Leakage: the test split is read only to score. The calibration factor, the
benchmark and every threshold come from train or validation. Asserted below.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import BACKEND, make_rf           # noqa: E402
from reliability import cols_all, matrix      # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "results" / "metrics" / "confirmatory.json"
SEEDS = [0, 1, 2, 3, 4]
PRIMARY_LEVEL = 0.80          # frozen in ACCEPTANCE_CRITERIA.md C1
PRIMARY_BUDGET = 0.33         # frozen in ACCEPTANCE_CRITERIA.md C1


def auroc(s, pos):
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    rk = np.empty(len(s))
    srt = s[o]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        rk[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((rk[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_plots(fn, plots, n_boot=4000, seed=0):
    """Generic plot-level bootstrap; fn takes an index array."""
    up = sorted(set(plots))
    by = {p: np.where(plots == p)[0] for p in up}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        s = [up[i] for i in rng.integers(0, len(up), len(up))]
        idx = np.concatenate([by[p] for p in s])
        v = fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n": 0}
    v = np.array(vals)
    return {"lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)), "n": len(v)}


def fit_predict(tr_rows, target_rows, cols):
    Xtr, Xte = matrix(tr_rows, cols), matrix(target_rows, cols)
    med = np.nanmedian(Xtr, 0)
    med[~np.isfinite(med)] = 0.0
    for X in (Xtr, Xte):
        i = np.where(~np.isfinite(X))
        X[i] = np.take(med, i[1])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr = np.array([r["yield"] for r in tr_rows])
    P = np.vstack([make_rf(s).fit(Xtr, ytr).predict(Xte) for s in SEEDS])
    return P.mean(0)


def main() -> int:
    rows = list(csv.DictReader((PROJ / "results/features.csv").open(encoding="utf8")))
    for r in rows:
        r["yield"] = float(r["yield"])
        r["plot"] = float(r["plot"])
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    te = [r for r in rows if r["split"] == "test"]
    cols = cols_all()

    # Leakage guard, asserted rather than asserted-in-prose.
    tp = {r["plot"] for r in tr} | {r["plot"] for r in va}
    assert not (tp & {r["plot"] for r in te}), "a test plot is in train/val"

    bench = float(np.mean([r["yield"] for r in tr]))     # TRAIN only
    yva = np.array([r["yield"] for r in va])
    yte = np.array([r["yield"] for r in te])
    plots = np.array([r["plot"] for r in te])

    # ---- calibration factor from VALIDATION only ----------------------
    pv = fit_predict(tr, va, cols)
    r_val = float(np.corrcoef(pv, yva)[0, 1])
    ratio_val = float(pv.std() / yva.std())
    k = r_val / ratio_val if ratio_val > 0 else 1.0
    print(f"backend {BACKEND} | benchmark {bench:.0f} kg/acre (train only)")
    print(f"validation: r {r_val:+.3f}, SD ratio {ratio_val:.3f} "
          f"-> inflation factor {k:.3f} (fitted on validation)\n")

    # ---- score the test set, once --------------------------------------
    raw = fit_predict(tr + va, te, cols)
    pred = raw.mean() + (raw - raw.mean()) * k

    mae = float(np.mean(np.abs(pred - yte)))
    floor = float(np.mean(np.abs(bench - yte)))
    bias = float(np.mean(pred - yte))
    sd_ratio = float(pred.std() / yte.std())
    r_te = float(np.corrcoef(pred, yte)[0, 1])

    print("TEST, scored once")
    print(f"  MAE {mae:.1f}  (train-mean floor {floor:.1f})")
    print(f"  bias {bias:+.1f} | SD ratio {sd_ratio:.3f} | r {r_te:+.3f}\n")

    res = {"backend": BACKEND, "benchmark": bench,
           "inflation_factor_from_validation": k,
           "n_test_rows": len(te), "n_test_plots": int(len(set(plots))),
           "mae": mae, "floor_mae": floor, "bias": bias,
           "sd_ratio": sd_ratio, "r": r_te, "criteria": {}}

    def record(cid, desc, passed, detail):
        res["criteria"][cid] = {"description": desc, "pass": bool(passed),
                                "detail": detail}
        print(f"  [{'PASS' if passed else 'FAIL'}] {cid}  {desc}")
        print(f"         {detail}")

    print("ACCEPTANCE CRITERIA")
    record("A1", "test MAE <= train-mean floor", mae <= floor,
           f"MAE {mae:.1f} vs floor {floor:.1f}")
    record("A2", "|bias| <= 184 kg/acre", abs(bias) <= 184.0,
           f"bias {bias:+.1f}")
    record("A3", "SD ratio reported, must not exceed 1.0", sd_ratio <= 1.0,
           f"SD ratio {sd_ratio:.3f}, MSE-optimal would be {r_te:.3f} "
           f"(report only)")

    # ---- B1: ranking quality at the frozen 80% level -------------------
    pos = yte < PRIMARY_LEVEL * bench
    a = auroc(-pred, pos)
    ci = boot_plots(lambda i: (auroc(-pred[i], pos[i])
                               if 0 < pos[i].sum() < len(i) else None), plots)
    record("B1", f"AUROC CI excludes 0.5 at the {PRIMARY_LEVEL:.0%} level",
           ci["lo"] > 0.5,
           f"AUROC {a:.3f}, 95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}], "
           f"{int(pos.sum())} true losses of {len(yte)}")

    b2 = {}
    for lev in (0.70, 0.90):
        p2 = yte < lev * bench
        if 0 < p2.sum() < len(yte):
            aa = auroc(-pred, p2)
            cc = boot_plots(lambda i, p2=p2: (auroc(-pred[i], p2[i])
                            if 0 < p2[i].sum() < len(i) else None), plots)
            b2[f"{lev:.2f}"] = {"auroc": aa, "ci": cc,
                                "n_true": int(p2.sum())}
            print(f"  [info] B2  level {lev:.0%}: AUROC {aa:.3f} "
                  f"[{cc['lo']:.3f}, {cc['hi']:.3f}], {int(p2.sum())} true")
    res["B2_sensitivity"] = b2

    # ---- C1: recall at the frozen 33% budget ---------------------------
    def recall_at(idx, budget=PRIMARY_BUDGET):
        pi, yi = pred[idx], pos[idx]
        if yi.sum() == 0:
            return None
        kk = max(1, int(round(budget * len(idx))))
        sel = np.argsort(pi)[:kk]
        m = np.zeros(len(idx), bool)
        m[sel] = True
        return float((m & yi).sum() / yi.sum())

    kk = max(1, int(round(PRIMARY_BUDGET * len(te))))
    sel = np.argsort(pred)[:kk]
    mask = np.zeros(len(te), bool)
    mask[sel] = True
    rec = float((mask & pos).sum() / max(1, pos.sum()))
    rci = boot_plots(lambda i: recall_at(i), plots)
    record("C1", f"recall CI lower bound > {PRIMARY_BUDGET:.0%} "
                 f"(random) at a {PRIMARY_BUDGET:.0%} budget",
           rci["lo"] > PRIMARY_BUDGET,
           f"recall {rec:.0%} ({int((mask & pos).sum())}/{int(pos.sum())}), "
           f"95% CI [{rci['lo']:.0%}, {rci['hi']:.0%}], random "
           f"{PRIMARY_BUDGET:.0%}, {kk} of {len(te)} referred")

    print("\n  [info] C3  budgets x levels, missed losses and false alarms")
    c3 = {}
    for lev in (0.70, 0.80, 0.90):
        p2 = yte < lev * bench
        for b in (0.20, 0.33, 0.50):
            kb = max(1, int(round(b * len(te))))
            m = np.zeros(len(te), bool)
            m[np.argsort(pred)[:kb]] = True
            caught = int((m & p2).sum())
            c3[f"{lev:.2f}_{b:.2f}"] = {
                "n_true": int(p2.sum()), "referred": kb, "caught": caught,
                "missed": int(p2.sum()) - caught,
                "recall": caught / max(1, int(p2.sum())),
                "random_expected": b,
                "false_referrals": int((m & ~p2).sum())}
            print(f"         level {lev:.0%} budget {b:.0%}: caught "
                  f"{caught}/{int(p2.sum())} ({caught/max(1,p2.sum()):.0%}), "
                  f"random {b:.0%}, {int((m & ~p2).sum())} referred in vain")
    res["C3_grid"] = c3

    primary = [res["criteria"][c]["pass"] for c in ("B1", "C1")]
    guards = [res["criteria"][c]["pass"] for c in ("A1", "A2", "A3")]
    res["verdict"] = {
        "primary_all_pass": all(primary),
        "guards_all_pass": all(guards),
        "summary": ("system accepted" if all(primary) and all(guards)
                    else "system NOT accepted")}
    print(f"\nVERDICT: {res['verdict']['summary']}")
    print(f"  primary (B1, C1): {'pass' if all(primary) else 'FAIL'}")
    print(f"  guards (A1-A3):   {'pass' if all(guards) else 'FAIL'}")
    print(f"CAVEAT: {len(set(plots))} independent test plots. One plot is "
          f"{100/len(set(plots)):.0f}% of the sample.")

    OUT.write_text(json.dumps(res, indent=2), encoding="utf8")
    np.save(PROJ / "results/predictions/test_pred_calibrated.npy", pred)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
