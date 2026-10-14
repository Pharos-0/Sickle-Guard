"""Paired plot-level comparison of ablation variants on the test set.

    python src/paired_tests.py

WHY PAIRED, AND WHY THE MARGINAL INTERVALS ARE MISLEADING
----------------------------------------------------------
run_ablation.py reports a bootstrap interval on each variant's MAE
separately. Those intervals overlap heavily -- the train-mean floor spans
[208, 373] and the selected forest [188, 331] -- and a reader could conclude
the difference is not established. That reasoning is wrong in both
directions: overlapping marginal intervals neither prove nor disprove a
difference, because the two models are scored on THE SAME plots and their
errors are strongly correlated.

The correct statistic is the paired difference. Each bootstrap replicate
resamples PLOTS and recomputes both models' MAE on the same resampled plots,
so the shared difficulty of a plot cancels. The interval on that difference
is what decides whether a variant beats the floor.

A permutation test is also run: within each plot, the two models' absolute
errors are swapped at random, which is the exact null of "these two models
are interchangeable on this data".
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
PRED = PROJ / "results" / "predictions" / "test_predictions.csv"
OUT = PROJ / "results" / "metrics" / "paired_tests.json"
N_BOOT = 5000
N_PERM = 10000


def load():
    with PRED.open(encoding="utf8") as fh:
        rows = list(csv.DictReader(fh))
    variants = [c for c in rows[0] if c.startswith("V")]
    y = np.array([float(r["yield"]) for r in rows])
    plots = np.array([float(r["plot"]) for r in rows])
    P = {v: np.array([float(r[v]) for r in rows]) for v in variants}
    return y, plots, P, variants


def paired_boot(err_a, err_b, plots, n_boot=N_BOOT, seed=0):
    """CI on MAE(a) - MAE(b), resampling plots and scoring both on each."""
    up = sorted(set(plots))
    by = {p: np.where(plots == p)[0] for p in up}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        s = [up[i] for i in rng.integers(0, len(up), len(up))]
        idx = np.concatenate([by[p] for p in s])
        d.append(err_a[idx].mean() - err_b[idx].mean())
    d = np.array(d)
    return {"delta": float(err_a.mean() - err_b.mean()),
            "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
            "p_boot_gt0": float((d <= 0).mean())}


def paired_perm(err_a, err_b, plots, n_perm=N_PERM, seed=0):
    """Swap the two models' errors within a plot, at random, many times."""
    up = sorted(set(plots))
    by = {p: np.where(plots == p)[0] for p in up}
    obs = err_a.mean() - err_b.mean()
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        a, b = err_a.copy(), err_b.copy()
        flip = rng.random(len(up)) < 0.5
        for k, p in enumerate(up):
            if flip[k]:
                i = by[p]
                a[i], b[i] = err_b[i], err_a[i]
        if abs(a.mean() - b.mean()) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def main() -> int:
    y, plots, P, variants = load()
    err = {v: np.abs(P[v] - y) for v in variants}
    n_plots = len(set(plots))
    print(f"test: {len(y)} field-seasons from {n_plots} plots\n")

    print("EACH VARIANT vs the V1 train-mean floor, paired by plot")
    print(f"{'':4s}{'MAE':>8s}{'delta vs V1':>13s}{'95% CI on delta':>22s}"
          f"{'perm p':>9s}{'verdict':>14s}")
    res = {}
    for v in variants:
        if v == "V1":
            print(f"{v:<4s}{err[v].mean():>8.1f}{'floor':>13s}"
                  f"{'':>22s}{'':>9s}{'':>14s}")
            continue
        b = paired_boot(err["V1"], err[v], plots)
        p = paired_perm(err["V1"], err[v], plots)
        better = b["lo"] > 0
        worse = b["hi"] < 0
        verdict = ("better" if better else "worse" if worse
                   else "inconclusive")
        res[v] = {**b, "perm_p": p, "verdict": verdict,
                  "mae": float(err[v].mean())}
        print(f"{v:<4s}{err[v].mean():>8.1f}{b['delta']:>+13.1f}"
              f"{f'[{b[chr(108)+chr(111)]:+.1f}, {b[chr(104)+chr(105)]:+.1f}]':>22s}"
              f"{p:>9.3f}{verdict:>14s}")
    print("\n  delta > 0 means the variant has LOWER error than the floor.")
    print("  'inconclusive' means the interval on the paired difference "
          "contains zero.")

    print("\nADJACENT ABLATION STEPS, paired")
    print(f"{'step':<14s}{'delta MAE':>11s}{'95% CI':>22s}{'perm p':>9s}"
          f"{'verdict':>14s}")
    steps = {}
    order = [v for v in variants]
    for a, b in zip(order, order[1:]):
        bb = paired_boot(err[a], err[b], plots)
        pp = paired_perm(err[a], err[b], plots)
        verdict = ("helps" if bb["lo"] > 0 else "hurts" if bb["hi"] < 0
                   else "inconclusive")
        steps[f"{a}->{b}"] = {**bb, "perm_p": pp, "verdict": verdict}
        print(f"{a+chr(45)+chr(62)+b:<14s}{bb['delta']:>+11.1f}"
              f"{f'[{bb[chr(108)+chr(111)]:+.1f}, {bb[chr(104)+chr(105)]:+.1f}]':>22s}"
              f"{pp:>9.3f}{verdict:>14s}")

    n_inc = sum(1 for v in res.values() if v["verdict"] == "inconclusive")
    n_bet = sum(1 for v in res.values() if v["verdict"] == "better")
    print(f"\nSUMMARY: of {len(res)} variants compared with the floor, "
          f"{n_bet} are better, {n_inc} inconclusive, "
          f"{len(res)-n_bet-n_inc} worse.")
    print("Multiple comparisons: with 7 variants tested against one floor, a "
          "Bonferroni-adjusted\nthreshold would be 0.05/7 = 0.0071. "
          "Permutation p-values above are unadjusted.")

    OUT.write_text(json.dumps(
        {"n_field_seasons": int(len(y)), "n_plots": int(n_plots),
         "vs_floor": res, "adjacent_steps": steps,
         "n_boot": N_BOOT, "n_perm": N_PERM,
         "note": ("Paired by plot. Marginal per-variant intervals in "
                  "ablation.json overlap heavily and must not be read as "
                  "evidence of no difference; these paired intervals are "
                  "the correct comparison."),
         "bonferroni_threshold": 0.05 / max(1, len(res))},
        indent=2), encoding="utf8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
