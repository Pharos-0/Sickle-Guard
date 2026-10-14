"""Does satellite imagery add anything over knowing the field's history?

    python src/site_vs_season.py

THE QUESTION
------------
DISCOVERY.md found that 83 % of this model's loss-detection skill survives
being shown the wrong season's imagery of the right field. If the model is
really a site-quality estimator, then a baseline that knows only the field's
PAST YIELDS and sees no imagery at all should match it.

That is a fair deployment baseline, not a strawman: a scheme assessing a
registered field usually does know what that field produced before. If
satellite data cannot beat it, the imagery is not earning its place.

THE TWO STAGES
--------------
Stage 1  expected yield for this field  =  mean of its OTHER seasons
         (leave-one-season-out within the plot; no imagery, no model)
Stage 2  deviation from that expectation, predicted from imagery
         target = y - stage1 ;  this is the season-specific question

Stage 2 is the capability the project actually needs and has never tested
directly. Objective C -- "is this field doing unusually badly THIS season"
-- is exactly a question about the deviation, not about the level.

WHAT IS AND IS NOT ALLOWED
--------------------------
Stage 1 uses the held-out season's PLOT, never its own label. It uses the
labels of other seasons of that plot, which at deployment are past harvests
and therefore legitimately known. This is stated as a requirement: the
baseline needs persistent field identity and at least one prior season, and
is not deployable for a field seen for the first time.

Everything is evaluated under grouped cross-validation by plot, so the
satellite model never trains on a plot it is scored on. The history baseline
is computed per field-season within its own plot and so needs no folds.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import BACKEND, make_rf           # noqa: E402
from reliability import cols_all, matrix      # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "results" / "metrics" / "site_vs_season.json"
K, REPEATS, SEEDS = 5, 4, [0, 1]


def auroc(s, pos):
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    rk = np.empty(len(s)); srt = s[o]; i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        rk[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((rk[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def plot_boot(fn, plots, n_boot=3000, seed=0):
    up = sorted(set(plots))
    by = {p: np.where(plots == p)[0] for p in up}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        s = [up[i] for i in rng.integers(0, len(up), len(up))]
        idx = np.concatenate([by[p] for p in s])
        r = fn(idx)
        if r is not None and np.isfinite(r):
            v.append(r)
    if not v:
        return {"lo": float("nan"), "hi": float("nan")}
    v = np.array(v)
    return {"lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5))}


def grouped_oof(X, y, plots, up, tag):
    oof = np.zeros((REPEATS, len(y)))
    for rep in range(REPEATS):
        rng = np.random.default_rng(1000 + rep)
        for f in np.array_split(rng.permutation(up), K):
            m = np.isin(plots, f)
            Xtr, Xte = X[~m].copy(), X[m].copy()
            med = np.nanmedian(Xtr, 0); med[~np.isfinite(med)] = 0.0
            for A in (Xtr, Xte):
                i = np.where(~np.isfinite(A)); A[i] = np.take(med, i[1])
            mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            P = np.vstack([make_rf(s).fit(Xtr, y[~m]).predict(Xte)
                           for s in SEEDS])
            oof[rep, m] = P.mean(0)
    print(f"  {tag} done")
    return oof.mean(0)


def main() -> int:
    rows = list(csv.DictReader((PROJ / "results/features.csv").open(encoding="utf8")))
    y = np.array([float(r["yield"]) for r in rows])
    plots = np.array([float(r["plot"]) for r in rows])
    up = np.array(sorted(set(plots)))
    cols = cols_all()
    X = matrix(rows, cols)
    idx = defaultdict(list)
    for i, p in enumerate(plots):
        idx[p].append(i)
    print(f"{len(y)} field-seasons, {len(up)} plots | backend {BACKEND}\n")

    # ---------- Stage 1: plot history, no imagery ----------------------
    hist = np.full(len(y), np.nan)
    for p, v in idx.items():
        for i in v:
            others = [j for j in v if j != i]
            if others:
                hist[i] = y[others].mean()
    have = np.isfinite(hist)
    print(f"plot-history baseline available for {have.sum()}/{len(y)} "
          f"field-seasons (needs >=1 prior season)")

    gmean = float(y.mean())
    print("\nSTAGE 1 -- predicting the LEVEL")
    print(f"{'predictor':<34s}{'MAE':>8s}{'r':>8s}{'uses imagery':>14s}")
    sat = grouped_oof(X, y, plots, up, "  satellite model")
    cands = {
        "global mean (no info)": (np.full(len(y), gmean), "no"),
        "plot history (no imagery)": (hist, "no"),
        "satellite model": (sat, "YES"),
    }
    lvl = {}
    for k, (p_, img) in cands.items():
        m = have if k != "global mean (no info)" else np.ones(len(y), bool)
        m = m & np.isfinite(p_)
        mae = float(np.mean(np.abs(p_[m] - y[m])))
        r = float(np.corrcoef(p_[m], y[m])[0, 1]) if np.std(p_[m]) > 0 else float("nan")
        lvl[k] = {"mae": mae, "r": r, "n": int(m.sum())}
        print(f"{k:<34s}{mae:>8.1f}{r:>8.3f}{img:>14s}")

    # paired comparison on the fields where both exist
    both = have
    d = np.abs(sat[both] - y[both]) - np.abs(hist[both] - y[both])
    ci = plot_boot(lambda i: float(np.mean(np.abs(sat[i] - y[i])
                                           - np.abs(hist[i] - y[i])))
                   if np.isfinite(hist[i]).all() else None, plots[both])
    print(f"\n  satellite minus history, paired MAE difference: "
          f"{d.mean():+.1f} kg/acre")
    print(f"  95% CI over plots [{ci['lo']:+.1f}, {ci['hi']:+.1f}]  "
          f"({'satellite better' if ci['hi'] < 0 else 'history better' if ci['lo'] > 0 else 'INCONCLUSIVE'})")

    # ---------- Stage 2: can imagery predict the DEVIATION? ------------
    print("\nSTAGE 2 -- predicting the DEVIATION from the field's own history")
    dev = y - hist
    sub = np.where(have)[0]
    Xs, ys, ps = X[sub], dev[sub], plots[sub]
    ups = np.array(sorted(set(ps)))
    print(f"  target = y - plot history, n={len(sub)}, "
          f"SD {ys.std():.0f} kg/acre, {len(ups)} plots")
    dev_pred = grouped_oof(Xs, ys, ps, ups, "  deviation model")
    mae_dev = float(np.mean(np.abs(dev_pred - ys)))
    mae_zero = float(np.mean(np.abs(ys)))          # predict "no deviation"
    r_dev = float(np.corrcoef(dev_pred, ys)[0, 1])
    cid = plot_boot(lambda i: float(np.mean(np.abs(dev_pred[i] - ys[i]))
                                    - np.mean(np.abs(ys[i]))), ps)
    print(f"  deviation model MAE {mae_dev:.1f} vs "
          f"predicting zero deviation {mae_zero:.1f}")
    print(f"  paired difference 95% CI over plots "
          f"[{cid['lo']:+.1f}, {cid['hi']:+.1f}]")
    print(f"  correlation with the true deviation: r = {r_dev:+.3f}")

    # bad-season detection: deviation below a prespecified tolerance
    print("\nBAD-SEASON DETECTION (a field doing badly vs ITS OWN history)")
    print(f"{'definition':<34s}{'n_bad':>7s}{'AUROC':>8s}{'95% CI':>18s}")
    bad_res = {}
    for name, thresh in (("deviation < -10% of history", -0.10),
                         ("deviation < -20% of history", -0.20),
                         ("deviation < -1 SD of deviation", None)):
        if thresh is None:
            badm = ys < -ys.std()
        else:
            badm = ys < thresh * hist[sub]
        if badm.sum() < 5 or badm.all():
            print(f"{name:<34s}{int(badm.sum()):>7d}   too few")
            continue
        a = auroc(-dev_pred, badm)
        c = plot_boot(lambda i, b=badm: (auroc(-dev_pred[i], b[i])
                      if 0 < b[i].sum() < len(i) else None), ps)
        bad_res[name] = {"n_bad": int(badm.sum()), "auroc": a, "ci": c,
                         "informative": bool(not (c["lo"] <= 0.5 <= c["hi"]))}
        flag = "*" if bad_res[name]["informative"] else ""
        print(f"{name:<34s}{int(badm.sum()):>7d}{a:>8.3f}"
              f"{f'[{c[chr(108)+chr(111)]:.3f}, {c[chr(104)+chr(105)]:.3f}]':>18s} {flag}")

    verdict_lvl = ("satellite adds nothing over plot history"
                   if ci["lo"] > -1e-9 else "satellite beats plot history")
    verdict_dev = ("deviation is predictable" if cid["hi"] < 0
                   else "deviation is NOT predictable from imagery")
    print(f"\nVERDICT")
    print(f"  level:     {verdict_lvl}")
    print(f"  deviation: {verdict_dev}")

    OUT.write_text(json.dumps(
        {"backend": BACKEND, "n_rows": len(y), "n_plots": int(len(up)),
         "stage1_level": lvl,
         "stage1_paired_sat_minus_hist_mae": float(d.mean()),
         "stage1_paired_ci": ci,
         "stage2": {"n": len(sub), "sd_deviation": float(ys.std()),
                    "mae_model": mae_dev, "mae_predict_zero": mae_zero,
                    "paired_ci": cid, "r": r_dev},
         "bad_season": bad_res,
         "verdict_level": verdict_lvl, "verdict_deviation": verdict_dev,
         "caveat": ("Plot history needs persistent field identity and at "
                    "least one prior season; it is not deployable for an "
                    "unseen field. Exploratory grouped CV, not the spent "
                    "confirmatory split.")},
        indent=2), encoding="utf8")
    np.save(PROJ / "results/predictions/hist_baseline.npy", hist)
    np.save(PROJ / "results/predictions/dev_pred.npy", dev_pred)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
