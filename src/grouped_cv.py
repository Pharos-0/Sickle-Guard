"""Better-powered estimate of the ranking signal, over all 131 plots.

    python src/grouped_cv.py

WHY THIS EXISTS
---------------
The confirmatory run failed both primary criteria, but on point estimates
that look useful: AUROC 0.738 and recall 55 % against 33 % random. The
failure is one of precision, not direction -- 27 test plots and 11 true
losses give a CI of [0.434, 0.947].

ACCEPTANCE_CRITERIA.md section E anticipated this and says the response is
to identify the evidence gap through a separately designed experiment, not
to weaken the criterion. This is that experiment.

Repeated grouped K-fold over ALL 131 plots produces an out-of-fold
prediction for every one of the 319 field-seasons, so the ranking is
estimated on 131 independent units instead of 27. Every plot is held out
exactly once per repeat, and normalisation, imputation and model fitting
happen inside each fold on the training side only.

WHAT THIS IS AND IS NOT
-----------------------
It is a better-powered estimate of the same quantity.
It is NOT a replacement for the held-out test, and it does not overturn the
confirmatory verdict. Out-of-fold estimates reuse every plot for both
fitting and scoring across folds, so they are optimistic relative to a
single untouched split. Both numbers are reported side by side.
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
OUT = PROJ / "results" / "metrics" / "grouped_cv.json"
K, REPEATS, SEEDS = 5, 4, [0, 1]
LEVEL, BUDGET = 0.80, 0.33


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


def main() -> int:
    rows = list(csv.DictReader((PROJ / "results/features.csv").open(encoding="utf8")))
    for r in rows:
        r["yield"] = float(r["yield"]); r["plot"] = float(r["plot"])
    cols = cols_all()
    y = np.array([r["yield"] for r in rows])
    plots = np.array([r["plot"] for r in rows])
    up = np.array(sorted(set(plots)))
    print(f"{len(rows)} field-seasons from {len(up)} plots | backend {BACKEND}")
    print(f"{REPEATS} repeats of grouped {K}-fold, {len(SEEDS)} seeds per fit\n")

    oof = np.zeros((REPEATS, len(rows)))
    for rep in range(REPEATS):
        rng = np.random.default_rng(1000 + rep)
        perm = rng.permutation(up)
        folds = np.array_split(perm, K)
        for f in folds:
            te_m = np.isin(plots, f)
            tr_rows = [r for r, m in zip(rows, ~te_m) if m]
            te_rows = [r for r, m in zip(rows, te_m) if m]
            Xtr, Xte = matrix(tr_rows, cols), matrix(te_rows, cols)
            med = np.nanmedian(Xtr, 0); med[~np.isfinite(med)] = 0.0
            for X in (Xtr, Xte):
                i = np.where(~np.isfinite(X)); X[i] = np.take(med, i[1])
            mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            ytr = np.array([r["yield"] for r in tr_rows])
            P = np.vstack([make_rf(s).fit(Xtr, ytr).predict(Xte) for s in SEEDS])
            oof[rep, te_m] = P.mean(0)
        print(f"  repeat {rep+1}/{REPEATS} done")

    pred = oof.mean(0)
    bench = float(y.mean())
    mae = float(np.mean(np.abs(pred - y)))
    floor = float(np.mean(np.abs(bench - y)))
    print(f"\nout-of-fold MAE {mae:.1f} (floor {floor:.1f}) | "
          f"SD ratio {pred.std()/y.std():.3f} | r {np.corrcoef(pred,y)[0,1]:+.3f}")

    pos = y < LEVEL * bench
    a = auroc(-pred, pos)
    by = {p: np.where(plots == p)[0] for p in up}
    rng = np.random.default_rng(0)
    av, rv = [], []
    kk = max(1, int(round(BUDGET * len(y))))
    for _ in range(4000):
        s = [up[i] for i in rng.integers(0, len(up), len(up))]
        idx = np.concatenate([by[p] for p in s])
        if 0 < pos[idx].sum() < len(idx):
            av.append(auroc(-pred[idx], pos[idx]))
            k2 = max(1, int(round(BUDGET * len(idx))))
            sel = np.argsort(pred[idx])[:k2]
            m = np.zeros(len(idx), bool); m[sel] = True
            rv.append((m & pos[idx]).sum() / pos[idx].sum())
    av, rv = np.array(av), np.array(rv)
    aci = (float(np.percentile(av, 2.5)), float(np.percentile(av, 97.5)))
    sel = np.argsort(pred)[:kk]
    m = np.zeros(len(y), bool); m[sel] = True
    rec = float((m & pos).sum() / pos.sum())
    rci = (float(np.percentile(rv, 2.5)), float(np.percentile(rv, 97.5)))

    print(f"\nLOSS RANKING at the {LEVEL:.0%} level, {int(pos.sum())} true "
          f"losses of {len(y)}")
    print(f"  AUROC {a:.3f}  95% CI [{aci[0]:.3f}, {aci[1]:.3f}]  "
          f"{'EXCLUDES' if aci[0] > 0.5 else 'includes'} 0.5")
    print(f"  recall @ {BUDGET:.0%} budget {rec:.0%}  95% CI "
          f"[{rci[0]:.0%}, {rci[1]:.0%}]  random {BUDGET:.0%}  "
          f"{'EXCLUDES' if rci[0] > BUDGET else 'includes'} random")

    # how much does precision improve with plot count?
    print("\nPRECISION vs number of independent plots (subsample the CV set)")
    print(f"{'plots':>7s}{'AUROC CI width':>17s}")
    widths = {}
    for n in (27, 50, 80, 110, len(up)):
        if n > len(up):
            continue
        ws = []
        for t in range(40):
            rr = np.random.default_rng(t)
            sub = rr.choice(up, size=n, replace=False)
            idx = np.concatenate([by[p] for p in sub])
            if not (0 < pos[idx].sum() < len(idx)):
                continue
            vv = []
            for _ in range(300):
                s2 = [sub[i] for i in rr.integers(0, n, n)]
                i2 = np.concatenate([by[p] for p in s2])
                if 0 < pos[i2].sum() < len(i2):
                    vv.append(auroc(-pred[i2], pos[i2]))
            if vv:
                ws.append(np.percentile(vv, 97.5) - np.percentile(vv, 2.5))
        widths[n] = float(np.mean(ws)) if ws else float("nan")
        print(f"{n:>7d}{widths[n]:>17.3f}")
    print("\n  The confirmatory test had 27 plots. This shows what the "
          "interval\n  width would be at larger plot counts, which is the "
          "evidence gap.")

    OUT.write_text(json.dumps(
        {"backend": BACKEND, "k": K, "repeats": REPEATS,
         "n_rows": len(rows), "n_plots": int(len(up)),
         "oof_mae": mae, "floor_mae": floor,
         "sd_ratio": float(pred.std() / y.std()),
         "r": float(np.corrcoef(pred, y)[0, 1]),
         "level": LEVEL, "budget": BUDGET, "n_true_losses": int(pos.sum()),
         "auroc": a, "auroc_ci": {"lo": aci[0], "hi": aci[1]},
         "auroc_excludes_half": bool(aci[0] > 0.5),
         "recall": rec, "recall_ci": {"lo": rci[0], "hi": rci[1]},
         "recall_excludes_random": bool(rci[0] > BUDGET),
         "ci_width_by_plot_count": widths,
         "status": ("Exploratory follow-up to an underpowered confirmatory "
                    "run. Out-of-fold estimates reuse plots across folds and "
                    "are optimistic relative to the untouched test split. "
                    "This does NOT overturn the confirmatory verdict.")},
        indent=2), encoding="utf8")
    np.save(PROJ / "results/predictions/oof_pred.npy", pred)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
