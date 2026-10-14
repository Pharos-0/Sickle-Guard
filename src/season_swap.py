"""Does the yield model use season-specific information, or only site quality?

    python src/season_swap.py

THE QUESTION
------------
95 % of yield variance in SICKLE is between plots. A model evaluated on
unseen plots can therefore score well by learning which FIELDS are good,
without ever learning which SEASON at a field was bad. Those two abilities
have completely different operational value: an insurance or loss-assessment
workflow needs the second, because a chronically low-yielding field is not
an insurable event -- a bad year at a field is.

The decomposition in the research log shows between-plot r = +0.568 but
within-plot season-ordering accuracy of 53.7 % [47.4, 60.0], i.e. chance.
This script tests that directly rather than by inference.

THE TEST
--------
Each field-season's satellite features are replaced by those of a DIFFERENT,
randomly chosen season of the SAME plot. The label stays with the original
field-season. Everything the model could learn about the site is preserved;
everything season-specific is destroyed.

  If performance is unchanged  -> the model uses only site quality.
  If performance collapses     -> the model genuinely reads the season.

This is a counterfactual on the inputs, so it needs no new labels and no
new data. It is run inside the same grouped cross-validation as the real
model, so the comparison is paired on identical folds and plots.

CONTROL
-------
A second condition swaps features with a random season of a DIFFERENT plot.
That destroys site information too and should collapse performance; if it
does not, the features carry nothing and the whole comparison is void.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import BACKEND, make_rf           # noqa: E402
from reliability import cols_all, matrix      # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "results" / "metrics" / "season_swap.json"
K, REPEATS, SEEDS = 5, 4, [0, 1]
LEVEL = 0.80


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


def run_cv(X_all, y, plots, up, tag):
    """Grouped CV with a fixed feature matrix; returns out-of-fold preds."""
    oof = np.zeros((REPEATS, len(y)))
    for rep in range(REPEATS):
        rng = np.random.default_rng(1000 + rep)
        for f in np.array_split(rng.permutation(up), K):
            m = np.isin(plots, f)
            Xtr, Xte = X_all[~m].copy(), X_all[m].copy()
            med = np.nanmedian(Xtr, 0); med[~np.isfinite(med)] = 0.0
            for X in (Xtr, Xte):
                i = np.where(~np.isfinite(X)); X[i] = np.take(med, i[1])
            mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            P = np.vstack([make_rf(s).fit(Xtr, y[~m]).predict(Xte)
                           for s in SEEDS])
            oof[rep, m] = P.mean(0)
    print(f"  {tag} done")
    return oof.mean(0)


def within_plot_accuracy(pred, y, idx):
    ok = tot = 0
    for v in idx.values():
        for a, b in itertools.combinations(v, 2):
            if y[a] == y[b]:
                continue
            tot += 1
            ok += (pred[a] > pred[b]) == (y[a] > y[b])
    return ok / max(1, tot), tot


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
    print(f"{len(y)} field-seasons, {len(up)} plots | backend {BACKEND}")
    print(f"{REPEATS} repeats of grouped {K}-fold per condition\n")

    rng = np.random.default_rng(42)

    # --- condition 2: swap with another season of the SAME plot ---------
    X_same = X.copy()
    n_swapped = 0
    for p, v in idx.items():
        if len(v) < 2:
            continue
        for i in v:
            others = [j for j in v if j != i]
            X_same[i] = X[rng.choice(others)]
            n_swapped += 1

    # --- condition 3 (control): swap with a season of a DIFFERENT plot --
    X_diff = X.copy()
    allidx = np.arange(len(y))
    for i in range(len(y)):
        others = allidx[plots != plots[i]]
        X_diff[i] = X[rng.choice(others)]

    print("running three conditions on identical folds")
    pred_real = run_cv(X, y, plots, up, "1 real features")
    pred_same = run_cv(X_same, y, plots, up, "2 same-plot season swap")
    pred_diff = run_cv(X_diff, y, plots, up, "3 different-plot swap (control)")

    bench = float(y.mean())
    pos = y < LEVEL * bench
    res = {}
    print(f"\n{'condition':<34s}{'MAE':>8s}{'r':>8s}{'lossAUROC':>11s}"
          f"{'withinAcc':>11s}")
    for tag, p in (("1 real features", pred_real),
                   ("2 same-plot season swap", pred_same),
                   ("3 different-plot swap (ctrl)", pred_diff)):
        mae = float(np.mean(np.abs(p - y)))
        r = float(np.corrcoef(p, y)[0, 1])
        a = auroc(-p, pos)
        wa, wn = within_plot_accuracy(p, y, idx)
        res[tag] = {"mae": mae, "r": r, "loss_auroc": a,
                    "within_plot_accuracy": wa, "n_within_pairs": wn}
        print(f"{tag:<34s}{mae:>8.1f}{r:>8.3f}{a:>11.3f}{wa:>11.1%}")

    floor = float(np.mean(np.abs(bench - y)))
    print(f"{'train-mean floor':<34s}{floor:>8.1f}{'-':>8s}{0.5:>11.3f}"
          f"{'-':>11s}")

    d_mae = res["2 same-plot season swap"]["mae"] - res["1 real features"]["mae"]
    d_auc = (res["1 real features"]["loss_auroc"]
             - res["2 same-plot season swap"]["loss_auroc"])
    print(f"\nDestroying season-specific information costs:")
    print(f"   MAE        {d_mae:+.1f} kg/acre")
    print(f"   loss AUROC {d_auc:+.3f}")
    ctrl = (res["1 real features"]["loss_auroc"]
            - res["3 different-plot swap (ctrl)"]["loss_auroc"])
    print(f"Destroying site information too (control) costs AUROC {ctrl:+.3f}")

    verdict = ("the model reads the season" if d_auc > 0.5 * ctrl and d_auc > 0.03
               else "the model is a SITE-QUALITY estimator: season-specific "
                    "information contributes little or nothing")
    print(f"\nVERDICT: {verdict}")
    print("Interpretation: if condition 2 matches condition 1 while the")
    print("control collapses, the satellite time series is being used to")
    print("identify good FIELDS, not to observe what happened in a SEASON.")

    res["deltas"] = {"mae_cost_of_season_swap": d_mae,
                     "auroc_cost_of_season_swap": d_auc,
                     "auroc_cost_of_control": ctrl}
    res["verdict"] = verdict
    res["note"] = ("Out-of-fold, grouped by plot, identical folds across "
                   "conditions. Exploratory: not a confirmatory test on the "
                   "spent held-out split.")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
