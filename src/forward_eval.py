"""Forward-in-time evaluation: train on the earlier season, predict the later.

    python src/forward_eval.py

WHY THIS SPLIT, AND WHY IT IS LEGITIMATELY FRESH
-------------------------------------------------
The plot-grouped confirmatory test was scored once and is spent; tuning
against it again would be worthless. But the plot axis is not the only axis
in this dataset, and it is not the one a deployment actually faces.

Every one of the 131 plots appears in both a 2019-sown and a 2020-sown
season. Splitting on TIME therefore gives an evaluation that has never been
used for any decision in this project, and it matches the real situation:
a scheme fits on the seasons it has already observed and must then act on a
season that has not happened yet.

It is a harder test than the plot split in one respect and easier in
another, and both are stated rather than glossed:

  HARDER   the model must generalise across years, which brings weather,
           variety and management shifts the plot split never exercised.
  EASIER   the test plots were seen during training, in their earlier
           season. This is NOT a test of generalisation to new fields. It
           is a test of generalisation to a new season at known fields,
           which is the question the history-first strategy exists to
           answer.

Because the test plots are not new, no claim about unseen fields may be
drawn from this script. The plot-grouped results remain the evidence for
that, and they failed.

WHAT IS COMPARED, all predicting the 2020-sown season
-----------------------------------------------------
  global mean          mean of the 2019-sown training seasons
  causal history       each plot's own 2019-sown yield, no imagery
  satellite model      fitted on 2019-sown seasons only
  history-first hybrid history where available, else satellite

and, as the decision task, which fields to send for verification.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import BACKEND, make_rf           # noqa: E402
from reliability import cols_all, matrix      # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
# dataset root; SICKLE_DATA_ROOT as in sickle-project/.env.example
ROOT = Path(__import__("os").environ.get(
    "SICKLE_DATA_ROOT", "D:/FPD/sickledataset extracted"))
OUT = PROJ / "results" / "metrics" / "forward_eval.json"
SEEDS = [0, 1, 2, 3, 4]
BUDGET = 0.33
LEVEL = 0.80


def parse(d):
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(d.strip(), fmt)
        except Exception:
            continue
    return None


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


def boot(fn, plots, idx, n_boot=4000, seed=0):
    up = sorted(set(plots))
    by = {p: idx[np.where(plots == p)[0]] for p in up}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        s = [up[i] for i in rng.integers(0, len(up), len(up))]
        j = np.concatenate([by[p] for p in s])
        r = fn(j)
        if r is not None and np.isfinite(r):
            v.append(r)
    if not v:
        return (float("nan"), float("nan"))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> int:
    feats = list(csv.DictReader((PROJ / "results/features.csv").open(encoding="utf8")))
    tab = {r["UNIQUE_ID"]: r for r in
           csv.DictReader((ROOT / "sickle_dataset_tabular.csv").open(encoding="utf8"))}
    y = np.array([float(r["yield"]) for r in feats])
    plots = np.array([float(r["plot"]) for r in feats])
    sow = np.array([parse(tab[r["uid"]]["SOWING_DATE"]) for r in feats])
    harv = np.array([parse(tab[r["uid"]]["HARVESTING_DATE"]) for r in feats])

    cut = datetime(2020, 6, 1)          # declared before looking at results
    tr_m = np.array([d < cut for d in sow])
    te_m = ~tr_m
    print(f"forward split at sowing date {cut.date()}")
    print(f"  train (sown before): {tr_m.sum():3d} field-seasons, "
          f"{len(set(plots[tr_m])):3d} plots")
    print(f"  test  (sown after) : {te_m.sum():3d} field-seasons, "
          f"{len(set(plots[te_m])):3d} plots")
    print(f"  test plots also seen in training: "
          f"{len(set(plots[te_m]) & set(plots[tr_m]))} "
          f"(this is NOT a new-field test)\n")

    cols = cols_all()
    X = matrix(feats, cols)
    Xtr, Xte = X[tr_m].copy(), X[te_m].copy()
    med = np.nanmedian(Xtr, 0); med[~np.isfinite(med)] = 0.0
    for A in (Xtr, Xte):
        i = np.where(~np.isfinite(A)); A[i] = np.take(med, i[1])
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr, yte = y[tr_m], y[te_m]
    sat = np.vstack([make_rf(s).fit(Xtr, ytr).predict(Xte)
                     for s in SEEDS]).mean(0)

    # causal history: prior harvest before this sowing, from TRAIN only
    byplot = defaultdict(list)
    for i, p in enumerate(plots):
        byplot[p].append(i)
    te_idx = np.where(te_m)[0]
    hist = np.full(len(te_idx), np.nan)
    for k, i in enumerate(te_idx):
        prior = [j for j in byplot[plots[i]] if tr_m[j] and harv[j] < sow[i]]
        if prior:
            hist[k] = float(np.mean(y[prior]))
    have = np.isfinite(hist)
    gmean = float(ytr.mean())
    hybrid = np.where(have, hist, sat)
    print(f"causal history available for {have.sum()}/{len(te_idx)} "
          f"({have.mean():.0%}) test field-seasons\n")

    pte = plots[te_m]
    print(f"{'estimator':<26s}{'MAE':>8s}{'95% CI':>20s}{'r':>8s}")
    res = {}
    for nm, p in (("global mean", np.full(len(yte), gmean)),
                  ("causal history", hist),
                  ("satellite model", sat),
                  ("history-first hybrid", hybrid)):
        m = np.isfinite(p)
        sub = te_idx[m]
        yy, pp = y[sub], p[m]
        mae = float(np.mean(np.abs(pp - yy)))
        lo, hi = boot(lambda j, p_=p, m_=m: float(np.mean(
            np.abs(p_[np.searchsorted(te_idx, j)] - y[j]))), pte[m], sub)
        r = float(np.corrcoef(pp, yy)[0, 1]) if np.std(pp) > 0 else float("nan")
        res[nm] = {"mae": mae, "ci": [lo, hi], "r": r, "n": int(m.sum())}
        print(f"{nm:<26s}{mae:>8.1f}{f'[{lo:.0f}, {hi:.0f}]':>20s}{r:>8.3f}")

    # paired, on the subset where history exists
    s2 = te_idx[have]
    d = (np.abs(hist[have] - y[s2]) - np.abs(sat[have] - y[s2]))
    lo, hi = boot(lambda j: float(np.mean(
        np.abs(hist[np.searchsorted(te_idx, j)] - y[j])
        - np.abs(sat[np.searchsorted(te_idx, j)] - y[j]))), pte[have], s2)
    print(f"\npaired on the {int(have.sum())} with history: history minus "
          f"satellite {d.mean():+.1f} kg/acre, 95% CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"  -> {'history better' if hi < 0 else 'satellite better' if lo > 0 else 'INCONCLUSIVE'}")

    # ---- the decision ---------------------------------------------------
    bench = gmean
    loss = yte < LEVEL * bench
    k = max(1, int(round(BUDGET * len(yte))))
    print(f"\nVERIFICATION DECISION, {int(loss.sum())} loss seasons of "
          f"{len(yte)}, budget {BUDGET:.0%} = {k} fields")
    print(f"{'strategy':<26s}{'AUROC':>8s}{'95% CI':>18s}{'recall':>8s}"
          f"{'caught':>9s}")
    dec = {}
    for nm, sc in (("random", None), ("satellite model", -sat),
                   ("history-first hybrid", -hybrid)):
        if sc is None:
            dec[nm] = {"auroc": 0.5, "recall": BUDGET}
            print(f"{nm:<26s}{0.5:>8.3f}{'-':>18s}{BUDGET:>8.0%}"
                  f"{BUDGET*loss.sum():>9.1f}")
            continue
        a = auroc(sc, loss)
        lo2, hi2 = boot(lambda j, s_=sc: (
            auroc(s_[np.searchsorted(te_idx, j)], y[j] < LEVEL * bench)
            if 0 < (y[j] < LEVEL * bench).sum() < len(j) else None), pte, te_idx)
        sel = np.argsort(-sc)[:k]
        m2 = np.zeros(len(yte), bool); m2[sel] = True
        rec = float((m2 & loss).sum() / max(1, loss.sum()))
        dec[nm] = {"auroc": a, "ci": [lo2, hi2], "recall": rec,
                   "caught": int((m2 & loss).sum()), "n_loss": int(loss.sum())}
        print(f"{nm:<26s}{a:>8.3f}{f'[{lo2:.3f}, {hi2:.3f}]':>18s}"
              f"{rec:>8.0%}{int((m2 & loss).sum()):>9d}")

    dl, dh = boot(lambda j: (
        auroc(-hybrid[np.searchsorted(te_idx, j)], y[j] < LEVEL * bench)
        - auroc(-sat[np.searchsorted(te_idx, j)], y[j] < LEVEL * bench)
        if 0 < (y[j] < LEVEL * bench).sum() < len(j) else None), pte, te_idx)
    dd = auroc(-hybrid, loss) - auroc(-sat, loss)
    print(f"\nhybrid minus satellite AUROC {dd:+.3f}, 95% CI "
          f"[{dl:+.3f}, {dh:+.3f}] -> "
          f"{'hybrid better' if dl > 0 else 'INCONCLUSIVE'}")

    OUT.write_text(json.dumps(
        {"backend": BACKEND, "cut": str(cut.date()),
         "n_train": int(tr_m.sum()), "n_test": int(te_m.sum()),
         "test_plots_seen_in_training": int(len(set(plots[te_m]) & set(plots[tr_m]))),
         "frac_test_with_history": float(have.mean()),
         "estimators": res,
         "paired_history_minus_satellite": {"delta": float(d.mean()),
                                            "ci": [lo, hi]},
         "decision": dec,
         "hybrid_minus_satellite_auroc": {"delta": dd, "ci": [dl, dh]},
         "caveat": ("Test plots were seen in training in their earlier "
                    "season. This evaluates a new SEASON at known fields, "
                    "not a new field. No claim about unseen fields follows "
                    "from it.")},
        indent=2), encoding="utf8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
