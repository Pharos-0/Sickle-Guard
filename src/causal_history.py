"""History-first strategy with STRICTLY CAUSAL history. No future harvests.

    python src/causal_history.py

WHY THIS SUPERSEDES THE EARLIER COMPARISON
-------------------------------------------
`src/site_vs_season.py` built its plot-history baseline by leave-one-season-
out: a field-season's expectation was the mean of that plot's OTHER seasons.
Every plot in this dataset has both a 2019 and a 2020 season, so that
baseline predicted the 2019 harvest partly from the 2020 harvest. That is
future information and it is not available at any real decision point.

The reported advantage of history over satellite (80.5 kg/acre) was
therefore inflated by temporal leakage, and the headline needed re-testing
rather than defending.

THE CAUSAL RULE
---------------
A season's history is the mean yield of seasons at the SAME plot whose
HARVEST DATE precedes this season's SOWING DATE. Nothing else counts.
Dates come from the released index (`SOWING_DATE`, `HARVESTING_DATE`), which
has distinct harvest dates for all 131 plots, so the ordering is unambiguous.

The consequence is severe and is the point: a plot observed in 2019 and 2020
gives history for the 2020 season only. The 2019 season is a genuine cold
start. Roughly half the dataset therefore has no usable history, which is
exactly the position a deployed system is in during its first year.

WHAT IS COMPARED
----------------
  global mean           no information
  causal history        past harvests of this field only, no imagery
  satellite model       grouped-CV out-of-fold, no history
  history-first hybrid  causal history where it exists, else satellite

Evaluated overall and split by regime, because an average over both regimes
hides which estimator is doing the work.
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

PROJ = Path(__file__).resolve().parents[1]
# dataset root; SICKLE_DATA_ROOT as in sickle-project/.env.example
ROOT = Path(__import__("os").environ.get(
    "SICKLE_DATA_ROOT", "D:/FPD/sickledataset extracted"))
OUT = PROJ / "results" / "metrics" / "causal_history.json"


def parse(d: str):
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


def plot_boot(fn, plots, n_boot=4000, seed=0, subset=None):
    """Bootstrap over plots.

    `subset` holds GLOBAL row indices when the caller is evaluating a
    subset; without it the positions returned here index the subset array
    while fn indexes the global arrays, which silently produces NaN.
    """
    if subset is None:
        subset = np.arange(len(plots))
    up = sorted(set(plots))
    by = {p: subset[np.where(plots == p)[0]] for p in up}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        s = [up[i] for i in rng.integers(0, len(up), len(up))]
        idx = np.concatenate([by[p] for p in s])
        r = fn(idx)
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
    sow, harv = [], []
    for r in feats:
        t = tab[r["uid"]]
        sow.append(parse(t["SOWING_DATE"]))
        harv.append(parse(t["HARVESTING_DATE"]))
    bad = [i for i, (a, b) in enumerate(zip(sow, harv)) if a is None or b is None]
    if bad:
        raise SystemExit(f"{len(bad)} rows have unparseable dates, e.g. "
                         f"{tab[feats[bad[0]]['uid']]['SOWING_DATE']!r}")
    print(f"{len(y)} field-seasons, {len(set(plots))} plots, all dates parsed")

    # ---- strictly causal history ---------------------------------------
    byplot = defaultdict(list)
    for i, p in enumerate(plots):
        byplot[p].append(i)
    hist = np.full(len(y), np.nan)
    n_prior = np.zeros(len(y), int)
    for _, v in byplot.items():
        for i in v:
            prior = [j for j in v if harv[j] < sow[i]]
            n_prior[i] = len(prior)
            if prior:
                hist[i] = float(np.mean([y[j] for j in prior]))
    have = np.isfinite(hist)
    print(f"\nSTRICTLY CAUSAL history available for {have.sum()}/{len(y)} "
          f"({have.mean():.0%}) field-seasons")
    print(f"   prior-season counts: "
          f"{dict(zip(*np.unique(n_prior, return_counts=True)))}")
    # contrast with the leaky version
    leaky = np.full(len(y), np.nan)
    for _, v in byplot.items():
        for i in v:
            o = [j for j in v if j != i]
            if o:
                leaky[i] = float(np.mean(y[o]))
    print(f"   the earlier leave-one-season-out version had history for "
          f"{np.isfinite(leaky).sum()}/{len(y)} (100%) -- it used future "
          f"harvests")

    sat = np.load(PROJ / "results/predictions/oof_pred.npy")
    gmean = float(y.mean())
    hybrid = np.where(have, hist, sat)

    def mae(p, m):
        return float(np.mean(np.abs(p[m] - y[m])))

    allm = np.ones(len(y), bool)
    print(f"\nMAE BY REGIME (kg/acre)")
    print(f"{'estimator':<26s}{'all':>9s}{'has history':>13s}{'cold start':>12s}")
    rows_out = {}
    for nm, p in (("global mean", np.full(len(y), gmean)),
                  ("causal history", hist),
                  ("satellite model", sat),
                  ("history-first hybrid", hybrid)):
        a = mae(p, allm & np.isfinite(p))
        h = mae(p, have & np.isfinite(p)) if (have & np.isfinite(p)).any() else float("nan")
        c = mae(p, ~have & np.isfinite(p)) if (~have & np.isfinite(p)).any() else float("nan")
        rows_out[nm] = {"all": a, "has_history": h, "cold_start": c}
        print(f"{nm:<26s}{a:>9.1f}"
              f"{(f'{h:.1f}' if np.isfinite(h) else '-'):>13s}"
              f"{(f'{c:.1f}' if np.isfinite(c) else '-'):>12s}")
    print(f"{'n':<26s}{len(y):>9d}{int(have.sum()):>13d}{int((~have).sum()):>12d}")

    # ---- the decisive paired test, ON FIELDS WITH HISTORY ONLY ---------
    print(f"\nPAIRED: causal history vs satellite, on the "
          f"{int(have.sum())} field-seasons that HAVE history")
    hi_idx = np.where(have)[0]
    d = np.abs(hist[hi_idx] - y[hi_idx]) - np.abs(sat[hi_idx] - y[hi_idx])
    lo, hiq = plot_boot(
        lambda i: float(np.mean(np.abs(hist[i] - y[i]) - np.abs(sat[i] - y[i]))),
        plots[hi_idx], subset=hi_idx)
    verdict = ("history better" if hiq < 0 else
               "satellite better" if lo > 0 else "INCONCLUSIVE")
    print(f"   history minus satellite: {d.mean():+.1f} kg/acre, "
          f"95% CI [{lo:+.1f}, {hiq:+.1f}]  -> {verdict}")
    print(f"   (the leaky comparison reported +80.5 in satellite's "
          f"disfavour, CI [+40.9, +124.8])")

    # ---- verification ranking under the causal rule --------------------
    bench = gmean
    loss = y < 0.80 * bench
    print(f"\nVERIFICATION RANKING, {int(loss.sum())} loss seasons of {len(y)}")
    print(f"{'strategy':<26s}{'AUROC':>8s}{'95% CI':>18s}{'recall@33%':>12s}")
    rank = {}
    for nm, sc in (("satellite model", -sat),
                   ("history-first hybrid", -hybrid),
                   ("leaky history (old)", -leaky)):
        a = auroc(sc, loss)
        l_, h_ = plot_boot(lambda i, s=sc: (auroc(s[i], loss[i])
                           if 0 < loss[i].sum() < len(i) else None), plots)
        k = max(1, int(round(0.33 * len(y))))
        sel = np.argsort(-sc)[:k]
        m = np.zeros(len(y), bool); m[sel] = True
        rec = float((m & loss).sum() / loss.sum())
        rank[nm] = {"auroc": a, "ci": [l_, h_], "recall_33": rec}
        print(f"{nm:<26s}{a:>8.3f}{f'[{l_:.3f}, {h_:.3f}]':>18s}{rec:>12.0%}")

    dl, dh = plot_boot(lambda i: (auroc(-hybrid[i], loss[i])
                                  - auroc(-sat[i], loss[i])
                                  if 0 < loss[i].sum() < len(i) else None), plots)
    dd = auroc(-hybrid, loss) - auroc(-sat, loss)
    print(f"\n   hybrid minus satellite AUROC: {dd:+.3f}, "
          f"95% CI [{dl:+.3f}, {dh:+.3f}]  -> "
          f"{'hybrid better' if dl > 0 else 'INCONCLUSIVE'}")

    OUT.write_text(json.dumps(
        {"n": len(y), "n_plots": int(len(set(plots))),
         "n_with_causal_history": int(have.sum()),
         "frac_with_causal_history": float(have.mean()),
         "mae_by_regime": rows_out,
         "paired_history_minus_satellite_on_history_subset": {
             "delta": float(d.mean()), "ci": [lo, hiq], "verdict": verdict},
         "verification_ranking": rank,
         "hybrid_minus_satellite_auroc": {"delta": dd, "ci": [dl, dh]},
         "note": ("Supersedes site_vs_season.py, whose history baseline used "
                  "future harvests. Causal history = mean yield of seasons "
                  "at the same plot harvested before this season was sown.")},
        indent=2), encoding="utf8")
    np.save(PROJ / "results/predictions/causal_hist.npy", hist)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
