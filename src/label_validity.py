"""What is the "loss" label actually made of, and is the 72 % circular?

    python src/label_validity.py

THE CONCERN
-----------
The headline -- 72 % of losses caught at a 33 % verification budget, AUROC
0.838 -- is measured against a label defined as

    yield < 0.80 x (mean yield of the training split)

That is a RELATIVE threshold against the area average, computed on
farmer-recalled yields. It is not a measurement of crop damage. Two
consequences follow and neither has been tested:

  1. A chronically low-yielding field is labelled "loss" every season,
     including seasons where nothing went wrong.
  2. A field that lost half its crop but started from a high base may not
     cross the threshold at all.

If the label is mostly (1), then a predictor built from the field's own past
harvests is close to circular: history predicts the field's typical level,
and the label largely IS the field's typical level. High AUROC would then
mean "this tool identifies fields that are usually poor", which an
administrator already knows and which is not an insurable event.

THE TEST
--------
Split the labelled losses by whether the plot is ever NOT in loss:

  chronic plots  every observed season below the threshold
  event plots    some seasons below, some above  <- the operational case
  never plots    no season below

Then score the ranking three ways: overall, on chronic+never plots only,
and on event plots only. The third is the honest question, because it is
the only one where the answer is not already known from the field's record.
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
OUT = PROJ / "results" / "metrics" / "label_validity.json"
LEVEL, BUDGET = 0.80, 0.33


def parse(d):
    for f in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(d.strip(), f)
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
    tab = {r["UNIQUE_ID"]: r for r in csv.DictReader(
        (ROOT / "sickle_dataset_tabular.csv").open(encoding="utf8"))}
    y = np.array([float(r["yield"]) for r in feats])
    plots = np.array([float(r["plot"]) for r in feats])
    sow = [parse(tab[r["uid"]]["SOWING_DATE"]) for r in feats]
    harv = [parse(tab[r["uid"]]["HARVESTING_DATE"]) for r in feats]
    sat = np.load(PROJ / "results/predictions/oof_pred.npy")

    byplot = defaultdict(list)
    for i, p in enumerate(plots):
        byplot[p].append(i)
    hist = np.full(len(y), np.nan)
    for _, v in byplot.items():
        for i in v:
            prior = [j for j in v if harv[j] < sow[i]]
            if prior:
                hist[i] = float(np.mean(y[prior]))
    gmean = float(y.mean())
    hist_only = np.where(np.isfinite(hist), hist, gmean)
    loss = y < LEVEL * gmean

    chronic, event, never = [], [], []
    for p, v in byplot.items():
        f = [loss[i] for i in v]
        (chronic if all(f) else never if not any(f) else event).append(p)
    n_chr = sum(int(loss[i]) for p in chronic for i in byplot[p])
    n_evt = sum(int(loss[i]) for p in event for i in byplot[p])

    print(f"{len(y)} field-seasons, {len(byplot)} plots, "
          f"{int(loss.sum())} labelled losses")
    print(f"  threshold = {LEVEL:.0%} of the training mean "
          f"= {LEVEL * gmean:.0f} kg/acre (a RELATIVE cut on recalled yield)\n")
    print(f"{'plot type':<34s}{'plots':>7s}{'loss seasons':>14s}")
    print(f"{'chronic (every season in loss)':<34s}{len(chronic):>7d}{n_chr:>14d}")
    print(f"{'event (some in, some out)':<34s}{len(event):>7d}{n_evt:>14d}")
    print(f"{'never in loss':<34s}{len(never):>7d}{0:>14d}")
    frac = n_chr / max(1, int(loss.sum()))
    print(f"\n  {frac:.0%} of labelled 'losses' are at chronically "
          f"low-yielding plots.")
    print("  Those are a property of the land, not an insurable event.")

    allidx = np.arange(len(y))
    print(f"\nRANKING BY SUBSET (history-only predictor)")
    print(f"{'evaluated on':<34s}{'n':>6s}{'losses':>8s}{'AUROC':>8s}"
          f"{'95% CI':>18s}")
    res = {}
    subsets = {
        "everything (the headline)": allidx,
        "chronic + never plots only":
            np.array([i for p in chronic + never for i in byplot[p]]),
        "EVENT plots only (the real case)":
            np.array([i for p in event for i in byplot[p]]),
    }
    for nm, idx in subsets.items():
        if len(idx) == 0 or not (0 < loss[idx].sum() < len(idx)):
            print(f"{nm:<34s}{len(idx):>6d}  degenerate")
            continue
        a = auroc(-hist_only[idx], loss[idx])
        lo, hi = boot(lambda j: (auroc(-hist_only[j], loss[j])
                                 if 0 < loss[j].sum() < len(j) else None),
                      plots[idx], idx)
        res[nm] = {"n": int(len(idx)), "n_loss": int(loss[idx].sum()),
                   "auroc": a, "ci": [lo, hi],
                   "informative": bool(lo > 0.5)}
        print(f"{nm:<34s}{len(idx):>6d}{int(loss[idx].sum()):>8d}{a:>8.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18s}")

    key = "EVENT plots only (the real case)"
    if key in res:
        r = res[key]
        print(f"\n  On event plots the predictor must say WHICH SEASON at a")
        print(f"  field went wrong, not which fields are poor.")
        print(f"  AUROC {r['auroc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] "
              f"-> {'informative' if r['informative'] else 'NOT demonstrated'}")

    # recall at budget, restricted to event plots
    if key in res:
        idx = subsets[key]
        k = max(1, int(round(BUDGET * len(idx))))
        sel = idx[np.argsort(hist_only[idx])[:k]]
        m = np.zeros(len(y), bool); m[sel] = True
        rec = float((m & loss)[idx].sum() / max(1, loss[idx].sum()))
        print(f"\n  recall at a {BUDGET:.0%} budget on event plots: "
              f"{rec:.0%} (random {BUDGET:.0%})")
        res[key]["recall_at_budget"] = rec

    verdict = ("the headline is largely circular: it is driven by "
               "chronically poor fields, which a record already identifies"
               if res.get(key, {}).get("informative") is False else
               "the predictor retains signal on event losses")
    print(f"\nVERDICT: {verdict}")
    print("\nWhat the 72 % headline does and does not mean:")
    print("  DOES  -- 72 % of field-seasons below 80 % of the area average")
    print("           are among the 33 % lowest-ranked.")
    print("  DOES NOT -- 72 % of farmers suffering crop damage would be")
    print("           identified. The label is a relative cut on recalled")
    print("           yield, not a measurement of damage.")

    OUT.write_text(json.dumps(
        {"threshold_level": LEVEL, "threshold_value": LEVEL * gmean,
         "n": len(y), "n_loss": int(loss.sum()),
         "plots": {"chronic": len(chronic), "event": len(event),
                   "never": len(never)},
         "loss_seasons": {"at_chronic_plots": n_chr, "at_event_plots": n_evt},
         "frac_losses_chronic": frac,
         "ranking_by_subset": res, "verdict": verdict,
         "construct_validity_note": (
             "The label is yield < 80% of the training-split mean, on "
             "farmer-recalled yields. It is a relative cut against the area, "
             "not a measurement of crop damage. A chronically low-yielding "
             "field is labelled a loss every season; a field that lost half "
             "its crop from a high base may never be labelled at all.")},
        indent=2), encoding="utf8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
