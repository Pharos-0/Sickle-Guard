"""Does SICKLE-Guard add anything over a causal-history-only baseline?

    python src/does_satellite_add.py

WHY THIS HAD TO BE RUN
----------------------
The headline reported for SICKLE-Guard -- ranking AUROC 0.904 on the
forward-in-time split -- was obtained where causal history covered 100 % of
test field-seasons. On that split the history-first hybrid and a
history-only baseline are the SAME PREDICTOR, so the number said nothing
about the satellite component. It was attributed to the system when it
belonged entirely to the history table.

The question therefore has to be asked where a fallback can matter: under
grouped cross-validation, causal history exists for 188 of 319 field-seasons
(59 %), leaving 131 genuine cold starts.

THE COMPARISON
--------------
  history_only   causal history where it exists, GLOBAL TRAIN MEAN otherwise
                 (no imagery anywhere in the system)
  hybrid         causal history where it exists, SATELLITE MODEL otherwise
                 (what SICKLE-Guard actually does)

Identical everywhere history exists, so every difference between them comes
from the 131 cold-start fields alone. If they do not separate, the satellite
component earns nothing and the system should be a history table with an
honest "no estimate available" for new fields.

Reported overall and restricted to the cold-start subset, because an
average over both regimes dilutes the only place the question lives.
Paired, bootstrapped over plots.
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
OUT = PROJ / "results" / "metrics" / "does_satellite_add.json"
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
    have = np.isfinite(hist)
    cold = ~have
    gmean = float(y.mean())

    hist_only = np.where(have, hist, gmean)      # no imagery anywhere
    hybrid = np.where(have, hist, sat)           # SICKLE-Guard

    print(f"{len(y)} field-seasons, {len(set(plots))} plots")
    print(f"  with causal history : {int(have.sum())}  (identical in both)")
    print(f"  cold start          : {int(cold.sum())}  (the ONLY difference)\n")

    allidx = np.arange(len(y))
    loss = y < LEVEL * gmean

    print("MAE")
    print(f"{'':28s}{'overall':>10s}{'cold-start only':>18s}")
    for nm, p in (("history-only (no imagery)", hist_only),
                  ("hybrid = SICKLE-Guard", hybrid)):
        print(f"{nm:<28s}{np.mean(np.abs(p - y)):>10.1f}"
              f"{np.mean(np.abs(p[cold] - y[cold])):>18.1f}")

    d_all = np.abs(hybrid - y) - np.abs(hist_only - y)
    lo_a, hi_a = boot(lambda j: float(np.mean(np.abs(hybrid[j] - y[j])
                                              - np.abs(hist_only[j] - y[j]))),
                      plots, allidx)
    ci = np.where(cold)[0]
    lo_c, hi_c = boot(lambda j: float(np.mean(np.abs(hybrid[j] - y[j])
                                              - np.abs(hist_only[j] - y[j]))),
                      plots[cold], ci)
    print(f"\nhybrid minus history-only (negative = satellite helps)")
    print(f"  overall    {d_all.mean():+7.1f} kg/acre  95% CI "
          f"[{lo_a:+.1f}, {hi_a:+.1f}]")
    print(f"  cold start {np.mean(d_all[cold]):+7.1f} kg/acre  95% CI "
          f"[{lo_c:+.1f}, {hi_c:+.1f}]")
    mae_helps = hi_c < 0

    print(f"\nLOSS RANKING, {int(loss.sum())} loss seasons of {len(y)}")
    print(f"{'':28s}{'AUROC':>8s}{'95% CI':>18s}{'recall@33%':>12s}")
    rank = {}
    for nm, p in (("history-only (no imagery)", hist_only),
                  ("hybrid = SICKLE-Guard", hybrid)):
        a = auroc(-p, loss)
        l_, h_ = boot(lambda j, p_=p: (auroc(-p_[j], loss[j])
                      if 0 < loss[j].sum() < len(j) else None), plots, allidx)
        k = max(1, int(round(BUDGET * len(y))))
        sel = np.argsort(p)[:k]
        m = np.zeros(len(y), bool); m[sel] = True
        rec = float((m & loss).sum() / loss.sum())
        rank[nm] = {"auroc": a, "ci": [l_, h_], "recall": rec}
        print(f"{nm:<28s}{a:>8.3f}{f'[{l_:.3f}, {h_:.3f}]':>18s}{rec:>12.0%}")

    dd = auroc(-hybrid, loss) - auroc(-hist_only, loss)
    dl, dh = boot(lambda j: (auroc(-hybrid[j], loss[j])
                             - auroc(-hist_only[j], loss[j])
                             if 0 < loss[j].sum() < len(j) else None),
                  plots, allidx)
    print(f"\n  hybrid minus history-only AUROC {dd:+.3f}, 95% CI "
          f"[{dl:+.3f}, {dh:+.3f}]")
    rank_helps = dl > 0

    verdict = ("satellite earns its place in the system" if (mae_helps or rank_helps)
               else "satellite adds NOTHING measurable over history-only; "
                    "SICKLE-Guard should be a history table")
    print(f"\nVERDICT: {verdict}")
    if not (mae_helps or rank_helps):
        print("  Neither the error on cold-start fields nor the overall")
        print("  ranking separates from a system with no imagery at all.")

    OUT.write_text(json.dumps(
        {"n": len(y), "n_plots": int(len(set(plots))),
         "n_with_history": int(have.sum()), "n_cold_start": int(cold.sum()),
         "mae": {"history_only_overall": float(np.mean(np.abs(hist_only - y))),
                 "hybrid_overall": float(np.mean(np.abs(hybrid - y))),
                 "history_only_cold": float(np.mean(np.abs(hist_only[cold] - y[cold]))),
                 "hybrid_cold": float(np.mean(np.abs(hybrid[cold] - y[cold])))},
         "paired_mae_hybrid_minus_history_only": {
             "overall": {"delta": float(d_all.mean()), "ci": [lo_a, hi_a]},
             "cold_start": {"delta": float(np.mean(d_all[cold])),
                            "ci": [lo_c, hi_c]}},
         "ranking": rank,
         "ranking_delta": {"delta": dd, "ci": [dl, dh]},
         "satellite_helps_mae_on_cold_start": bool(mae_helps),
         "satellite_helps_ranking": bool(rank_helps),
         "verdict": verdict,
         "note": ("The forward-in-time split cannot answer this: causal "
                  "history covered 100 % of its test set, so hybrid and "
                  "history-only were the same predictor there and the "
                  "reported 0.904 belonged entirely to history.")},
        indent=2), encoding="utf8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
