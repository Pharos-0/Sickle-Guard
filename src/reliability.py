"""RQ3: can we flag field-seasons whose yield prediction is badly wrong?

    python src/reliability.py

SETUP
-----
The predictor is V7, the random forest over all features, which was selected
on validation and is the only variant that nominally beat the train-mean
floor on test. Its errors are the thing to be screened.

Every signal below is computable at inference time with no yield label:

  rf_spread      standard deviation across the forest's own trees, relative
                 to the prediction. The textbook ensemble-disagreement
                 signal, and here the members really are independent draws.
  seed_spread    spread across forests fitted with different random seeds.
  n_clear        count of cloud-free Sentinel-2 acquisitions -- an input
                 quality measure, not a model output.
  n_px           how many pixels the plot occupies.
  ood            Mahalanobis-style distance from the training feature
                 distribution, using a diagonal covariance because 80
                 training plots cannot support a full one.
  pred           the predicted yield itself, as an operational ranking.

TARGET AND LEAKAGE
------------------
A field-season is "high error" when its absolute percentage error exceeds a
threshold. The threshold grid is declared here before the results are read,
and the headline uses the median APE of the test set so that roughly half
the cases are positive -- chosen because fixing an arbitrary 50 % from a
different experiment would be borrowing a number that was never justified
for this target.

The signals never see the label. The screening evaluation does, because
scoring a screen requires knowing which predictions were wrong. No threshold
on any signal is fitted on the test set: AUROC is threshold-free, and the
workload-based operating points are quantiles of the signal itself, which is
label-free.

UNCERTAINTY
-----------
Grouped bootstrap over the 27 test PLOTS. A plot contributes 2-4 correlated
seasons; treating them as independent would shrink every interval.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import BACKEND, make_rf, tree_predictions  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
FEATS = PROJ / "results" / "features.csv"
OUT = PROJ / "results" / "metrics" / "reliability.json"
SEEDS = [0, 1, 2, 3, 4]
WORKLOADS = [0.20, 0.33, 0.50]


def load():
    with FEATS.open(encoding="utf8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["yield"] = float(r["yield"])
        r["plot"] = float(r["plot"])
    return rows


def cols_all():
    S2 = ["s2_ndvi", "s2_evi", "s2_ndwi", "s2_ndmi", "s2_reci",
          "s2_red", "s2_nir", "s2_swir"]
    S1 = ["s1_vv", "s1_vh", "s1_rvi"]
    st = ["mean", "max", "min", "sd"]
    return ([f"{b}_{s}" for b in S2 for s in st]
            + ["ph_peak_frac", "ph_rise", "ph_fall", "ph_n_acq"]
            + [f"{b}_{s}" for b in S1 for s in st]
            + ["area", "n_px", "s2_n_clear", "s1_n_acq"])


def matrix(rows, cols):
    X = np.empty((len(rows), len(cols)), float)
    for j, c in enumerate(cols):
        for i, r in enumerate(rows):
            v = r.get(c, "")
            X[i, j] = float(v) if v not in ("", "nan", "None") else np.nan
    return X


def auroc(score, pos):
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_ci(score, pos, plots, n_boot=2000, seed=0):
    up = sorted(set(plots))
    by = {p: np.where(plots == p)[0] for p in up}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        s = [up[i] for i in rng.integers(0, len(up), len(up))]
        idx = np.concatenate([by[p] for p in s])
        if pos[idx].all() or not pos[idx].any():
            continue
        v.append(auroc(score[idx], pos[idx]))
    if not v:
        return {"lo": float("nan"), "hi": float("nan")}
    v = np.array(v)
    return {"lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5))}


def screen(score, pos, workload):
    """Refer the top `workload` fraction by signal. Threshold is a quantile
    of the signal itself, so no label is used to set it."""
    k = max(1, int(round(workload * len(score))))
    flag = np.zeros(len(score), bool)
    flag[np.argsort(-score)[:k]] = True
    tp = int((flag & pos).sum())
    return {"workload": float(flag.mean()), "n_referred": int(flag.sum()),
            "n_caught": tp, "n_missed": int(pos.sum() - tp),
            "recall": tp / max(1, int(pos.sum())),
            "precision": tp / max(1, int(flag.sum())),
            "random_expected_recall": float(flag.mean())}


def main() -> int:
    rows = load()
    cols = cols_all()
    tr = [r for r in rows if r["split"] in ("train", "val")]
    te = [r for r in rows if r["split"] == "test"]
    ytr = np.array([r["yield"] for r in tr])
    yte = np.array([r["yield"] for r in te])
    plots = np.array([r["plot"] for r in te])
    print(f"train+val {len(tr)} rows | test {len(te)} rows from "
          f"{len(set(plots))} plots")

    Xtr, Xte = matrix(tr, cols), matrix(te, cols)
    med = np.nanmedian(Xtr, axis=0)
    med[~np.isfinite(med)] = 0.0
    for X in (Xtr, Xte):
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(med, idx[1])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

    # ---- fit the forests, keep tree-level and seed-level spread --------
    per_seed, tree_preds = [], None
    for s in SEEDS:
        m = make_rf(s).fit(Xtr_s, ytr)
        per_seed.append(m.predict(Xte_s))
        if tree_preds is None:
            tree_preds = tree_predictions(m, Xte_s)
    per_seed = np.vstack(per_seed)
    pred = per_seed.mean(0)
    print(f"backend: {BACKEND}")

    # ---- quantile regression interval width ----------------------------
    # A genuine predictive interval, not a spread proxy. Fitted on
    # train+val only; its WIDTH is label-free at inference.
    qwidth = None
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        qlo = GradientBoostingRegressor(loss="quantile", alpha=0.05,
                                        n_estimators=200, max_depth=2,
                                        learning_rate=0.05,
                                        random_state=0).fit(Xtr_s, ytr)
        qhi = GradientBoostingRegressor(loss="quantile", alpha=0.95,
                                        n_estimators=200, max_depth=2,
                                        learning_rate=0.05,
                                        random_state=0).fit(Xtr_s, ytr)
        lo_p, hi_p = qlo.predict(Xte_s), qhi.predict(Xte_s)
        qwidth = (hi_p - lo_p) / np.abs(pred)
        cover = float(((yte >= lo_p) & (yte <= hi_p)).mean())
        print(f"quantile interval: nominal 90% coverage, empirical "
              f"{cover:.0%} on test")
    except Exception as exc:                      # pragma: no cover
        print(f"quantile intervals unavailable: {type(exc).__name__}")
        cover = None

    ape = np.abs(pred - yte) / yte * 100.0
    print(f"V7 on test: MAE {np.mean(np.abs(pred-yte)):.1f} kg/acre, "
          f"median APE {np.median(ape):.1f}%")

    # ---- label-free signals ---------------------------------------------
    sig = {
        "rf_spread": tree_preds.std(axis=0, ddof=1) / np.abs(pred),
        "seed_spread": per_seed.std(axis=0, ddof=1) / np.abs(pred),
        "low_n_clear": -np.array([float(r["s2_n_clear"]) for r in te]),
        "small_n_px": -np.array([float(r["n_px"]) for r in te]),
        "ood_distance": np.sqrt(((Xte_s - Xtr_s.mean(0)) ** 2
                                 / (Xtr_s.var(0) + 1e-9)).mean(1)),
        "low_predicted_yield": -pred,
    }
    if qwidth is not None:
        sig["quantile_width"] = qwidth

    thr = float(np.median(ape))
    print(f"\nhigh-error threshold: APE > {thr:.1f}% (test median), "
          f"{int((ape > thr).sum())} of {len(ape)} positive\n")
    pos = ape > thr

    print("THRESHOLD-FREE RANKING (random = 0.500)")
    print(f"{'signal':<22s}{'AUROC':>8s}{'95% CI':>20s}{'verdict':>16s}")
    res = {}
    for k, v in sig.items():
        a = auroc(v, pos)
        ci = auroc_ci(v, pos, plots)
        ok = not (ci["lo"] <= 0.5 <= ci["hi"])
        res[k] = {"auroc": a, "ci": ci, "informative": bool(ok)}
        print(f"{k:<22s}{a:>8.3f}"
              f"{f'[{ci[chr(108)+chr(111)]:.3f}, {ci[chr(104)+chr(105)]:.3f}]':>20s}"
              f"{('informative' if ok else 'not demonstrated'):>16s}")

    best = max(res, key=lambda k: res[k]["auroc"])
    print(f"\nSCREENING at fixed verification budgets, signal = {best}")
    print(f"{'workload':>9s}{'referred':>10s}{'caught':>8s}{'recall':>8s}"
          f"{'random':>8s}{'precision':>11s}")
    scr = {}
    for w in WORKLOADS:
        d = screen(sig[best], pos, w)
        scr[f"{w:.2f}"] = d
        print(f"{w:>9.0%}{d['n_referred']:>10d}{d['n_caught']:>8d}"
              f"{d['recall']:>8.0%}{d['random_expected_recall']:>8.0%}"
              f"{d['precision']:>11.0%}")

    print("\nSENSITIVITY to the high-error threshold (AUROC of each signal)")
    hdr = f"{'APE >':>7s}{'n_pos':>7s}" + "".join(f"{k[:11]:>13s}" for k in sig)
    print(hdr)
    sens = {}
    for t in (15.0, 20.0, 25.0, 30.0, 40.0):
        p2 = ape > t
        if p2.all() or not p2.any():
            continue
        row = {k: auroc(v, p2) for k, v in sig.items()}
        # Intervals across the whole pre-declared grid, so the reader can
        # see where the signal strengthens without the headline being moved
        # to whichever threshold happened to look best.
        cis = {k: auroc_ci(v, p2, plots, n_boot=1200) for k, v in sig.items()}
        sens[str(t)] = {"n_pos": int(p2.sum()),
                        **{k: {"auroc": row[k], "ci": cis[k],
                               "excludes_half": bool(
                                   not (cis[k]["lo"] <= 0.5 <= cis[k]["hi"]))}
                           for k in row}}
        star = "".join("*" if sens[str(t)][k]["excludes_half"] else " "
                       for k in sig)
        print(f"{t:>7.0f}{int(p2.sum()):>7d}"
              + "".join(f"{row[k]:>13.3f}" for k in sig) + "  " + star)

    # Report, but do not promote, any grid point that clears chance.
    # Selecting a threshold after seeing it would not be confirmatory.
    strong = [(t, k) for t, d in sens.items() for k in sig
              if isinstance(d.get(k), dict) and d[k]["excludes_half"]]
    if strong:
        print("")
        print("Grid points whose CI excludes chance (EXPLORATORY). The")
        print("pre-declared headline is the median split above, which "
              "does not.")
        for t, k in strong:
            d = sens[t][k]
            print(f"   APE > {float(t):.0f}%  {k:<20s} "
                  f"AUROC {d[chr(39)+chr(39)] if False else d['auroc']:.3f} "
                  f"[{d['ci']['lo']:.3f}, {d['ci']['hi']:.3f}] "
                  f"(n_pos {sens[t]['n_pos']})")

    any_good = [k for k, v in res.items() if v["informative"]]
    verdict = ("informative: " + ", ".join(any_good) if any_good
               else "no signal demonstrated better than chance")
    print(f"\nVERDICT: {verdict}")
    print(f"CAVEAT: {len(set(plots))} independent test plots, "
          f"{int(pos.sum())} high-error field-seasons. One plot is "
          f"{100/len(set(plots)):.0f}% of the independent sample.")

    OUT.write_text(json.dumps(
        {"predictor": "V7 random forest, all features",
         "backend": BACKEND,
         "quantile_interval_empirical_coverage_nominal_90": cover,
         "n_test_rows": len(te), "n_test_plots": int(len(set(plots))),
         "mae": float(np.mean(np.abs(pred - yte))),
         "median_ape": float(np.median(ape)),
         "high_error_threshold_ape": thr,
         "n_high_error": int(pos.sum()),
         "signals": res, "screening_best_signal": best, "screening": scr,
         "sensitivity": sens,
         "exploratory_grid_points_clearing_chance":
             [{"ape_threshold": float(t), "signal": k}
              for t, k in strong],
         "verdict": verdict,
         "note": ("Signals use no label. Workload thresholds are quantiles "
                  "of the signal, so they are label-free. Bootstrap "
                  "resamples plots.")},
        indent=2), encoding="utf8")
    np.save(PROJ / "results/predictions/test_pred_v7.npy", pred)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
