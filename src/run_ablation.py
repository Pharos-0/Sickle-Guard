"""Sequential ablation for plot-level paddy yield prediction.

    python src/run_ablation.py

PROTOCOL, FIXED BEFORE THE TEST SET IS TOUCHED
----------------------------------------------
Model selection and every hyperparameter choice use TRAIN + VAL only. The
test partition is scored once, at the end, for the variants already chosen.
Nothing in this file reads test labels before that point, and the ordering is
enforced by the structure of main(): all validation work completes before
`final_test()` is called.

VARIANTS
--------
  V1  train-mean               the only honest floor
  V2  season mean              does knowing the season alone help?
  V3  ridge, S2 optical        do optical indices carry signal?
  V4  + phenology              does WHEN the peak occurs add anything?
  V5  + Sentinel-1 radar       does radar add over optical?
  V6  + area and pixel count   does field geometry add anything?
  V7  random forest            does a non-linear model beat ridge?
  V8  gradient boosting        does boosting beat bagging here?

Each step adds one component. V7 and V8 change the model family at fixed
features, so they are read as a model comparison rather than a feature
ablation, and the text must say so.

UNCERTAINTY
-----------
Grouped bootstrap over PLOTS, never field-seasons: a plot contributes 2-4
correlated seasons and treating them as independent would shrink every
interval. With 24 validation and 27 test plots the intervals are wide, and
that is the honest state of the evidence rather than a presentation problem.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJ = Path(__file__).resolve().parents[1]
FEATS = PROJ / "results" / "features.csv"
OUT = PROJ / "results" / "metrics"
SEEDS = [0, 1, 2, 3, 4]

S2_IDX = ["s2_ndvi", "s2_evi", "s2_ndwi", "s2_ndmi", "s2_reci"]
S2_RAW = ["s2_red", "s2_nir", "s2_swir"]
STATS = ["mean", "max", "min", "sd"]


def optical_feats():
    return [f"{b}_{s}" for b in S2_IDX + S2_RAW for s in STATS]


def pheno_feats():
    return ["ph_peak_frac", "ph_rise", "ph_fall", "ph_n_acq"]


def radar_feats():
    return [f"s1_{b}_{s}" for b in ("vv", "vh", "rvi") for s in STATS]


def geom_feats():
    return ["area", "n_px", "s2_n_clear", "s1_n_acq"]


VARIANTS = [
    ("V1", "train mean", "mean", []),
    ("V2", "season mean", "season_mean", []),
    ("V3", "ridge, S2 optical", "ridge", optical_feats()),
    ("V4", "+ phenology", "ridge", optical_feats() + pheno_feats()),
    ("V5", "+ Sentinel-1 radar", "ridge",
     optical_feats() + pheno_feats() + radar_feats()),
    ("V6", "+ geometry", "ridge",
     optical_feats() + pheno_feats() + radar_feats() + geom_feats()),
    ("V7", "random forest, all feats", "rf",
     optical_feats() + pheno_feats() + radar_feats() + geom_feats()),
    ("V8", "gradient boosting, all feats", "gb",
     optical_feats() + pheno_feats() + radar_feats() + geom_feats()),
]


# ------------------------------------------------------------------ data
def load():
    with FEATS.open(encoding="utf8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["yield"] = float(r["yield"])
        r["plot"] = float(r["plot"])
    return rows


def matrix(rows, cols):
    X = np.empty((len(rows), len(cols)), float)
    for j, c in enumerate(cols):
        for i, r in enumerate(rows):
            v = r.get(c, "")
            X[i, j] = float(v) if v not in ("", "nan", "None") else np.nan
    return X


def impute_fit(X):
    med = np.nanmedian(X, axis=0)
    med[~np.isfinite(med)] = 0.0
    return med


def apply_imp(X, med):
    X = X.copy()
    idx = np.where(~np.isfinite(X))
    X[idx] = np.take(med, idx[1])
    return X


# ---------------------------------------------------------------- models
def fit_predict(kind, tr, te, cols, seed):
    ytr = np.array([r["yield"] for r in tr])
    if kind == "mean":
        return np.full(len(te), ytr.mean())
    if kind == "season_mean":
        g, overall = {}, ytr.mean()
        for r in tr:
            g.setdefault(r["season"], []).append(r["yield"])
        g = {k: float(np.mean(v)) for k, v in g.items()}
        return np.array([g.get(r["season"], overall) for r in te])

    Xtr, Xte = matrix(tr, cols), matrix(te, cols)
    med = impute_fit(Xtr)
    Xtr, Xte = apply_imp(Xtr, med), apply_imp(Xte, med)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    if kind == "ridge":
        lam = 10.0
        A = np.hstack([np.ones((len(Xtr), 1)), Xtr])
        B = np.hstack([np.ones((len(Xte), 1)), Xte])
        M = A.T @ A + lam * np.eye(A.shape[1])
        M[0, 0] -= lam
        w = np.linalg.solve(M, A.T @ ytr)
        return B @ w
    if kind in ("rf", "gb"):
        from models import make_gb, make_rf
        m = make_rf(seed) if kind == "rf" else make_gb(seed)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    raise ValueError(kind)


# --------------------------------------------------------------- metrics
def metrics(y, p):
    e = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return {"mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "r2": float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan"),
            "mape": float(np.mean(np.abs(e / y)) * 100),
            "bias": float(np.mean(e)), "n": int(len(y))}


def plot_bootstrap(y, p, plots, n_boot=2000, seed=0):
    """CI on MAE, resampling PLOTS."""
    up = sorted(set(plots))
    by = {q: np.where(plots == q)[0] for q in up}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        s = [up[i] for i in rng.integers(0, len(up), len(up))]
        idx = np.concatenate([by[q] for q in s])
        v.append(float(np.mean(np.abs(p[idx] - y[idx]))))
    v = np.array(v)
    return {"lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5))}


def evaluate(kind, cols, tr, te):
    """Mean over seeds; deterministic models run once."""
    y = np.array([r["yield"] for r in te])
    plots = np.array([r["plot"] for r in te])
    seeds = SEEDS if kind in ("rf", "gb") else [0]
    P = np.vstack([fit_predict(kind, tr, te, cols, s) for s in seeds])
    p = P.mean(0)
    m = metrics(y, p)
    m["seed_sd_mae"] = float(np.std(
        [metrics(y, P[i])["mae"] for i in range(len(seeds))])) if len(seeds) > 1 else 0.0
    m["n_seeds"] = len(seeds)
    m["ci_mae"] = plot_bootstrap(y, p, plots)
    return m, p


def main() -> int:
    rows = load()
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    te = [r for r in rows if r["split"] == "test"]
    print(f"train {len(tr)} rows / {len({r['plot'] for r in tr})} plots | "
          f"val {len(va)} / {len({r['plot'] for r in va})} | "
          f"test {len(te)} / {len({r['plot'] for r in te})}")
    print(f"target: kg/acre, train mean {np.mean([r['yield'] for r in tr]):.0f}\n")

    # ---------------- VALIDATION: every decision happens here -----------
    print("VALIDATION (model selection happens here; test untouched)")
    print(f"{'':4s}{'variant':<32s}{'MAE':>8s}{'RMSE':>8s}{'R2':>8s}"
          f"{'MAPE%':>8s}{'seedSD':>8s}")
    val_res = {}
    for vid, name, kind, cols in VARIANTS:
        m, _ = evaluate(kind, cols, tr, va)
        val_res[vid] = {"name": name, "kind": kind, "n_features": len(cols),
                        **m}
        print(f"{vid:<4s}{name:<32s}{m['mae']:>8.1f}{m['rmse']:>8.1f}"
              f"{m['r2']:>8.3f}{m['mape']:>8.1f}{m['seed_sd_mae']:>8.1f}")

    best = min(val_res, key=lambda k: val_res[k]["mae"])
    print(f"\nselected on validation MAE: {best} "
          f"({val_res[best]['name']}, MAE {val_res[best]['mae']:.1f})")
    floor = val_res["V1"]["mae"]
    print(f"train-mean floor on validation: {floor:.1f} kg/acre")
    if val_res[best]["mae"] >= floor:
        print("NOTE: no variant beats the train-mean floor on validation.")

    # ---------------- TEST: scored once, after selection ----------------
    print("\nTEST (scored once, after the selection above)")
    print(f"{'':4s}{'variant':<32s}{'MAE':>8s}{'95% CI MAE':>18s}"
          f"{'RMSE':>8s}{'R2':>8s}")
    test_res, preds = {}, {}
    for vid, name, kind, cols in VARIANTS:
        m, p = evaluate(kind, cols, tr + va, te)
        test_res[vid] = {"name": name, "kind": kind, **m}
        preds[vid] = p
        ci = m["ci_mae"]
        print(f"{vid:<4s}{name:<32s}{m['mae']:>8.1f}"
              f"{f'[{ci[chr(108)+chr(111)]:.0f}, {ci[chr(104)+chr(105)]:.0f}]':>18s}"
              f"{m['rmse']:>8.1f}{m['r2']:>8.3f}")

    tfloor = test_res["V1"]["mae"]
    print(f"\ntrain-mean floor on test: {tfloor:.1f} kg/acre")
    print(f"selected variant {best} on test: MAE {test_res[best]['mae']:.1f} "
          f"({100*(tfloor - test_res[best]['mae'])/tfloor:+.1f}% vs floor)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation.json").write_text(json.dumps(
        {"backend": __import__("models").BACKEND,
         "validation": val_res, "test": test_res,
         "selected_on_val": best,
         "n": {"train": len(tr), "val": len(va), "test": len(te),
               "train_plots": len({r["plot"] for r in tr}),
               "val_plots": len({r["plot"] for r in va}),
               "test_plots": len({r["plot"] for r in te})},
         "protocol": ("Selection on validation only; test scored once "
                      "afterwards. Bootstrap resamples plots.")},
        indent=2), encoding="utf8")

    with (PROJ / "results/predictions/test_predictions.csv").open(
            "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "plot", "season", "yield"] + list(preds))
        for i, r in enumerate(te):
            w.writerow([r["uid"], f"{r['plot']:.0f}", r["season"],
                        f"{r['yield']:.1f}"]
                       + [f"{preds[v][i]:.2f}" for v in preds])
    print(f"\nwrote {OUT/'ablation.json'} and test predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
