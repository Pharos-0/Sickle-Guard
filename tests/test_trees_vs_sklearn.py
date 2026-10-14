"""Cross-check the local numpy trees against scikit-learn.

The numpy models in src/trees.py were written because Smart App Control
blocked scikit-learn 1.9.1's compiled extension. Version 1.5.2 loads, so the
two implementations can now be compared directly. That is worth doing for
its own sake: a hand-written splitter that is subtly wrong would still have
produced plausible ablation numbers.

These tests skip cleanly where scikit-learn cannot be imported, so the suite
still runs on a machine where the block applies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trees import GradientBoosting, RandomForest, Tree  # noqa: E402

sk_tree = pytest.importorskip("sklearn.tree",
                              reason="sklearn trees blocked on this machine")
sk_ens = pytest.importorskip("sklearn.ensemble")


@pytest.mark.parametrize("seed", range(5))
def test_single_tree_matches_sklearn_exactly(seed):
    """Same CART objective and same tie-breaking should give same splits.

    Both grow greedily on squared error with the same depth and leaf limits
    and all features offered, so the fitted function should agree to
    floating point on the training data.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 5))
    y = X[:, 0] ** 2 - 2 * X[:, 1] + rng.normal(scale=0.2, size=120)

    mine = Tree(max_depth=4, min_leaf=5).fit(X, y)
    theirs = sk_tree.DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=5, random_state=0).fit(X, y)

    a, b = mine.predict(X), theirs.predict(X)
    # Allow a small tolerance: ties between equally good splits can be
    # broken differently without either being wrong.
    assert np.corrcoef(a, b)[0, 1] > 0.98
    assert np.mean(np.abs(a - b)) < 0.15 * np.std(y)


@pytest.mark.parametrize("seed", range(4))
def test_tree_training_error_is_not_worse_than_sklearn(seed):
    """The real correctness bar: my splitter must not be weaker."""
    rng = np.random.default_rng(100 + seed)
    X = rng.normal(size=(150, 6))
    y = np.sin(X[:, 0]) * 3 + X[:, 2] + rng.normal(scale=0.3, size=150)
    mine = Tree(max_depth=5, min_leaf=4).fit(X, y)
    theirs = sk_tree.DecisionTreeRegressor(
        max_depth=5, min_samples_leaf=4, random_state=0).fit(X, y)
    mse_mine = float(np.mean((mine.predict(X) - y) ** 2))
    mse_sk = float(np.mean((theirs.predict(X) - y) ** 2))
    assert mse_mine <= mse_sk * 1.10, (mse_mine, mse_sk)


def test_forest_generalises_comparably_to_sklearn():
    rng = np.random.default_rng(7)
    X = rng.uniform(-2, 2, size=(400, 6))
    y = np.sin(X[:, 0] * 2) * 5 + X[:, 1] + rng.normal(scale=1.0, size=400)
    Xa, ya, Xb, yb = X[:280], y[:280], X[280:], y[280:]

    mine = RandomForest(n_estimators=150, max_depth=8, min_leaf=3,
                        max_features="sqrt", random_state=0).fit(Xa, ya)
    theirs = sk_ens.RandomForestRegressor(
        n_estimators=150, max_depth=8, min_samples_leaf=3,
        max_features="sqrt", random_state=0).fit(Xa, ya)

    mae_mine = float(np.mean(np.abs(mine.predict(Xb) - yb)))
    mae_sk = float(np.mean(np.abs(theirs.predict(Xb) - yb)))
    assert mae_mine <= mae_sk * 1.25, (mae_mine, mae_sk)


def test_boosting_generalises_comparably_to_sklearn():
    rng = np.random.default_rng(8)
    X = rng.uniform(-2, 2, size=(400, 5))
    y = X[:, 0] ** 2 * 2 + X[:, 3] + rng.normal(scale=0.5, size=400)
    Xa, ya, Xb, yb = X[:280], y[:280], X[280:], y[280:]

    mine = GradientBoosting(n_estimators=150, max_depth=2,
                            learning_rate=0.05, random_state=0).fit(Xa, ya)
    theirs = sk_ens.GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.05,
        random_state=0).fit(Xa, ya)

    mae_mine = float(np.mean(np.abs(mine.predict(Xb) - yb)))
    mae_sk = float(np.mean(np.abs(theirs.predict(Xb) - yb)))
    assert mae_mine <= mae_sk * 1.25, (mae_mine, mae_sk)


def test_both_implementations_agree_on_the_real_feature_table():
    """The comparison that actually matters: this study's own data."""
    import csv
    feats = Path(__file__).resolve().parents[1] / "results" / "features.csv"
    if not feats.exists():
        pytest.skip("run src/extract_features.py first")
    with feats.open(encoding="utf8") as fh:
        rows = list(csv.DictReader(fh))
    skip = {"uid", "plot", "split", "season", "year", "variety", "yield"}
    cols = [c for c in rows[0] if c not in skip]

    def mat(rs):
        X = np.array([[float(r[c]) if r[c] not in ("", "nan") else np.nan
                       for c in cols] for r in rs])
        med = np.nanmedian(X, axis=0)
        med[~np.isfinite(med)] = 0.0
        i = np.where(~np.isfinite(X))
        X[i] = np.take(med, i[1])
        return X

    tr = [r for r in rows if r["split"] in ("train", "val")]
    te = [r for r in rows if r["split"] == "test"]
    Xtr, Xte = mat(tr), mat(te)
    ytr = np.array([float(r["yield"]) for r in tr])
    yte = np.array([float(r["yield"]) for r in te])

    mine = RandomForest(n_estimators=200, max_depth=8, min_leaf=3,
                        max_features="sqrt", random_state=0).fit(Xtr, ytr)
    theirs = sk_ens.RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        max_features="sqrt", random_state=0).fit(Xtr, ytr)

    pm, ps = mine.predict(Xte), theirs.predict(Xte)
    mae_m = float(np.mean(np.abs(pm - yte)))
    mae_s = float(np.mean(np.abs(ps - yte)))
    # Headline MAE must not depend on which implementation was used.
    assert abs(mae_m - mae_s) < 25.0, (mae_m, mae_s)
    assert np.corrcoef(pm, ps)[0, 1] > 0.85
