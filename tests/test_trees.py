"""Correctness tests for the numpy tree models.

These exist because the models replace scikit-learn, which is blocked on this
machine. A hand-written splitter that is subtly wrong would still produce
plausible ablation numbers, so the split finder is checked against brute
force and each model is checked on a signal a linear model provably cannot
represent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trees import GradientBoosting, RandomForest, Tree, _best_split  # noqa: E402


def brute_force_split(X, y, min_leaf):
    """Exhaustive search for the squared-error-minimising split."""
    n, p = X.shape
    best = None
    for j in range(p):
        for thr in np.unique(X[:, j]):
            m = X[:, j] <= thr
            if m.sum() < min_leaf or (~m).sum() < min_leaf:
                continue
            sse = (((y[m] - y[m].mean()) ** 2).sum()
                   + ((y[~m] - y[~m].mean()) ** 2).sum())
            if best is None or sse < best[1] - 1e-12:
                best = (j, sse)
    return best


@pytest.mark.parametrize("seed", range(6))
def test_split_finder_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] > 0) * 3.0 + rng.normal(scale=0.3, size=40)
    got = _best_split(X, y, np.arange(4), min_leaf=3)
    want = brute_force_split(X, y, min_leaf=3)
    assert got is not None and want is not None
    assert got[0] == want[0], "chose a different feature than brute force"
    assert got[2] == pytest.approx(want[1], rel=1e-9)


def test_split_is_never_placed_between_identical_values():
    """A cut inside a run of equal values is not a real partition."""
    X = np.array([[1.0]] * 10 + [[2.0]] * 10)
    y = np.array([0.0] * 10 + [5.0] * 10)
    j, thr, _ = _best_split(X, y, np.arange(1), min_leaf=2)
    assert 1.0 < thr < 2.0


def test_tree_respects_min_leaf():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)
    t = Tree(max_depth=10, min_leaf=7).fit(X, y)

    def walk(node):
        if node["leaf"]:
            assert node["n"] >= 1
            return
        walk(node["L"]); walk(node["R"])
    walk(t.root)


def test_tree_learns_a_step_a_linear_model_cannot():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(200, 1))
    y = np.where(X[:, 0] > 0, 10.0, -10.0)
    t = Tree(max_depth=3, min_leaf=5).fit(X, y)
    assert np.mean(np.abs(t.predict(X) - y)) < 0.5


def test_constant_target_gives_a_single_leaf():
    X = np.random.default_rng(0).normal(size=(30, 2))
    t = Tree().fit(X, np.full(30, 4.2))
    assert t.root["leaf"]
    assert t.predict(X) == pytest.approx(np.full(30, 4.2))


def test_bagging_beats_a_single_tree_when_all_features_are_offered():
    """Averaging bootstrapped trees should reduce held-out error.

    max_features is set to None here deliberately. With sqrt subsampling on
    five columns of which only one is informative, most splits never see the
    useful column, and the forest can legitimately lose to a single tree --
    a property of the method, not a defect, so the test does not assert it.
    """
    rng = np.random.default_rng(2)
    X = rng.uniform(-2, 2, size=(300, 5))
    y = np.sin(X[:, 0] * 2) * 5 + rng.normal(scale=1.0, size=300)
    Xa, ya, Xb, yb = X[:200], y[:200], X[200:], y[200:]
    one = Tree(max_depth=8, min_leaf=3).fit(Xa, ya)
    many = RandomForest(n_estimators=80, max_depth=8, max_features=None,
                        random_state=0).fit(Xa, ya)
    assert (np.mean(np.abs(many.predict(Xb) - yb))
            < np.mean(np.abs(one.predict(Xb) - yb)))


def test_forest_predictions_are_more_stable_than_a_single_tree():
    """Variance reduction is what bagging actually guarantees."""
    rng = np.random.default_rng(11)
    X = rng.uniform(-2, 2, size=(160, 4))
    y = np.sin(X[:, 0] * 2) * 5 + rng.normal(scale=1.0, size=160)
    Xq = rng.uniform(-2, 2, size=(50, 4))

    def spread(make):
        P = []
        for s in range(5):
            idx = np.random.default_rng(s).integers(0, len(y), len(y))
            P.append(make(s).fit(X[idx], y[idx]).predict(Xq))
        return float(np.mean(np.std(np.vstack(P), axis=0)))

    tree_spread = spread(lambda s: Tree(
        max_depth=8, min_leaf=3, rng=np.random.default_rng(s)))
    forest_spread = spread(lambda s: RandomForest(
        n_estimators=40, max_depth=8, max_features=None, random_state=s))
    assert forest_spread < tree_spread


def test_forest_is_deterministic_given_a_seed():
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(80, 4)), rng.normal(size=80)
    a = RandomForest(n_estimators=20, random_state=7).fit(X, y).predict(X)
    b = RandomForest(n_estimators=20, random_state=7).fit(X, y).predict(X)
    assert a == pytest.approx(b)


def test_boosting_reduces_training_error_monotonically_early_on():
    rng = np.random.default_rng(4)
    X = rng.uniform(-2, 2, size=(200, 3))
    y = X[:, 0] ** 2 + rng.normal(scale=0.2, size=200)
    errs = []
    for n in (1, 10, 60):
        m = GradientBoosting(n_estimators=n, max_depth=2,
                             learning_rate=0.1, random_state=0).fit(X, y)
        errs.append(np.mean((m.predict(X) - y) ** 2))
    assert errs[0] > errs[1] > errs[2]


def test_boosting_recovers_a_quadratic_better_than_the_mean():
    rng = np.random.default_rng(5)
    X = rng.uniform(-2, 2, size=(300, 2))
    y = X[:, 0] ** 2 * 3
    Xa, ya, Xb, yb = X[:200], y[:200], X[200:], y[200:]
    m = GradientBoosting(n_estimators=150, max_depth=2,
                         learning_rate=0.1, random_state=0).fit(Xa, ya)
    assert (np.mean(np.abs(m.predict(Xb) - yb))
            < 0.4 * np.mean(np.abs(ya.mean() - yb)))


def test_models_ignore_a_pure_noise_feature():
    """Adding noise columns must not wreck a clean signal."""
    rng = np.random.default_rng(6)
    X = rng.uniform(-1, 1, size=(300, 1))
    y = np.where(X[:, 0] > 0, 5.0, -5.0)
    Xn = np.hstack([X, rng.normal(size=(300, 6))])
    m = RandomForest(n_estimators=80, max_depth=6, random_state=0).fit(Xn, y)
    assert np.mean(np.abs(m.predict(Xn) - y)) < 1.5
