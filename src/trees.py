"""Regression trees, a bagged forest and gradient boosting, in numpy only.

WHY THIS EXISTS
---------------
scikit-learn installs fine on this machine but Windows Smart App Control
blocks its freshly written compiled extension (`sklearn.tree._criterion`),
the same policy that blocks a fresh torch install here. Rather than drop the
non-linear variants from the ablation -- which would leave the study unable
to say whether a tree model beats ridge -- they are implemented here in pure
Python and numpy, so there is no compiled artefact to block.

These are deliberately small and standard: CART with squared-error splits,
bootstrap aggregation with feature subsampling, and gradient boosting on
squared-error residuals. They are not meant to beat a tuned library
implementation. They are meant to answer "does a non-linear model help on
197 training rows", which does not need a fast implementation.

Correctness is pinned by tests/test_trees.py, which checks the split finder
against brute force and checks that each model recovers signals a linear
model cannot.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- CART
def _best_split(X, y, feat_idx, min_leaf):
    """Squared-error split, found with prefix sums over each sorted feature.

    Returns (feature, threshold, sse) or None when no split beats the parent
    while leaving min_leaf samples on both sides.
    """
    n = len(y)
    if n < 2 * min_leaf:
        return None
    best = None
    for j in feat_idx:
        order = np.argsort(X[:, j], kind="mergesort")
        xs, ys = X[order, j], y[order]
        csum = np.cumsum(ys)
        csq = np.cumsum(ys ** 2)
        tot, tot_sq = csum[-1], csq[-1]
        # candidate cut after position i (0-based), left = i+1 samples
        lo, hi = min_leaf - 1, n - min_leaf
        if hi <= lo:
            continue
        i = np.arange(lo, hi)
        nl = i + 1.0
        nr = n - nl
        sl, sr = csum[i], tot - csum[i]
        ql, qr = csq[i], tot_sq - csq[i]
        sse = (ql - sl ** 2 / nl) + (qr - sr ** 2 / nr)
        # a cut is only valid between two distinct feature values
        valid = xs[i + 1] > xs[i]
        if not valid.any():
            continue
        sse = np.where(valid, sse, np.inf)
        k = int(np.argmin(sse))
        if best is None or sse[k] < best[2]:
            thr = 0.5 * (xs[i[k]] + xs[i[k] + 1])
            best = (int(j), float(thr), float(sse[k]))
    return best


class Tree:
    """CART regression tree with squared-error splits."""

    def __init__(self, max_depth=6, min_leaf=3, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(0)
        self.root = None

    def fit(self, X, y):
        self.root = self._build(np.asarray(X, float), np.asarray(y, float), 0)
        return self

    def _build(self, X, y, depth):
        node = {"leaf": True, "value": float(y.mean()), "n": len(y)}
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf \
                or np.allclose(y, y[0]):
            return node
        p = X.shape[1]
        k = p if self.max_features is None else max(1, int(self.max_features))
        feat = (np.arange(p) if k >= p
                else self.rng.choice(p, size=k, replace=False))
        sp = _best_split(X, y, feat, self.min_leaf)
        if sp is None:
            return node
        j, thr, sse = sp
        parent_sse = float(((y - y.mean()) ** 2).sum())
        if sse >= parent_sse - 1e-12:
            return node
        m = X[:, j] <= thr
        if m.sum() < self.min_leaf or (~m).sum() < self.min_leaf:
            return node
        return {"leaf": False, "j": j, "thr": thr,
                "L": self._build(X[m], y[m], depth + 1),
                "R": self._build(X[~m], y[~m], depth + 1)}

    def predict(self, X):
        X = np.asarray(X, float)
        return np.array([self._one(x, self.root) for x in X])

    def _one(self, x, node):
        while not node["leaf"]:
            node = node["L"] if x[node["j"]] <= node["thr"] else node["R"]
        return node["value"]


# ------------------------------------------------------- random forest
class RandomForest:
    """Bootstrap aggregation of CART trees with feature subsampling."""

    def __init__(self, n_estimators=200, max_depth=8, min_leaf=3,
                 max_features="sqrt", random_state=0):
        self.n = n_estimators
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.seed = random_state
        self.trees: list[Tree] = []

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        p = X.shape[1]
        mf = (int(np.sqrt(p)) if self.max_features == "sqrt"
              else p if self.max_features is None else int(self.max_features))
        self.trees = []
        for _ in range(self.n):
            idx = rng.integers(0, len(y), len(y))
            t = Tree(self.max_depth, self.min_leaf, mf,
                     np.random.default_rng(rng.integers(1 << 31)))
            self.trees.append(t.fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees], axis=0)


# --------------------------------------------------- gradient boosting
class GradientBoosting:
    """Squared-error gradient boosting on shallow trees."""

    def __init__(self, n_estimators=200, max_depth=2, learning_rate=0.05,
                 min_leaf=3, random_state=0):
        self.n = n_estimators
        self.max_depth = max_depth
        self.lr = learning_rate
        self.min_leaf = min_leaf
        self.seed = random_state
        self.base = 0.0
        self.trees: list[Tree] = []

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        self.base = float(y.mean())
        resid = y - self.base
        self.trees = []
        for _ in range(self.n):
            t = Tree(self.max_depth, self.min_leaf, None,
                     np.random.default_rng(rng.integers(1 << 31)))
            t.fit(X, resid)
            resid = resid - self.lr * t.predict(X)
            self.trees.append(t)
        return self

    def predict(self, X):
        out = np.full(len(X), self.base)
        for t in self.trees:
            out += self.lr * t.predict(X)
        return out
