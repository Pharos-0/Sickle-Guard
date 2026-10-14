"""Model factory: scikit-learn where it loads, numpy fallback where it does not.

Windows Smart App Control blocks scikit-learn 1.9.1's compiled tree
extension on this machine; 1.5.2 loads. Rather than depend on which version
happens to be installed, this module tries scikit-learn and falls back to
the local numpy implementations in src/trees.py.

Both paths are exercised: tests/test_trees_vs_sklearn.py checks that the two
agree on this study's own feature table to within 25 kg/acre MAE, so the
reported results do not depend on which one ran. `BACKEND` records which was
used and every result file carries it.
"""

from __future__ import annotations

try:                                              # pragma: no cover
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  RandomForestRegressor)
    BACKEND = "scikit-learn"
except Exception:                                 # pragma: no cover
    BACKEND = "numpy-fallback"


def make_rf(seed: int):
    if BACKEND == "scikit-learn":
        return RandomForestRegressor(n_estimators=300, max_depth=8,
                                     min_samples_leaf=3, max_features="sqrt",
                                     random_state=seed, n_jobs=-1)
    from trees import RandomForest
    return RandomForest(n_estimators=200, max_depth=8, min_leaf=3,
                        max_features="sqrt", random_state=seed)


def make_gb(seed: int):
    if BACKEND == "scikit-learn":
        return GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                         learning_rate=0.05,
                                         random_state=seed)
    from trees import GradientBoosting
    return GradientBoosting(n_estimators=150, max_depth=2,
                            learning_rate=0.05, random_state=seed)


def tree_predictions(fitted_rf, X):
    """Per-tree predictions, for the ensemble-disagreement signal."""
    import numpy as np
    if BACKEND == "scikit-learn":
        return np.vstack([t.predict(X) for t in fitted_rf.estimators_])
    return np.vstack([t.predict(X) for t in fitted_rf.trees])
