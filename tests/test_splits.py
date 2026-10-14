"""Integrity tests for the frozen plot-grouped split and the feature table.

These pin the properties whose violation would silently inflate every result
in the study: a plot appearing in two partitions, the target leaking into a
feature, or a season being scored against the wrong plot.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

PROJ = Path(__file__).resolve().parents[1]
SPLITS = PROJ / "configs" / "splits.json"
FEATS = PROJ / "results" / "features.csv"


@pytest.fixture(scope="module")
def split():
    if not SPLITS.exists():
        pytest.skip("run src/make_splits.py first")
    return json.loads(SPLITS.read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def feats():
    if not FEATS.exists():
        pytest.skip("run src/extract_features.py first")
    with FEATS.open(encoding="utf8") as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------- the split
def test_no_plot_crosses_partitions(split):
    """The single property the whole evaluation rests on."""
    tr = set(split["plots"]["train"])
    va = set(split["plots"]["val"])
    te = set(split["plots"]["test"])
    assert not tr & va
    assert not tr & te
    assert not va & te


def test_every_plot_assigned_exactly_once(split):
    n = sum(len(split["plots"][s]) for s in ("train", "val", "test"))
    assert n == split["n_plots"]


def test_no_field_season_duplicated_across_splits(split):
    uids = [u for s in ("train", "val", "test")
            for u in split["field_seasons"][s]]
    assert len(uids) == len(set(uids))
    assert len(uids) == split["n_field_seasons"]


def test_test_partition_is_large_enough_to_report(split):
    """A test set below ~20 plots cannot support the planned intervals."""
    assert len(split["plots"]["test"]) >= 20


def test_split_is_deterministic(split):
    assert isinstance(split["seed"], int)
    assert split["grouping_key"] == "PLOT_ID"


# -------------------------------------------------- the feature table
def test_feature_rows_match_the_frozen_split(feats, split):
    assign = {u: s for s in ("train", "val", "test")
              for u in split["field_seasons"][s]}
    for r in feats:
        assert r["split"] == assign[r["uid"]], r["uid"]


def test_each_field_season_appears_once(feats):
    uids = [r["uid"] for r in feats]
    assert len(uids) == len(set(uids))


def test_a_plots_seasons_all_sit_in_one_split(feats):
    by_plot: dict[str, set] = {}
    for r in feats:
        by_plot.setdefault(r["plot"], set()).add(r["split"])
    bad = {p: s for p, s in by_plot.items() if len(s) > 1}
    assert not bad, f"plots spanning splits: {bad}"


def test_target_is_a_plausible_per_acre_density(feats):
    """YIELD is kg/acre, not a field total -- see DATA_AUDIT.md section 2.1."""
    y = np.array([float(r["yield"]) for r in feats])
    assert y.min() >= 500 and y.max() <= 5000
    area = np.array([float(r["area"]) for r in feats if r["area"]])
    yy = np.array([float(r["yield"]) for r in feats if r["area"]])
    # A total would scale with area; a density should not.
    assert abs(np.corrcoef(area, yy)[0, 1]) < 0.6


def test_no_feature_is_a_transform_of_the_target(feats):
    y = np.array([float(r["yield"]) for r in feats])
    skip = {"uid", "plot", "split", "season", "year", "variety", "yield"}
    for c in feats[0]:
        if c in skip:
            continue
        v = np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan
                      for r in feats])
        ok = np.isfinite(v)
        if ok.sum() < 10 or np.std(v[ok]) == 0:
            continue
        assert abs(np.corrcoef(v[ok], y[ok])[0, 1]) < 0.99, c


def test_every_sample_has_some_usable_optical_data(feats):
    n = np.array([int(r["s2_n_clear"]) for r in feats])
    assert (n > 0).all(), f"{int((n == 0).sum())} samples lost all S2 to cloud"


def test_plot_pixel_counts_are_small_as_documented(feats):
    """The binding physical constraint: a plot is ~15 Sentinel-2 pixels."""
    px = np.array([int(r["n_px"]) for r in feats])
    assert px.min() >= 1
    assert np.median(px) < 120, "patches are larger than the audit found"
