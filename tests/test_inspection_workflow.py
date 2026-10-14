"""The inspection workflow keeps its eight rules.

Each test pins one rule from src/inspection_workflow.py's docstring, and the
tamper tests check the audit chain by attacking it: an audit trail that is
never attacked in a test is only assumed to work.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import inspection_workflow as iw  # noqa: E402


@pytest.fixture(scope="module")
def world():
    feats, D = iw.field_table()
    below = {feats[i]["uid"]: bool(D["bad"][i]) for i in range(len(feats))}
    return feats, D, below


@pytest.fixture()
def con(tmp_path):
    c = iw.connect(tmp_path / "t.sqlite")
    yield c
    c.close()


def tasks_of(con, pid):
    return con.execute("SELECT * FROM tasks WHERE plan_id=? ORDER BY rank",
                       (pid,)).fetchall()


# ------------------------------------------------------- rules 1 and 2: plan
def test_plan_respects_budget_and_reserve(world, con):
    feats, D, _ = world
    pool = [i for i, r in enumerate(feats) if r["split"] == "test"]
    pid = iw.plan(con, feats, D["hist"], "test", 0.33, 0.10, seed=0)
    t = tasks_of(con, pid)
    k = max(1, int(round(0.33 * len(pool))))
    assert len(t) == k
    n_res = sum(1 for x in t if x["reason"] == iw.REASON_RESERVE)
    assert n_res == int(round(0.10 * k))


def test_recorded_fields_are_the_lowest_past_harvests(world, con):
    feats, D, _ = world
    hist = D["hist"]
    pid = iw.plan(con, feats, hist, "test", 0.33, 0.10, seed=0)
    chosen = {x["uid"] for x in tasks_of(con, pid)
              if x["reason"] == iw.REASON_RECORD}
    rec = [i for i, r in enumerate(feats)
           if r["split"] == "test" and np.isfinite(hist[i])]
    worst = sorted(rec, key=lambda i: (hist[i], feats[i]["uid"]))[:len(chosen)]
    assert chosen == {feats[i]["uid"] for i in worst}


def test_reserve_ignores_every_yield_estimate(world, con):
    """Experiment B: satellite ordering did not beat random within cold
    starts, so the planner takes no estimate for those fields at all.
    Scrambling every yield in the history table leaves the reserve picks
    unchanged; only the seed moves them."""
    feats, D, _ = world
    hist = D["hist"].copy()
    a = iw.plan(con, feats, hist, "test", 0.33, 0.25, seed=0)
    scrambled = hist.copy()
    fin = np.isfinite(scrambled)
    scrambled[fin] = np.random.default_rng(9).permutation(scrambled[fin])
    b = iw.plan(con, feats, scrambled, "test", 0.33, 0.25, seed=0)
    c = iw.plan(con, feats, hist, "test", 0.33, 0.25, seed=1)
    res = lambda pid: [x["uid"] for x in tasks_of(con, pid)  # noqa: E731
                       if x["reason"] == iw.REASON_RESERVE]
    assert res(a) == res(b)
    assert res(a) != res(c)
    assert "satellite" not in iw.plan.__code__.co_varnames


# ----------------------------------------------------------- rule 3: blind
def test_task_sheet_is_blind_to_the_estimate(world, con):
    feats, D, _ = world
    pid = iw.plan(con, feats, D["hist"], "test", 0.33, 0.10, seed=0)
    for t in tasks_of(con, pid):
        sheet = iw.task_sheet(con, t["task_id"])
        assert "withheld" in sheet
        assert "kg" not in sheet.lower()
        i = next(j for j, r in enumerate(feats) if r["uid"] == t["uid"])
        if np.isfinite(D["hist"][i]):
            assert f"{D['hist'][i]:.0f}" not in sheet
        assert str(int(feats[i]["yield"])) not in sheet


# ------------------------------------------------- rules 4 and 5: recording
def test_unknown_finding_is_refused(world, con):
    feats, D, _ = world
    iw.plan(con, feats, D["hist"], "test", 0.10, 0.0, seed=0)
    with pytest.raises(iw.WorkflowError):
        iw.record(con, 1, "damage verified", "officer")


def test_problem_without_evidence_is_refused(world, con):
    feats, D, _ = world
    iw.plan(con, feats, D["hist"], "test", 0.10, 0.0, seed=0)
    with pytest.raises(iw.WorkflowError, match="evidence"):
        iw.record(con, 1, iw.POSITIVE, "officer", note="looks bad")
    iw.record(con, 1, iw.POSITIVE, "officer", photos=["IMG_0001.jpg"])


@pytest.mark.parametrize("note", [
    "Claim approved after visit.", "Damage is verified, pay the farmer.",
    "Farmer not eligible.", "Crop is fine, nothing to see.",
    "No loss here at all."])
def test_decision_language_is_refused(world, con, note):
    feats, D, _ = world
    iw.plan(con, feats, D["hist"], "test", 0.10, 0.0, seed=0)
    with pytest.raises(iw.WorkflowError, match="decisions"):
        iw.record(con, 1, "no problem observed", "officer", note=note)


def test_records_are_append_only(world, con):
    feats, D, _ = world
    iw.plan(con, feats, D["hist"], "test", 0.10, 0.0, seed=0)
    iw.record(con, 1, "could not assess", "officer", note="field flooded")
    with pytest.raises(iw.WorkflowError, match="append-only"):
        iw.record(con, 1, "no problem observed", "officer")


# ------------------------------------------------------- rule 6: the chain
def _filled(world, con):
    feats, D, below = world
    iw.plan(con, feats, D["hist"], "test", 0.33, 0.10, seed=0)
    iw.simulate(con, below, 0.8, 0.1, seed=0)
    assert iw.verify(con)["ok"]


def test_editing_an_inspection_breaks_the_chain(world, con):
    _filled(world, con)
    con.execute("UPDATE inspections SET finding='no problem observed' "
                "WHERE finding=?", (iw.POSITIVE,))
    v = iw.verify(con)
    assert not v["ok"]
    assert any("differs from its audit entry" in p for p in v["problems"])


def test_editing_the_log_breaks_the_chain(world, con):
    _filled(world, con)
    con.execute("UPDATE audit SET payload=replace(payload, 'below', 'above') "
                "WHERE seq=3")
    v = iw.verify(con)
    assert not v["ok"]
    assert any("altered" in p for p in v["problems"])


def test_deleting_a_log_entry_breaks_the_chain(world, con):
    _filled(world, con)
    con.execute("DELETE FROM audit WHERE seq=4")
    assert not iw.verify(con)["ok"]


def test_reordering_the_plan_breaks_the_chain(world, con):
    _filled(world, con)
    con.execute("UPDATE tasks SET rank = 999 WHERE rank = 1")
    assert not iw.verify(con)["ok"]


# ------------------------------------------------ rule 7: simulated is said
def test_simulated_inspections_declare_themselves(world, con):
    _filled(world, con)
    s = iw.summary(con)
    assert s["n_simulated"] == s["n_inspected"] > 0
    assert "SIMULATED" in s["label"]
    acts = {r["action"] for r in con.execute("SELECT action FROM audit")}
    assert "record (SIMULATED)" in acts and "record" not in acts
    notes = [r["note"] for r in con.execute("SELECT note FROM inspections")]
    assert all("No field visit took place" in n for n in notes)


def test_perfect_simulated_inspector_confirms_exactly_the_truth(world, con):
    feats, D, below = world
    iw.plan(con, feats, D["hist"], "test", 0.33, 0.10, seed=0)
    iw.simulate(con, below, 1.0, 0.0, seed=0)
    a = iw.summary(con, below)["against_study_threshold"]
    assert a["confirmed_real"] == a["below_threshold_among_tasks"]
    assert a["confirmed_not_below"] == 0


# ------------------------------------------------------ rule 8: not cleared
def test_uninspected_fields_are_not_cleared(world, con):
    _filled(world, con)
    s = iw.summary(con)
    assert s["n_pool_not_inspected"] > 0
    assert "NOT INSPECTED" in s["not_inspected_means"]
    assert "not evidence that the crop is fine" in s["not_inspected_means"]
    text = json.dumps(s).lower()
    assert not re.search(r"\bcleared\b", text)


def test_demo_artefact_is_labelled_and_intact():
    p = iw.DEMO_OUT
    if not p.exists():
        pytest.skip("run src/inspection_workflow.py demo first")
    s = json.loads(p.read_text(encoding="utf8"))
    assert s["audit"]["ok"]
    assert s["inspector_model"]["label"] == "SIMULATED"
    assert "SIMULATED" in s["label"]
    assert "withheld" in s["example_task_sheet"]
