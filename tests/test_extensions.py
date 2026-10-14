"""Experiments C, D and L keep their labels, their protocol and their sums.

These do not re-derive the results (the scripts do that); they pin the
properties a reader relies on: the official data is internally consistent,
the simulated outputs say they are simulated, the protocol was frozen and
is the one the scripts executed, and the arithmetic identities that the
prose leans on hold exactly.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
M = PROJ / "results" / "metrics"
OFF = PROJ / "data" / "official"


def j(name):
    p = M / name
    if not p.exists():
        pytest.skip(f"run the script that writes {name} first")
    return json.loads(p.read_text(encoding="utf8"))


# --------------------------------------------------------------- protocol
def test_protocol_is_frozen_and_discloses_what_was_seen():
    p = yaml.safe_load((PROJ / "configs/extension_protocol.yaml")
                       .read_text(encoding="utf8"))
    assert p["frozen_before_results"] is True
    assert p["experiment_c"]["label"] == "SIMULATED"
    assert p["experiment_d"]["label"] == "SIMULATED"
    assert p["experiment_l"]["label"] == "OBSERVED"
    assert p["seen_before_freezing"]["C"] == "nothing"
    assert p["seen_before_freezing"]["D"] == "nothing"
    assert "season means" in p["seen_before_freezing"]["L"]


@pytest.mark.parametrize("name", ["inspection_sim.json", "yestech_whatif.json",
                                  "official_records.json"])
def test_artefacts_record_the_protocol_they_ran(name):
    r = j(name)
    assert r["protocol_version"] == 1
    assert r["protocol_frozen"] == "2026-10-12"


# ------------------------------------------------------------ official data
def test_official_rice_rows_are_internally_consistent():
    """Every transcribed row must satisfy yield = production / area.

    The state rounds to whole kg/ha and some districts carry the state
    average where they grew almost nothing, so a cell is checked only where
    area is at least 1,000 ha, to within 1 %.
    """
    seasons = ("kuruvai", "samba_thaladi", "navarai", "total")
    n = 0
    with (OFF / "tn_rice_district_season_2019_20.csv").open(encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            for s in seasons:
                a, p, y = (r[f"area_{s}_ha"], r[f"prod_{s}_t"],
                           r[f"yield_{s}_kg_ha"])
                if not a or not p or not y or float(a) < 1000:
                    continue
                calc = 1000.0 * float(p) / float(a)
                assert abs(calc - float(y)) / float(y) < 0.01, (
                    f"{r['district']} {s}: {calc:.0f} vs {y}")
                n += 1
    assert n > 60


def test_official_sources_state_unit_and_conversion():
    s = yaml.safe_load((OFF / "SOURCES.yaml").read_text(encoding="utf8"))
    assert "RICE" in s["files"]["tn_rice_district_season_2019_20.csv"]["units"]["yield"]
    assert "2/3" in s["conversion"]["rice_from_paddy"]
    assert set(s["delta_districts"]["core"]) == {
        "Thanjavur", "Nagapattinam", "Mayiladuthurai", "Thiruvarur"}


def test_core_delta_districts_all_received_relief_in_2020_21():
    r = j("official_records.json")
    got = set(r["l2_event"]["relief"]["districts_with_relief_2020_21"])
    assert got == {"Thanjavur", "Nagapattinam", "Mayiladuthurai", "Thiruvarur"}
    tot = r["l2_event"]["relief"]["total_lakh"]
    k20 = next(k for k in tot if k.startswith("2020-21"))
    k19 = next(k for k in tot if k.startswith("2019-20"))
    assert tot[k20] > 100 * tot[k19]


def test_every_damage_season_crop_was_standing_in_january_2021():
    e = j("official_records.json")["l2_event"]["exposure"]
    assert e["n_standing_in_january_2021"] == e["n_samba_2020_21"] > 100


def test_unit_conversion_is_the_official_one():
    r = j("official_records.json")
    assert r["conversion_rice_from_paddy"] == pytest.approx(2 / 3)
    cell = r["l1_level"]["cells"]["Samba/Thaladi/Pishanam 2019-20"]
    paddy = cell["sickle_mean_paddy_kg_ha"]
    rice = cell["by_conversion"]["2/3 (official)"]["sickle_mean_rice_eq_kg_ha"]
    assert rice == pytest.approx(paddy * 2 / 3)


# ---------------------------------------------------------- experiment C
def test_inspector_identities_hold_exactly():
    r = j("inspection_sim.json")
    for st in r["sets"].values():
        for b, row in st["budgets"].items():
            k = row["k"]
            for key, cell in row["grid"].items():
                s = float(key.split("|")[0][2:])
                f = float(key.split("|")[1][2:])
                for pol in ("register", "history_only", "workflow"):
                    found = row[pol]["n_found"]
                    assert cell[pol]["confirmed_real"] == pytest.approx(s * found)
                    assert cell[pol]["confirmed_false"] == pytest.approx(f * (k - found))
            be = row["register"]["break_even_sensitivity"]
            assert be * row["register"]["n_found"] == pytest.approx(
                row["random"]["mean_found"])


def test_perfect_inspector_reproduces_experiment_a():
    c = j("inspection_sim.json")["sets"]["held-out test split (VERDICT)"]
    a = j("verification_policies.json")["experiment_a"]["held_out"]
    for b in ("0.1", "0.2", "0.33"):
        for pol in ("register", "history_only", "workflow"):
            assert c["budgets"][b][pol]["n_found"] == a["budgets"][b][pol]["n_found"]


def test_exposure_is_labelled_observed_and_counts_are_coherent():
    r = j("inspection_sim.json")
    assert "OBSERVED" in r["label"] and "SIMULATED" in r["label"]
    for st in r["observed_exposure"].values():
        for row in st["budgets"].values():
            assert row["n_missed_estimate_at_or_above_threshold"] <= row["n_missed"]
            assert row["n_missed_cold_start"] <= row["n_missed"]
            assert len(row["missed_estimates_kg_per_acre"]) == row["n_missed"]


# ---------------------------------------------------------- experiment D
def test_yestech_whatif_declares_itself_simulated():
    r = j("yestech_whatif.json")
    assert r["label"] == "SIMULATED"
    assert "says nothing about YES-TECH" in r["verdict"]["qualifier"]


def test_more_error_never_helps_a_short_unit_at_full_weight():
    """At 100 % model weight, P(paid nothing) for a unit truly 10 % short
    must rise with the technology error. A Monte Carlo slip that broke this
    would put a false ordering into the prose."""
    c = j("yestech_whatif.json")["conditional_curves"]
    p = [c[f"w=1.00|sd_t={s:.2f}|bias=+0.00|sd_c=0.10"]["0.10"]["p_paid_nothing"]
         for s in (0.10, 0.20, 0.30)]
    assert p[0] < p[1] < p[2]


def test_full_weight_is_worse_than_minimum_weight_at_sourced_error():
    c = j("yestech_whatif.json")["conditional_curves"]
    for s in (0.20, 0.30):
        full = c[f"w=1.00|sd_t={s:.2f}|bias=+0.00|sd_c=0.10"]["0.10"]["p_paid_nothing"]
        mini = c[f"w=0.30|sd_t={s:.2f}|bias=+0.00|sd_c=0.10"]["0.10"]["p_paid_nothing"]
        assert full > mini


def test_lowest_first_recheck_can_recover_no_unpaid_unit():
    """Rechecking the lowest estimates visits units already being paid, so
    it cannot recover a wrongly unpaid one. If this ever fails the guard
    simulation has a bookkeeping error."""
    g = j("yestech_whatif.json")["guard"]
    n_checked = 0
    for cell in g.values():
        n_paid = (cell["n_due"] - cell["n_wrongly_unpaid"]
                  + cell["n_wrongly_paid"])
        for row in cell["budgets"].values():
            # once the budget exceeds the paid units, lowest-first spills
            # into unpaid ones and may legitimately recover some
            if row["k"] <= n_paid:
                assert row["policies"]["lowest_estimate"]["recovered"] == 0
                n_checked += 1
    assert n_checked > 100


def test_near_threshold_recheck_never_unpays_a_due_unit():
    g = j("yestech_whatif.json")["guard"]
    for cell in g.values():
        for row in cell["budgets"].values():
            assert row["policies"]["near_threshold"]["due_units_unpaid_by_recheck"] == 0
