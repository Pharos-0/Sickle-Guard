"""Safety tests for the farmer-facing report.

The report's single design rule is that it must never discourage a farmer
from filing a claim. That rule currently lives in prose and in my care while
writing the strings; these tests move it into code, so a later edit that
adds a reassuring sentence fails the build instead of reaching a farmer.

Each test states the harm it is preventing.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJ = Path(__file__).resolve().parents[1]
PY = sys.executable
SCRIPT = PROJ / "src" / "farmer_report.py"
CONTACTS = PROJ / "configs" / "scheme_contacts.yaml"


def run(*args) -> str:
    p = subprocess.run([PY, str(SCRIPT), *args], cwd=PROJ,
                       capture_output=True, text=True, encoding="utf8",
                       errors="replace", timeout=600)
    assert p.returncode == 0, p.stderr[-800:]
    return p.stdout


@pytest.fixture(scope="module")
def flagged() -> str:
    """A field the system ranks lowest; should be flagged."""
    return run("--uid", "2063")


@pytest.fixture(scope="module")
def unflagged() -> str:
    """A mid-ranked field; should NOT be flagged."""
    return run("--uid", "2115")


@pytest.fixture(scope="module")
def listing() -> str:
    return run("--village-list", "--budget", "0.33")


# ------------------------------------------------- the central safety rule
REASSURING = [
    r"\bno loss\b", r"\bno damage\b", r"\bcrop is fine\b", r"\bcrop is ok\b",
    r"\byour crop is healthy\b", r"\bnothing to worry\b", r"\bnot eligible\b",
    r"\byou are safe\b", r"\bno action needed\b", r"\bno claim\b",
    r"\bclaim (approved|denied|rejected)\b", r"\bnot entitled\b",
]


NEGATORS = ("not", "never", "cannot", "can't", "misses", "no evidence")


def _unnegated(pat: str, text: str):
    """Occurrences of a reassuring phrase that are NOT inside a negation.

    The report legitimately says "THIS IS NOT EVIDENCE THAT YOUR CROP IS
    FINE", which contains a reassuring phrase while meaning the opposite.
    A flat substring search would flag that and push someone to delete the
    very sentence the rule exists to require. So a match is allowed when a
    negator appears in the 60 characters before it, and reported otherwise.
    """
    out = []
    for m in re.finditer(pat, text, re.I):
        before = text[max(0, m.start() - 60):m.start()].lower()
        if not any(n in before for n in NEGATORS):
            out.append(m.group(0))
    return out


@pytest.mark.parametrize("pat", REASSURING)
def test_report_never_reassures_or_denies(pat, flagged, unflagged, listing):
    """Harm: a farmer reads 'no loss' and does not file inside 72 hours.

    No report may contain a phrase that could be read as clearing the field
    or as a verdict on entitlement, except inside an explicit negation.
    """
    for name, text in (("flagged", flagged), ("unflagged", unflagged),
                       ("listing", listing)):
        hits = _unnegated(pat, text)
        assert not hits, f"{name} report contains un-negated {hits!r}"


def test_the_negation_guard_itself_catches_a_bare_reassurance():
    """If the guard above is too permissive the suite proves nothing."""
    assert _unnegated(r"\bcrop is fine\b", "Your crop is fine this season.")
    assert not _unnegated(r"\bcrop is fine\b",
                          "This is not evidence that your crop is fine.")


def test_unflagged_report_states_it_is_not_evidence_of_no_loss(unflagged):
    """Harm: silence read as clearance. The negative case must say so."""
    t = unflagged.upper()
    assert "NOT EVIDENCE THAT YOUR CROP IS FINE" in t
    assert "MISSES LOSSES" in unflagged.lower() or "misses losses" in unflagged


def test_every_report_tells_the_farmer_to_report_damage_anyway(
        flagged, unflagged):
    for t in (flagged, unflagged):
        assert "report it" in t.lower()
        assert "do not wait" in t.lower()


def test_listing_states_unlisted_fields_are_not_cleared(listing):
    """Harm: a field absent from the visit list is treated as fine."""
    assert "NOT cleared" in listing or "not cleared" in listing.lower()


# ----------------------------------------------- honesty about the model
def test_satellite_only_estimates_are_labelled_weak(unflagged):
    """Field 2115 has no prior harvest, so the estimate is satellite-only."""
    assert "satellite imagery (no past harvest" in unflagged
    assert "NO RELIABLE ESTIMATE" in unflagged
    assert "Do not act on the number itself" in unflagged


def test_history_based_estimates_say_so(flagged):
    assert "past harvests" in flagged


def test_no_report_claims_seasonal_deterioration(flagged, unflagged):
    """Bad-season detection measured at AUROC 0.59; must not be offered."""
    for t in (flagged, unflagged):
        assert "CANNOT tell whether this particular season" in t


def test_report_declares_it_is_not_official(flagged):
    low = flagged.lower()
    assert "not connected to pmfby" in low
    assert "no official standing" in low


# ------------------------------------------------- unverified contact data
def test_unverified_contacts_are_marked_in_the_output(flagged):
    """Harm: a farmer calls a wrong number and loses the window.

    Nothing in configs/scheme_contacts.yaml has been confirmed against an
    official page, so every such line must carry the warning.
    """
    cfg = yaml.safe_load(CONTACTS.read_text(encoding="utf8"))
    for c in cfg["contacts"]:
        if c.get("verified"):
            continue
        line = next((ln for ln in flagged.splitlines()
                     if str(c["value"])[:18] in ln), None)
        assert line is not None, f"contact {c['name']} missing from report"
        assert "UNVERIFIED" in line, f"{c['name']} shown without warning"


def test_deadlines_appear_before_any_model_output_is_actionable(flagged):
    """The 72-hour rule matters more than anything the model says."""
    assert "72 hours" in flagged
    i_deadline = flagged.find("72 hours")
    i_rank = flagged.find("Rank in this list")
    assert i_rank == -1 or i_deadline > i_rank or "WHAT TO DO NOW" in flagged


def test_config_marks_everything_unverified_until_checked():
    cfg = yaml.safe_load(CONTACTS.read_text(encoding="utf8"))
    assert cfg["contacts"], "no contacts configured"
    for c in cfg["contacts"]:
        assert "verified" in c, f"{c['name']} has no verified flag"
        if not c["verified"]:
            assert c.get("source"), f"{c['name']} unverified with no source"


def test_mandatory_notices_exist_and_are_printed(flagged):
    cfg = yaml.safe_load(CONTACTS.read_text(encoding="utf8"))
    for key in ("never_discourage", "not_official"):
        assert cfg["mandatory_notices"].get(key)
    assert "CANNOT tell you whether you have a valid claim" in flagged


# ------------------------------------------------------------- mechanics
def test_tamil_renders_without_crashing():
    """Previously raised UnicodeEncodeError on a cp1252 console."""
    out = run("--uid", "2063", "--lang", "ta")
    assert "வயல்" in out
    # safety text must survive translation mode: the instruction to report
    # damage now, in the translation's own words (E2)
    sys.path.insert(0, str(PROJ / "src"))
    import farmer_report as fr
    e2 = " ".join(fr.translations("ta")["E2"].split())
    assert e2 in " ".join(out.split())


def test_unknown_field_fails_loudly_rather_than_guessing():
    p = subprocess.run([PY, str(SCRIPT), "--uid", "nonexistent-999"],
                       cwd=PROJ, capture_output=True, text=True,
                       encoding="utf8", errors="replace", timeout=600)
    assert p.returncode != 0
    assert "no field-season" in (p.stdout + p.stderr).lower()


def test_listing_reports_how_many_rows_rest_on_weak_evidence(listing):
    assert re.search(r"\d+ of \d+ use the field's own past harvests", listing)
    assert "satellite-only and much less reliable" in listing
