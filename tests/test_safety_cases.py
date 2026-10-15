"""Predefined safety cases: does the report stay safe when the model is wrong?

The existing suites check that report text avoids reassurance in general.
These build specific situations in which a careless report would mislead,
and require the safe behaviour in each:

  1  the model is WRONG: real held-out field-seasons below threshold that
     the workflow did not flag. Their reports must not clear them.
  2  evidence is MISSING: a cold-start field, whatever the satellite says.
  3  INVARIANCE: when evidence is insufficient, the safety message must be
     the same whatever number the model produces. A warning that softens
     as the estimate rises would be reassurance by another route.
  4  PREDICTION versus DAMAGE: no report may present a model output as
     verified crop damage or as a decision on a claim.

These establish a property of the software. They do not establish that a
farmer reading the report understands it or acts on it; that needs people,
and none were involved.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import farmer_report as fr  # noqa: E402

LEVEL = 0.80


def render(r, est, basis, flagged, lang="en", rank=1, total=62):
    cfg = fr.contacts()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fr.report(r, est, basis, 100 * est / BENCH, flagged, cfg, lang,
                  rank, total)
    return buf.getvalue()


@pytest.fixture(scope="module")
def world():
    global BENCH
    feats = fr.load()
    hist = fr.causal_history(feats)
    sat = fr.satellite(feats)
    y = np.array([r["yield"] for r in feats])
    split = np.array([r["split"] for r in feats])
    BENCH = float(y[split == "train"].mean())
    est = np.where(np.isfinite(hist), hist, sat)
    basis = np.where(np.isfinite(hist), "history", "satellite")
    test = np.where(split == "test")[0]
    k = max(1, int(round(0.33 * len(test))))
    order = test[np.argsort(est[test], kind="mergesort")]
    flagged = set(order[:k].tolist())
    bad = y < LEVEL * BENCH
    missed = [i for i in test if bad[i] and i not in flagged]
    cold = [i for i in test if basis[i] == "satellite"]
    return dict(feats=feats, est=est, basis=basis, flagged=flagged,
                missed=missed, cold=cold, bad=bad)


BENCH = 1.0

# phrases that would clear a field or decide a claim
CLEARING = [r"\bcrop is (fine|ok|okay|healthy|safe)\b", r"\bno (crop )?loss\b",
            r"\bnot (eligible|entitled)\b", r"\bno action (is )?(needed|required)\b",
            r"\bnothing to worry\b", r"\byou are covered\b",
            r"\bdamage (is )?(confirmed|verified)\b",
            r"\bloss (is )?(confirmed|verified|detected)\b",
            r"\bclaim (is )?(approved|rejected|valid|invalid)\b"]
NEGATORS = ("not ", "never ", "cannot ", "does not ", "do not ", "is not ",
            "no evidence", "rather than", "without ")


def unnegated(pat, text):
    hits = []
    for m in re.finditer(pat, text, re.I):
        window = text[max(0, m.start() - 90):m.start()].lower()
        if not any(n in window for n in NEGATORS):
            hits.append(m.group(0))
    return hits


def safety_text(report: str) -> str:
    """The report with every number removed, so only the wording remains."""
    t = re.sub(r"[0-9][0-9,]*(\.[0-9]+)?", "#", report)
    return re.sub(r"[ \t]+", " ", t)


# ------------------------------------------------------------ 1. model wrong
def test_there_are_real_misses_to_test(world):
    """If the workflow missed nothing, case 1 would test nothing."""
    assert len(world["missed"]) >= 1


def test_missed_fields_are_never_cleared(world):
    """The fields the model got wrong are exactly the ones that matter."""
    for i in world["missed"]:
        rep = render(world["feats"][i], world["est"][i], world["basis"][i],
                     flagged=False)
        for pat in CLEARING:
            assert not unnegated(pat, rep), (
                f"missed field-season {world['feats'][i]['uid']} report "
                f"contains un-negated {pat!r}")
        assert "NOT EVIDENCE THAT YOUR CROP IS FINE" in rep.upper()
        assert "REPORT IT NOW" in rep.upper()


# --------------------------------------------------------- 2. missing evidence
def test_cold_start_says_there_is_no_record(world):
    for i in world["cold"][:5]:
        rep = render(world["feats"][i], world["est"][i], "satellite",
                     flagged=i in world["flagged"])
        assert "NO RELIABLE ESTIMATE" in rep.upper()
        assert "NO RECORD" in rep.upper()


# --------------------------------------------------------------- 3. invariance
@pytest.mark.parametrize("flagged", [False, True])
def test_cold_start_warning_does_not_depend_on_the_number(world, flagged):
    """Same insufficient evidence, very different estimates, same message.

    If the wording softened as the satellite estimate rose, a high number
    would act as reassurance even though the warning says the number cannot
    be trusted.
    """
    i = world["cold"][0]
    r = world["feats"][i]
    texts = {safety_text(render(r, est, "satellite", flagged))
             for est in (300.0, 900.0, 1500.0, 2200.0, 3500.0)}
    assert len(texts) == 1, (
        "the cold-start report's wording changes with the satellite "
        "estimate; only the numbers may change")


def test_history_based_wording_also_invariant_to_the_number(world):
    i = next(j for j in range(len(world["feats"]))
             if world["basis"][j] == "history")
    r = world["feats"][i]
    texts = {safety_text(render(r, est, "history", False))
             for est in (500.0, 1800.0, 3000.0)}
    assert len(texts) == 1


def test_the_number_is_the_only_thing_that_moves(world):
    """Invariance must not be achieved by printing no number at all."""
    i = world["cold"][0]
    r = world["feats"][i]
    a = render(r, 400.0, "satellite", False)
    b = render(r, 3000.0, "satellite", False)
    assert a != b
    assert "400" in a and "3,000" in b or "3000" in b


# ---------------------------------------------------- 4. prediction vs damage
def test_no_report_presents_a_prediction_as_verified_damage(world):
    idx = list(world["flagged"])[:6] + world["missed"] + world["cold"][:4]
    for i in idx:
        rep = render(world["feats"][i], world["est"][i], world["basis"][i],
                     flagged=i in world["flagged"])
        for pat in CLEARING[-3:]:
            assert not unnegated(pat, rep), (
                f"report presents a model output as {pat!r}")


def test_flagged_report_says_it_is_not_a_finding_of_loss(world):
    i = next(iter(world["flagged"]))
    rep = render(world["feats"][i], world["est"][i], world["basis"][i],
                 flagged=True)
    assert "NOT A FINDING OF LOSS" in rep.upper()


# --------------------------------------------------------- language
NUMS = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def test_tamil_report_carries_the_safety_sentences_in_tamil(world):
    """Fully translated: the key warnings appear in Tamil, not English."""
    tr = fr.translations("ta")
    i = world["cold"][0]
    rep = render(world["feats"][i], world["est"][i], "satellite", False,
                 lang="ta")
    assert "NOT FULLY TRANSLATED" not in rep
    flat = " ".join(rep.split())
    for rid in ("C2", "C3", "D3", "D6", "D8", "E2"):
        assert " ".join(tr[rid].split()) in flat, f"{rid} missing in Tamil"
    assert "NOT EVIDENCE THAT YOUR CROP IS FINE" not in rep.upper()


@pytest.mark.parametrize("basis,flagged", [("satellite", False),
                                           ("history", True)])
def test_tamil_report_keeps_every_number(world, basis, flagged):
    """Numbers carry the evidence and the helpline; translation must not
    drop, add or change one."""
    i = world["cold"][0]
    r = world["feats"][i]
    en = render(r, 1850.0, basis, flagged, lang="en", rank=3, total=62)
    ta = render(r, 1850.0, basis, flagged, lang="ta", rank=3, total=62)
    assert sorted(NUMS.findall(en)) == sorted(NUMS.findall(ta))


@pytest.mark.parametrize("flagged", [False, True])
def test_tamil_wording_does_not_depend_on_the_number(world, flagged):
    """Invariance holds in Tamil too: a high estimate cannot soften it."""
    i = world["cold"][0]
    r = world["feats"][i]
    texts = {safety_text(render(r, est, "satellite", flagged, lang="ta"))
             for est in (300.0, 900.0, 1500.0, 2200.0, 3500.0)}
    assert len(texts) == 1


def test_a_stale_translation_falls_back_to_english_and_says_so(world,
                                                               monkeypatch):
    """If an English safety sentence changes, its old Tamil must not be
    printed; the English is, under a banner saying the report is not fully
    translated."""
    monkeypatch.setitem(fr.EN_SENT, "C2", fr.EN_SENT["C2"])   # unchanged
    i = world["cold"][0]
    r = world["feats"][i]
    monkeypatch.setitem(fr.EN_SENT, "C2",
                        "THIS IS NOT EVIDENCE THAT YOUR CROP IS FINE TODAY.")
    rep = render(r, 1850.0, "satellite", False, lang="ta")
    assert "NOT FULLY TRANSLATED" in rep
    assert "NOT EVIDENCE THAT YOUR CROP IS FINE" in rep.upper()


def test_translation_file_has_no_stale_entries():
    """Every sentence has a valid translation; an edit to the English fails
    here until the translation is redone, rather than silently degrading."""
    now = fr.current_english()
    have = fr.translations("ta")
    stale = sorted(set(now) - set(have))
    assert not stale, f"re-translate: {stale}"


def test_english_blocks_rejoin_to_their_sentences():
    """The English line breaks and the sentences translators saw agree."""
    for ids, lines in fr.REPORT_LINES.items():
        joined = " ".join(" ".join(ln.split()) for ln in lines)
        assert joined == " ".join(fr.EN_SENT[i] for i in ids), ids


# ------------------------------------------------------------ the guard bites
def test_invariance_check_catches_wording_that_softens_with_the_number():
    """A report that reassures at high estimates must fail the comparison.

    Verified once by mutation (a report adding "your crop is fine" above
    2000 kg/acre failed both invariance tests); pinned here so the check
    cannot quietly stop working.
    """
    low = "  Estimated yield: 900 kg/acre\n  NO RECORD for this field.\n"
    high = ("  Estimated yield: 3000 kg/acre\n  NO RECORD for this field.\n"
            "  Good news: your crop is fine this season.\n")
    assert safety_text(low) != safety_text(high)
    assert unnegated(CLEARING[0], high)
    assert not unnegated(CLEARING[0],
                         "This is not evidence that your crop is fine.")
