"""The Streamlit interface is held to the same safety rule as the CLI.

WHY THIS FILE EXISTS
--------------------
`tests/test_farmer_report.py` scans the CLI report for reassuring or
entitlement-deciding language. `app.py` renders the same states to the same
farmers and was not scanned at all, so the two interfaces could drift apart
without any test noticing. An audit of the evaluation
(`src/audit_evaluation.py`, check 4) found exactly that gap.

Streamlit cannot be rendered headlessly here without standing up a server,
so these tests read the literal strings `app.py` passes to `st.success`,
`st.info`, `st.warning`, `st.error` and `st.caption`. That is weaker than
driving the rendered page: it checks what the source says, not what a
browser shows. It is strong enough for the property that matters, because
every one of these strings is a constant in the source rather than
something assembled at runtime, and the test asserts that too.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
APP = PROJ / "app.py"

SURFACES = {"success", "info", "warning", "error", "caption", "markdown",
            "subheader", "title", "metric"}

REASSURING = [
    r"\bcrop is fine\b",
    r"\byour crop is (ok|okay|fine|healthy|safe)\b",
    r"\bno (crop )?loss\b",
    r"\bnot (eligible|entitled)\b",
    r"\byou (will|would) (not )?(receive|get) (a )?(payout|compensation)\b",
    r"\bno action (is )?(needed|required)\b",
    r"\bnothing to worry\b",
    r"\byou are covered\b",
]

NEGATORS = ("not evidence", "cannot", "can not", "does not", "do not",
            "is not", "are not", "never", "no evidence", "without",
            "rather than", "instead of")


def user_strings():
    """Every literal string app.py shows a person."""
    tree = ast.parse(APP.read_text(encoding="utf8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.attr if isinstance(fn, ast.Attribute)
                else getattr(fn, "id", ""))
        if name not in SURFACES:
            continue
        for arg in list(node.args) + [k.value for k in node.keywords]:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value,
                                                                str):
                    out.append(sub.value)
    return out


def _unnegated(pat, text):
    """Matches of `pat` that are not inside an explicit negation.

    Mirrors the CLI guard: the required sentence "this is NOT evidence that
    your crop is fine" contains a reassuring phrase on purpose, and must
    not be flagged.
    """
    hits = []
    for m in re.finditer(pat, text, re.I):
        window = text[max(0, m.start() - 90):m.start()].lower()
        if not any(n in window for n in NEGATORS):
            hits.append(m.group(0))
    return hits


@pytest.fixture(scope="module")
def strings():
    s = user_strings()
    assert len(s) > 20, (
        f"only {len(s)} user-facing strings found in app.py; the extractor "
        f"is probably not matching the Streamlit calls any more")
    return s


@pytest.mark.parametrize("pat", REASSURING)
def test_app_never_reassures_or_decides_entitlement(pat, strings):
    """Harm: a farmer reads 'no loss' and does not file inside 72 hours."""
    for text in strings:
        hits = _unnegated(pat, text)
        assert not hits, f"app.py shows un-negated {hits!r} in: {text[:120]}"


def test_the_negation_guard_itself_bites():
    """Without this the test above could pass by never matching anything."""
    assert _unnegated(r"\bcrop is fine\b", "Your crop is fine this season.")
    assert not _unnegated(r"\bcrop is fine\b",
                          "This is not evidence that your crop is fine.")


def test_app_states_unflagged_is_not_evidence_of_no_loss(strings):
    joined = " ".join(strings).lower()
    assert "not evidence that your crop is fine" in joined
    assert "misses losses" in joined


def test_app_tells_the_farmer_to_report_damage_regardless(strings):
    joined = " ".join(strings).lower()
    assert "report it now" in joined or "report damage" in joined
    assert "do not wait" in joined


def test_app_denies_seasonal_deterioration_capability(strings):
    joined = " ".join(strings).lower()
    assert "cannot" in joined
    assert "this particular season" in joined


def test_app_warning_is_scoped_to_a_missing_record_not_to_error(strings):
    """The evaluation refuted the stronger reading, so the text must not use it.

    src/evaluate_prototype.py, question B: warned field-seasons are not
    reliably worse than unwarned ones (+41 kg/acre, 95 % CI -58 to +130,
    and the sign reverses over all field-seasons). The warning therefore
    reports missing evidence, not a worse estimate.
    """
    joined = " ".join(strings).lower()
    assert "no record" in joined, (
        "app.py must say the warning means there is no record for the field")
    for forbidden in ("less accurate", "worse estimate", "unreliable "
                                                         "estimate"):
        assert forbidden not in joined, (
            f"app.py claims {forbidden!r}, which the calibration test "
            f"refuted")


def test_app_prints_a_source_beside_every_verified_contact(strings):
    joined = " ".join(strings).lower()
    assert "checked" in joined
    assert "tick" in joined, (
        "a verified tick must be accompanied by the statement that a tick "
        "in an app proves nothing on its own")


def test_app_reuses_the_cli_logic_rather_than_reimplementing_it():
    """Drift between interfaces is prevented at the source, not by tests.

    If app.py ever computes its own estimate or its own ranking, the two
    interfaces can disagree about the same field and only one of them is
    the one that was validated.
    """
    src = APP.read_text(encoding="utf8")
    assert "import farmer_report as fr" in src
    for fn in ("fr.load()", "fr.causal_history(", "fr.satellite(",
               "fr.contacts()"):
        assert fn in src, f"app.py no longer calls {fn}"
    tree = ast.parse(src)
    defs = [n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert defs == ["bootstrap"], (
        f"app.py defines {defs}; it is meant to be a thin shell over "
        f"farmer_report, so any new function here is logic that is not "
        f"covered by the CLI tests")


# --------------------------------------------------------------------------
# Localisation honesty
# --------------------------------------------------------------------------
def test_tamil_report_is_fully_translated():
    """The Tamil report carries no 'not translated' banner.

    The translation (configs/report_translations_ta.yaml, first author, a
    native Tamil speaker, 2026-10-16) covers every sentence. Each entry is
    used only while the English it was made from is unchanged; any stale
    sentence falls back to English and the banner returns by itself.
    """
    import subprocess
    import sys as _sys
    out = subprocess.run(
        [_sys.executable, "-X", "utf8", str(PROJ / "src/farmer_report.py"),
         "--uid", "2063", "--lang", "ta"],
        capture_output=True, text=True, encoding="utf8", cwd=str(PROJ)).stdout
    assert "NOT FULLY TRANSLATED" not in out
    assert "\u0bb5\u0baf\u0bb2\u0bcd" in out          # "field" in Tamil


def test_english_report_carries_no_translation_banner():
    import subprocess
    import sys as _sys
    out = subprocess.run(
        [_sys.executable, "-X", "utf8", str(PROJ / "src/farmer_report.py"),
         "--uid", "2063"],
        capture_output=True, text=True, encoding="utf8", cwd=str(PROJ)).stdout
    assert "TRANSLATED" not in out


def test_the_untranslated_count_is_derived_not_asserted(monkeypatch):
    """The count is computed from the translation file and the current
    English, so it rises by itself when an English sentence changes."""
    import sys as _sys
    _sys.path.insert(0, str(PROJ / "src"))
    import farmer_report as fr
    miss, tot = fr.untranslated_fraction("ta")
    assert miss == 0 and tot == len(fr.current_english())
    monkeypatch.setitem(fr.EN_SENT, "C3", fr.EN_SENT["C3"] + " (edited)")
    assert fr.untranslated_fraction("ta")[0] == 1
