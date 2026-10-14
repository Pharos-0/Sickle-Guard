"""The prose guard must catch an overstatement, not merely never fire.

`src/check_claims.py` enforces that the canonical objective appears
verbatim and that a list of overstatements appears nowhere. Both halves
pass today. A guard that passes because its matching is broken looks
exactly the same from the outside, so these tests feed it violations and
require it to fail.

The negation-awareness matters as much as the matching. Every document in
this project is REQUIRED to say the system does not detect crop damage, so
a guard that flagged the phrase unconditionally would force the honest
sentence out of the writing -- which is the opposite of what it is for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from check_claims import norm, prose_rules  # noqa: E402

OBJECTIVE = (
    "We identify conditions under which satellite-based agricultural "
    "predictions can be unreliable, empirically evaluate safeguards for "
    "detecting those failures, and develop a decision-support prototype "
    "to help prioritize verification while guarding against false "
    "reassurance.")


def registry(must_appear=True, forbid=("detects crop damage",)):
    r = {"prose_rules": {"must_appear": [], "must_not_appear": [
        {"phrase": p, "why": "test"} for p in forbid]}}
    if must_appear:
        r["prose_rules"]["must_appear"] = [
            {"id": "canonical_objective", "document": "SCOPE.md",
             "text": OBJECTIVE, "why": "test"}]
    return r


def write(root: Path, name: str, text: str):
    (root / name).write_text(text, encoding="utf8")


def test_guard_catches_an_unnegated_overstatement(tmp_path, capsys):
    write(tmp_path, "SCOPE.md", OBJECTIVE)
    write(tmp_path, "CLAIM.md", "The system detects crop damage reliably.")
    ok, out = prose_rules(registry(), tmp_path)
    assert ok is False
    hits = out["must_not_appear"][0]["unnegated_hits"]
    assert hits and "CLAIM.md" in hits[0]


def test_guard_allows_the_required_negated_sentence(tmp_path):
    write(tmp_path, "SCOPE.md", OBJECTIVE)
    write(tmp_path, "CLAIM.md",
          "This system does not detect crop damage, and must never be "
          "described as if it did.")
    ok, out = prose_rules(registry(), tmp_path)
    assert ok is True
    assert out["must_not_appear"][0]["unnegated_hits"] == []


def test_guard_scans_files_nobody_declared(tmp_path):
    """An overstatement in an undeclared file is just as damaging."""
    write(tmp_path, "SCOPE.md", OBJECTIVE)
    (tmp_path / "docs").mkdir()
    write(tmp_path, "docs/stray.md", "Our tool detects crop damage.")
    ok, out = prose_rules(registry(), tmp_path)
    assert ok is False
    assert "stray.md" in out["must_not_appear"][0]["unnegated_hits"][0]


def test_missing_canonical_objective_fails(tmp_path):
    write(tmp_path, "SCOPE.md", "Some other framing of the objective.")
    ok, out = prose_rules(registry(), tmp_path)
    assert ok is False
    assert out["must_appear"][0]["present"] is False


def test_objective_must_match_verbatim_not_approximately(tmp_path):
    """A paraphrase that drops a load-bearing word must not pass.

    'protecting farmers from' claims an outcome for a person; 'guarding
    against' is a property of the artefact. The difference is the whole
    point, so near-enough is not enough.
    """
    weakened = OBJECTIVE.replace(
        "guarding against false reassurance",
        "protecting farmers from false reassurance")
    write(tmp_path, "SCOPE.md", weakened)
    ok, out = prose_rules(registry(forbid=()), tmp_path)
    assert ok is False
    assert out["must_appear"][0]["present"] is False


def test_objective_survives_markdown_line_wrapping(tmp_path):
    """Hard-wrapped prose must still match, or the guard is unusable.

    Wrapped at word boundaries, as an editor does it, and as a blockquote,
    because that is how the objective is presented. A guard that broke
    whenever someone re-flowed a paragraph would be abandoned in a week.
    """
    import textwrap
    wrapped = "\n".join("> " + ln
                        for ln in textwrap.wrap(OBJECTIVE, width=60))
    write(tmp_path, "SCOPE.md", "# Scope\n\n" + wrapped + "\n")
    ok, out = prose_rules(registry(forbid=()), tmp_path)
    assert out["must_appear"][0]["present"] is True
    assert ok is True


def test_wrapping_tolerance_does_not_excuse_a_changed_word(tmp_path):
    """Normalising whitespace must not become normalising meaning."""
    import textwrap
    altered = OBJECTIVE.replace("empirically evaluate", "demonstrate")
    wrapped = "\n".join("> " + ln
                        for ln in textwrap.wrap(altered, width=60))
    write(tmp_path, "SCOPE.md", "# Scope\n\n" + wrapped + "\n")
    ok, out = prose_rules(registry(forbid=()), tmp_path)
    assert out["must_appear"][0]["present"] is False
    assert ok is False


@pytest.mark.parametrize("phrase", [
    "validated with farmers", "ready for PMFBY", "improves insurance",
    "farmers are better off"])
def test_each_forbidden_phrase_is_actually_matchable(tmp_path, phrase):
    """Every entry on the list must be capable of firing.

    A typo in the registry would make an entry permanently green, which
    reads as a guard and is not one.
    """
    write(tmp_path, "SCOPE.md", OBJECTIVE)
    write(tmp_path, "CLAIM.md", f"This work is {phrase} already.")
    ok, out = prose_rules(registry(forbid=(phrase,)), tmp_path)
    assert ok is False
    assert out["must_not_appear"][0]["unnegated_hits"]


def test_the_real_registry_phrases_all_fire_on_a_bare_sentence(tmp_path):
    """Run the project's own list, not a fixture, against violations."""
    import yaml
    reg = yaml.safe_load(
        (PROJ / "configs/claims.yaml").read_text(encoding="utf8"))
    phrases = [r["phrase"] for r in reg["prose_rules"]["must_not_appear"]]
    assert len(phrases) >= 5
    write(tmp_path, "SCOPE.md", OBJECTIVE)
    for i, p in enumerate(phrases):
        write(tmp_path, f"v{i}.md", f"It is {p} today.")
    ok, out = prose_rules(
        {"prose_rules": {"must_appear": [],
                         "must_not_appear": reg["prose_rules"]["must_not_appear"]}},
        tmp_path)
    assert ok is False
    silent = [r["phrase"] for r in out["must_not_appear"]
              if not r["unnegated_hits"]]
    assert not silent, f"these registry phrases never fire: {silent}"


def test_norm_collapses_wrapping_without_touching_digits():
    assert norm("3 of\n  11") == "3 of 11"
    assert norm("−0.093") == "-0.093"


def test_thousands_separator_is_read_as_one_number():
    """'12,505' must parse as 12505, not as 12, or a large count could only
    be bound by printing it unformatted; and the rounding must still bite."""
    from check_claims import rendering_is_consistent
    assert rendering_is_consistent("about 12,505 fields", 12504.762)[0]
    assert not rendering_is_consistent("about 12,505 fields", 12404.762)[0]
    assert rendering_is_consistent("3 of 11", 3)[0]
    assert rendering_is_consistent("-0.093", -0.0931917641150728)[0]
