"""The verification console shows only what the artefacts say.

The page is generated, so these read the built file: the embedded data must
reproduce the audit chain with exactly the concatenation the page's script
uses, a tampered record must break it, every simulated panel must say so,
and no sentence may claim what the evidence rules out.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
PAGE = PROJ / "results" / "console" / "verification_console.html"


@pytest.fixture(scope="module")
def page():
    if not PAGE.exists():
        pytest.skip("run src/build_console.py first")
    html = PAGE.read_text(encoding="utf8")
    m = re.search(r"const DATA = (\{.*?\});\nconst \$", html, re.S)
    assert m, "embedded data not found"
    return html, json.loads(m.group(1))


def js_chain(rows):
    """The page's verifyChain(), in Python, line for line."""
    prev = "0" * 64
    for r in rows:
        h = hashlib.sha256("|".join(
            (prev, r["ts"], r["actor"], r["action"], r["payload"])
        ).encode("utf8")).hexdigest()
        if r["prev_hash"] != prev or h != r["hash"]:
            return r["seq"]
        prev = r["hash"]
    return None


def test_embedded_chain_verifies_as_the_page_computes_it(page):
    _, data = page
    assert len(data["demo"]["audit"]) > 10
    assert js_chain(data["demo"]["audit"]) is None


def test_the_tamper_button_breaks_the_chain_at_the_edited_entry(page):
    _, data = page
    rows = [dict(r) for r in data["demo"]["audit"]]
    i = next(k for k, r in enumerate(rows) if r["action"].startswith("record")
             and "below expectation observed" in r["payload"])
    rows[i]["payload"] = rows[i]["payload"].replace(
        "below expectation observed", "no problem observed")
    assert js_chain(rows) == rows[i]["seq"]


def test_queue_is_the_planners_output(page):
    import inspection_workflow as iw
    _, data = page
    feats, D = iw.field_table()
    con = iw.connect(":memory:")
    pid = iw.plan(con, feats, D["hist"], "test", 0.33, 0.10, seed=0)
    got = [t["uid"] for t in con.execute(
        "SELECT uid FROM tasks WHERE plan_id=? ORDER BY rank", (pid,))]
    assert got == [t["uid"] for t in data["plans"]["0.33|0.10"]]


def test_no_estimate_is_shipped_for_the_queue(page):
    _, data = page
    for tasks in data["plans"].values():
        for t in tasks:
            assert not ({"est", "estimate", "yield", "hist"} & set(t))


def test_every_simulated_panel_declares_itself(page):
    html, _ = page
    for pid in ("inspectors", "yestech"):
        block = html.split(f'id="{pid}"', 1)[1].split("</section>", 1)[0]
        assert 'class="stamp sim"' in block, pid
    assert html.count('class="panel sim"') == 2


def test_console_claims_nothing_the_evidence_rules_out(page):
    html, _ = page
    from check_claims import NEGATORS, norm
    reg = yaml.safe_load((PROJ / "configs/claims.yaml").read_text(encoding="utf8"))
    text = norm(re.sub(r"<[^>]+>", " ", html)).lower()
    for r in reg["prose_rules"]["must_not_appear"]:
        ph = norm(str(r["phrase"]).lower())
        i = text.find(ph)
        while i >= 0:
            assert any(n in text[max(0, i - 110):i] for n in NEGATORS), ph
            i = text.find(ph, i + 1)
    assert "does not detect crop damage" in text
