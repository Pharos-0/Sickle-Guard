"""Do the documents still agree with the artefacts?

    python src/check_claims.py
    python src/check_claims.py --update     # adopt new values, then review

WHAT THIS CATCHES THAT NOTHING ELSE DOES
----------------------------------------
tests/ proves the software behaves as specified.
src/audit_evaluation.py proves the evaluation was carried out correctly.
Neither reads a single sentence of prose.

Four documents now quote results. A rerun that moved a number would leave
every one of them stating the old value, and the prose is what a reader
believes. This binds each quoted figure to the artefact it came from and
to the exact string the documents must contain, and fails on either
mismatch.

Two checks per claim:

  VALUE   the figure in the artefact still equals the declared expectation
          within tolerance
  PROSE   the rendered string appears verbatim in every document that is
          declared to quote it

The second is the one that bites. If an artefact moves and the expectation
is updated, the rendered string changes, and every document quoting it
fails until it is rewritten.

Registry: configs/claims.yaml.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

PROJ = Path(__file__).resolve().parents[1]
REG = PROJ / "configs" / "claims.yaml"
METRICS = PROJ / "results" / "metrics"
OUT = PROJ / "results" / "metrics" / "check_claims.json"


# Documents are typeset: they use U+2212 MINUS SIGN rather than the ASCII
# hyphen, non-breaking spaces inside figures, and en dashes in ranges. None
# of that changes the number, so it is normalised before matching. This is
# a normalisation, not a loosening: the VALUE check below is unaffected.
TYPO = {"−": "-", "–": "-", "—": "-", " ": " ",
        " ": " ", " ": " "}


def norm(t: str) -> str:
    """Typography and line wrapping removed; digits untouched.

    Markdown is hard-wrapped, so a quoted figure routinely straddles a
    line break -- "3 of" at the end of one line and "11" at the start of
    the next. A plain substring test misses it and reports stale prose
    that is not stale. Collapsing every run of whitespace to one space
    fixes that without touching a single digit.
    """
    for a, b in TYPO.items():
        t = t.replace(a, b)
    # Markdown structure, not content: a hard-wrapped blockquote carries a
    # "> " on every line, and a wrapped list item carries indentation. Left
    # in, they split a quoted sentence into fragments that match nothing,
    # so the guard would fail the moment someone re-wrapped a paragraph --
    # which is the normal thing to do to a markdown file.
    t = re.sub(r"(?m)^[ 	]*(?:>[ 	]?)+", "", t)
    t = re.sub(r"(?m)^[ 	]*(?:[-*+][ 	]+|\d+[.)][ 	]+)", "", t)
    return re.sub(r"\s+", " ", t)


def renderings(c) -> list[str]:
    """`renders_as` may be one string or several equivalent ones.

    Several are allowed because a summary table and a detailed matrix
    legitimately round the same number differently. They are not a
    loophole: each one is checked against `expected` at its own precision,
    so a rendering that is simply wrong fails here rather than passing by
    being listed.
    """
    r = c["renders_as"]
    return [str(x) for x in (r if isinstance(r, list) else [r])]


def rendering_is_consistent(text: str, expected: float):
    """Does this rendered string round-trip to the expected value?

    Returns (ok, parsed). A string with no number in it, such as a count
    rendered as `3 of 11`, is checked on its first number.
    """
    # A thousands separator ("12,505") is part of the number, not the end
    # of it; without this a large count could only be bound by writing it
    # unformatted in the prose.
    m = re.search(r"[-+]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:[.][0-9]+)?",
                  norm(text))
    if not m:
        return False, None
    num = m.group(0).replace(",", "")
    v = float(num)
    dps = len(num.split(".")[1]) if "." in num else 0
    # the rendered figure must equal the expectation at its own precision,
    # allowing the rendering to be a percentage of a fraction
    # A signed delta is often written as a magnitude: a stored
    # -14.3 kg/acre improvement reads as "14.3 kg/acre" in prose. The
    # absolute value is therefore an accepted rendering, and a fraction may
    # be rendered as a percentage.
    e = float(expected)
    for cand in (e, e * 100.0, abs(e), abs(e) * 100.0):
        if abs(round(cand, dps) - v) < 10 ** (-dps) / 2 + 1e-12:
            return True, v
    return False, v


def dig(obj, path):
    for part in path.split("/"):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


NEGATORS = ("not ", "never ", "cannot ", "can not ", "does not ", "do not ",
            "is not ", "are not ", "no evidence", "without ", "rather than",
            "instead of", "must not", "would be", "would have")


def prose_rules(reg, projroot):
    """Canonical sentences that must appear, phrasings that must not.

    The forbidden list is checked against EVERY markdown file in the
    project, not only the declared ones: an overstatement in an
    undeclared file is just as damaging. Matching is negation-aware for
    the same reason the farmer-report scan is -- a document is allowed,
    and indeed required, to say the system does NOT detect crop damage.
    """
    rules = reg.get("prose_rules") or {}
    out = {"must_appear": [], "must_not_appear": []}
    ok = True

    for r in rules.get("must_appear", []):
        f = projroot / r["document"]
        want = norm(" ".join(str(r["text"]).split()))
        have = norm(f.read_text(encoding="utf8")) if f.exists() else ""
        present = want in have
        ok = ok and present
        out["must_appear"].append({"id": r["id"],
                                   "document": r["document"],
                                   "present": bool(present)})
        print(f"  {'ok  ' if present else 'FAIL'}  {r['id']} appears "
              f"verbatim in {r['document']}")

    md = sorted(projroot.glob("*.md")) + sorted(projroot.glob("docs/*.md"))
    for r in rules.get("must_not_appear", []):
        phrase = norm(str(r["phrase"]).lower())
        hits = []
        for f in md:
            t = norm(f.read_text(encoding="utf8").lower())
            start = 0
            while True:
                i = t.find(phrase, start)
                if i < 0:
                    break
                window = t[max(0, i - 110):i]
                if not any(n in window for n in NEGATORS):
                    hits.append(f"{f.name}: ...{t[max(0, i - 50):i + len(phrase) + 20]}...")
                start = i + 1
        if hits:
            ok = False
        out["must_not_appear"].append({"phrase": r["phrase"],
                                       "unnegated_hits": hits})
        print(f"  {'ok  ' if not hits else 'FAIL'}  no un-negated "
              f"{r['phrase']!r}"
              + (f"  -> {hits[0][:90]}" if hits else ""))
    return ok, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="rewrite expected values in the registry")
    args = ap.parse_args()

    if not REG.exists():
        print(f"FATAL: registry missing at {REG.relative_to(PROJ)}")
        return 1
    reg = yaml.safe_load(REG.read_text(encoding="utf8"))
    default_tol = float(reg.get("default_tolerance", 1e-9))
    claims = reg["claims"]

    docs = {}
    for name in sorted({d for c in claims for d in c.get("quoted_in", [])}):
        f = PROJ / name
        docs[name] = f.read_text(encoding="utf8") if f.exists() else None

    bar = "=" * 80
    print(bar)
    print(f"CLAIM CHECK against {REG.relative_to(PROJ)} version "
          f"{reg['version']}, declared {reg['declared_on']}")
    print(f"{len(claims)} claims across {len(docs)} documents")
    print(bar)
    print(f"{'id':<28s}{'expected':>16s}{'found':>16s}  value  prose")

    rows, bad_value, bad_prose, missing = [], [], [], []
    for c in claims:
        tol = float(c.get("tolerance", default_tol))
        f = METRICS / c["file"]
        got, note = None, ""
        if not f.exists():
            note = f"missing {c['file']}"
        else:
            try:
                got = dig(json.loads(f.read_text(encoding="utf8")),
                          c["path"])
            except (KeyError, IndexError, TypeError, ValueError) as e:
                note = f"{type(e).__name__}: {e}"

        if got is None:
            v_ok = False
            missing.append(c["id"])
        else:
            v_ok = abs(float(got) - float(c["expected"])) <= tol
            if not v_ok:
                bad_value.append(c["id"])

        rends = renderings(c)
        # a rendering that does not round-trip to the expectation is a bug
        # in the registry, not in the documents
        bad_rend = []
        if got is not None:
            for r_ in rends:
                ok_, parsed = rendering_is_consistent(r_, c["expected"])
                if not ok_:
                    bad_rend.append(f"{r_!r} parses to {parsed}")
        where_missing = [d for d in c.get("quoted_in", [])
                         if docs.get(d) is None
                         or not any(norm(r_) in norm(docs[d])
                                    for r_ in rends)]
        p_ok = not where_missing and not bad_rend
        s = " | ".join(rends)
        if not p_ok:
            bad_prose.append(c["id"])

        rows.append({"id": c["id"], "expected": c["expected"], "found": got,
                     "value_ok": bool(v_ok), "renders_as": s,
                     "prose_ok": bool(p_ok),
                     "documents_missing_the_string": where_missing,
                     "inconsistent_renderings": bad_rend,
                     "note": note})
        if bad_rend:
            note = (note + "; " if note else "") +                 "rendering disagrees with expected: " + "; ".join(bad_rend)
        g = "n/a" if got is None else f"{float(got):.9g}"
        print(f"{c['id']:<28s}{float(c['expected']):>16.9g}{g:>16s}"
              f"{('  ok ' if v_ok else ' FAIL'):>7s}"
              f"{('  ok' if p_ok else ' FAIL'):>7s}"
              + (f"   {note}" if note else "")
              + (f"   not in: {', '.join(where_missing)}"
                 if where_missing else ""))

    print()
    n_ok = sum(1 for r in rows if r["value_ok"] and r["prose_ok"])
    print(bar)
    print(f"{n_ok}/{len(rows)} claims agree with both the artefact and the "
          f"prose")
    if missing:
        print(f"MISSING ARTEFACT ({len(missing)}): {', '.join(missing)}")
        print("   Run the pipeline. A missing artefact is not a pass.")
    if bad_value:
        print(f"VALUE DRIFT ({len(bad_value)}): {', '.join(bad_value)}")
        print("   The artefact no longer matches what was declared.")
    if bad_prose:
        print(f"STALE PROSE ({len(bad_prose)}): {', '.join(bad_prose)}")
        print("   A document no longer contains the figure it is declared")
        print("   to quote. Rewrite the document; do not relax the check.")

    if args.update:
        txt = REG.read_text(encoding="utf8").replace("\r\n", "\n")
        lines = txt.split("\n")
        n = 0
        for r in rows:
            if r["value_ok"] or r["found"] is None:
                continue
            for i, ln in enumerate(lines):
                if ln.strip() == f"- id: {r['id']}":
                    for j in range(i, min(i + 40, len(lines))):
                        if lines[j].strip().startswith("expected:"):
                            pad = lines[j][:len(lines[j])
                                           - len(lines[j].lstrip())]
                            lines[j] = f"{pad}expected: {r['found']!r}"
                            n += 1
                            break
                    break
            print(f"  updated {r['id']}: {r['expected']!r} -> "
                  f"{r['found']!r}")
        if n:
            REG.write_text("\n".join(lines), encoding="utf8", newline="\n")
            print(f"rewrote {n} expected value(s).")
            print("Now update `renders_as` and every document that quotes "
                  "it, then re-run without --update.")

    print()
    print("PROSE RULES")
    prose_ok, prose = prose_rules(reg, PROJ)
    if not prose_ok:
        print("   A canonical sentence is missing, or a document claims")
        print("   something the evidence does not support. Fix the prose.")

    report = {"registry_version": reg["version"], "n_claims": len(rows),
              "prose_rules": prose, "prose_ok": bool(prose_ok),
              "n_ok": n_ok, "value_drift": bad_value,
              "stale_prose": bad_prose, "missing_artefact": missing,
              "claims": rows}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"wrote {OUT.relative_to(PROJ)}")
    if bad_value or bad_prose or missing or not prose_ok:
        print("CLAIM CHECK FAILED")
        return 1
    print("CLAIM CHECK PASSED: every quoted figure traces to an artefact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
