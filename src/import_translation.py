"""Import a completed translation workbook into the report.

    python src/import_translation.py <completed.xlsx>

Reads the 'Translate' sheet written by src/make_translation_workbook.py and
filled in by a translator, checks every row, and writes
configs/report_translations_ta.yaml, which src/farmer_report.py reads.

Each entry keeps the English it was translated FROM. The report uses a
translation only while that English is unchanged, so editing an English
sentence can never leave a stale Tamil sentence saying something else; the
sentence falls back to English and the report says it is not fully
translated.

Checks, per row (any failure stops the import):
  - a translation is present
  - every number in the English appears in the Tamil, as often
  - {rank} and {total} are kept
  - a [bracketed] marker in the English is still bracketed in the Tamil

The translator's comment on D9 proposed a plain explanation of the AUROC
sentence for farmers; it is added as D9b, with an English sentence written
to match, so both languages carry it.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "report_translations_ta.yaml"
NUM = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")   # 1,470 is one number; "+4," is "+4"
STEP = re.compile(r"^(\d+)\.\s")

D9B_EN = ("In plain terms: this score does not reliably pick out which "
          "fields need more attention.")


def problems(rid, en, ta):
    out = []
    if not ta or not str(ta).strip():
        return [f"{rid}: no translation"]
    if Counter(NUM.findall(en)) != Counter(NUM.findall(ta)):
        out.append(f"{rid}: numbers differ: {NUM.findall(en)} vs "
                   f"{NUM.findall(ta)}")
    for ph in re.findall(r"\{\w+\}", en):
        if ph not in ta:
            out.append(f"{rid}: placeholder {ph} missing")
    if re.search(r"\[[A-Za-z]", en) and not re.search(r"\[[^\]]+\]", ta):
        out.append(f"{rid}: bracketed marker lost")
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1])
    ws = load_workbook(src)["Translate"]
    head = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(head)}
    entries, bad, d9_comment = {}, [], None
    for row in ws.iter_rows(min_row=2, values_only=True):
        rid = row[col["ID"]]
        if not rid or rid == "EX":
            continue
        en = str(row[col["English (exact)"]])
        ta = row[col["Your Tamil translation"]]
        note = None
        # A list step number ("3. ...") is layout, not wording: if the
        # translation dropped it, put the same number back and say so.
        m = STEP.match(en)
        if m and ta and not STEP.match(str(ta).strip()):
            ta = f"{m.group(1)}. {str(ta).strip()}"
            note = f"step number '{m.group(1)}.' restored on import"
        bad += problems(rid, en, ta)
        entries[rid] = {"en": en, "ta": str(ta).strip() if ta else ""}
        if note:
            entries[rid]["note"] = note
        if rid == "D9":
            d9_comment = row[col["Your comments / better wording"]]
    if bad:
        print("NOT IMPORTED:")
        for b in bad:
            print("  -", b)
        return 1

    if d9_comment:
        # the comment's first sentence is the explanation itself
        first = str(d9_comment).split(". ")[0].strip()
        if not first.endswith("."):
            first += "."
        entries["D9b"] = {"en": D9B_EN, "ta": first,
                          "origin": "translator's comment on D9; English "
                                    "written to match"}

    blob = {
        "language": "ta",
        "translated_by": "Pharos Sophy Samuel T J (first author; native "
                         "Tamil speaker)",
        "imported": str(date.today()),
        "source_workbook": src.name,
        "review": "translated by one native speaker; not yet checked by a "
                  "second speaker or with farmers",
        "rule": "an entry is used only while its 'en' matches the report's "
                "current English exactly",
        "entries": entries,
    }
    OUT.write_text(yaml.safe_dump(blob, allow_unicode=True, sort_keys=False,
                                  width=1000), encoding="utf8")
    print(f"wrote {OUT}: {len(entries)} entries, all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
