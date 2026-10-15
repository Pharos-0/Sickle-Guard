"""Translation workbook for the farmer report (English -> Tamil).

    python src/make_translation_workbook.py

Writes translation/SICKLE-Guard_report_Tamil_translation.xlsx with:
  Read me          what to fill in, and the rules that keep the report safe
  Translate        every sentence of the report, in reading order, with the
                   current Tamil where it exists and a column for yours
  Report (English) the report exactly as the code prints it, two variants
  Report (Tamil)   the current Tamil report, showing what is untranslated

The rows are SENTENCES, not printed lines: the English report wraps
sentences across lines, and the line breaks will be redone for Tamil.
The English text, the contacts, the deadlines and the notices are read from
src/farmer_report.py and configs/scheme_contacts.yaml, so the workbook
cannot drift from the report. The two report variants use ILLUSTRATIVE
inputs (field 999 is not a SICKLE field).
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import farmer_report as fr  # noqa: E402

OUT = ROOT / "translation" / "SICKLE-Guard_report_Tamil_translation.xlsx"
EN, TA = fr.STR["en"], fr.STR["ta"]
CFG = fr.contacts()

ARIAL = "Arial"
TAMIL = "Nirmala UI"          # ships with Windows and covers Tamil script
MONO = "Consolas"
YELLOW = PatternFill("solid", fgColor="FFF2CC")
HEAD = PatternFill("solid", fgColor="1F3B73")
SAFE = PatternFill("solid", fgColor="FCE4E4")
GREY = PatternFill("solid", fgColor="EFEFEF")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

ALWAYS = "always"
SAFETY = "SAFETY-CRITICAL"


def rows():
    """(id, section, when shown, priority, English, current Tamil, notes)."""
    d = CFG["deadlines"]
    c = {x["name"]: x for x in CFG["contacts"]}
    nt = {k: " ".join(v.split()) for k, v in CFG["mandatory_notices"].items()}
    keep_num = "Keep every number exactly as written."
    return [
        # ---- header and estimate
        ("A1", "Header", ALWAYS, "", EN["title"], TA["title"],
         "Report title."),
        ("A2", "Header", ALWAYS, "", EN["field"], TA["field"],
         "Followed by the field number, e.g. 'Field 301'."),
        ("A3", "Header", ALWAYS, "", EN["season"], TA["season"],
         "Followed by the season name and year, e.g. 'Season oct-mar 2020'."),
        ("A4", "Estimate", ALWAYS, "", EN["est"], TA["est"],
         "Followed by a number and the unit kg/acre (row A5)."),
        ("A5", "Estimate", ALWAYS, "", "kg/acre", "",
         "Unit after the yield number. Suggest the form a Tamil farmer "
         "reads most easily (e.g. kilograms per acre)."),
        ("A6", "Estimate", ALWAYS, "", EN["basis"], TA["basis"],
         "Label; followed by row A7 or A8."),
        ("A7", "Estimate", "field has a harvest record", "", EN["hist"],
         TA["hist"], ""),
        ("A8", "Estimate", "field has NO harvest record", "", EN["sat"],
         TA["sat"], ""),
        ("A9", "Estimate", ALWAYS, "", EN["vs"], TA["vs"],
         "Followed by a percentage, e.g. '101%'."),
        # ---- flagged
        ("B1", "Flag", "field is flagged", "", EN["flag"], TA["flag"],
         "Printed in capitals between >>> <<< marks."),
        ("B2", "Flag", "field is flagged", SAFETY,
         "This field is among the lower estimates in its area. That is a "
         "reason to look, not a finding of loss.", "",
         "Must not read as 'your crop is damaged' or 'you will be paid'."),
        ("B3", "Flag", "field is flagged", "",
         "Most fields flagged this way yield below average MOST YEARS. Being "
         "flagged is usually a statement about the land, not about this "
         "season.", "", "Capitals in English mark emphasis; keep the emphasis."),
        # ---- not flagged
        ("C1", "Flag", "field is NOT flagged", "", EN["noflag"] + ".",
         TA["noflag"], ""),
        ("C2", "Flag", "field is NOT flagged", SAFETY,
         "THIS IS NOT EVIDENCE THAT YOUR CROP IS FINE.", "",
         "The most important sentence in the report. It must not soften "
         "into reassurance in Tamil."),
        ("C3", "Flag", "field is NOT flagged", SAFETY,
         "This system misses losses. If you think your crop was damaged, "
         "report it anyway.", "", ""),
        # ---- how much to trust this
        ("D1", "Trust", ALWAYS, "", EN["conf"], TA["conf"], "Section heading."),
        ("D2", "Trust", "field has a harvest record", "",
         "Moderate. Built from this field's own past harvests, which on test "
         "data predicted yield far better than the satellite model (error 200 "
         "vs 346 kg/acre).", "", keep_num),
        ("D3", "Trust", "field has NO harvest record", SAFETY,
         "NO RELIABLE ESTIMATE.", "", "Heading-like warning; keep capitals."),
        ("D4", "Trust", "field has NO harvest record", "",
         "No past harvest on record, so the figure above is a satellite "
         "guess.", "", ""),
        ("D5", "Trust", "field has NO harvest record", "",
         "On test data it was NOT shown to beat simply using the area average "
         "(gain 14 kg/acre, interval -30 to +4, includes zero).", "",
         keep_num + " If the bracket is too technical for a farmer, say so "
         "in your comments."),
        ("D6", "Trust", "field has NO harvest record", SAFETY,
         "Do not act on the number itself.", "", ""),
        ("D7", "Trust", "field has NO harvest record", "",
         "This warning means there is NO RECORD for this field. It does NOT "
         "mean this estimate is worse than the ones without the warning: "
         "tested, the difference in error is +41 kg/acre, interval -58 to "
         "+130, which includes zero.", "", keep_num),
        ("D8", "Trust", ALWAYS, SAFETY,
         "This system CANNOT tell whether this particular season was bad for "
         "this field.", "", ""),
        ("D9", "Trust", ALWAYS, "",
         "That was tested and did not work: on fields that are sometimes below "
         "and sometimes above the threshold, ranking AUROC is 0.626 [0.386, "
         "0.887], which includes chance.", "",
         keep_num + " 'AUROC' is a technical score; it may be better to keep "
         "the number and add a plain explanation. Suggest wording in your "
         "comments."),
        ("D10", "Trust", "report lists a rank", "",
         "Rank in this list: {rank} of {total} (lowest first).", "",
         "Keep {rank} and {total} exactly as written; the program fills them "
         "in."),
        # ---- what to do now
        ("E1", "What to do", ALWAYS, "", EN["whatnow"], TA["whatnow"],
         "Section heading."),
        ("E2", "What to do", ALWAYS, SAFETY,
         "1. If your crop was damaged, report it NOW. Do not wait for any "
         "report, including this one.", "",
         "The first instruction. It must stay urgent."),
        ("E3", "What to do", ALWAYS, "", d[0]["applies_to"], "",
         "Kind of loss this deadline applies to."),
        ("E4", "What to do", ALWAYS, SAFETY, d[0]["limit"], "",
         "Keep '72'. This deadline is not yet confirmed on an official page "
         "(see E7)."),
        ("E5", "What to do", ALWAYS, "", d[1]["applies_to"], "", ""),
        ("E6", "What to do", ALWAYS, "", d[1]["limit"], "",
         "'Intimation' means telling the insurer about the loss."),
        ("E7", "What to do", "after an unconfirmed item", SAFETY,
         "[UNVERIFIED]", "",
         "Marker shown after a deadline that could not be confirmed on an "
         "official page. Keep the square brackets."),
        ("E8", "What to do", ALWAYS, "",
         "2. Contact one of these to report or to check status:", "", ""),
        ("E9", "What to do", ALWAYS, "",
         f"{list(c)[0]}: {c[list(c)[0]]['value']}", "",
         "Official name of the national helpline; 14447 is verified. Keep the "
         "number. Give the name as farmers would know it."),
        ("E10", "What to do", ALWAYS, "",
         f"{list(c)[1]}: {c[list(c)[1]]['value']}", "",
         "Keep the number exactly."),
        ("E11", "What to do", ALWAYS, "",
         f"{list(c)[2]}: {c[list(c)[2]]['value']}", "",
         "'Crop Insurance' is the app's name as it appears in the app store; "
         "keep it in English so it can be searched."),
        ("E12", "What to do", ALWAYS, "",
         f"{list(c)[3]}: {c[list(c)[3]]['value']}", "",
         "The website's menu names are in English on the site; keep "
         "'Farmer Corner' and 'Application Status' in English, explain if "
         "needed."),
        ("E13", "What to do", "after an unconfirmed item", SAFETY,
         "[UNVERIFIED - confirm first]", "",
         "Marker after a contact that could not be confirmed. Keep the "
         "brackets."),
        ("E14", "What to do", ALWAYS, "",
         "3. Confirm the numbers above with your bank, the Common Service "
         "Centre, or the District Agriculture Officer. They could not be "
         "verified against an official page.", "",
         "Use the names these offices go by locally."),
        ("E15", "Notices", ALWAYS, SAFETY, nt["never_discourage"], "",
         "Mandatory notice. Must not be shortened."),
        ("E16", "Notices", ALWAYS, SAFETY, nt["not_official"], "",
         "Mandatory notice. Must not be shortened."),
    ]


def report_text(lang, variant):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if variant == "record":
            fr.report({"plot": 999, "season": "oct-mar", "year": 2020},
                      1240.0, "history", 72.0, True, CFG, lang, 3, 20)
        else:
            fr.report({"plot": 999, "season": "oct-mar", "year": 2020},
                      1850.0, "satellite", 101.0, False, CFG, lang)
    return buf.getvalue().rstrip("\n").split("\n")


def cell(ws, ref, value, font=ARIAL, size=10, bold=False, italic=False,
         fill=None, wrap=True, color="000000"):
    c = ws[ref]
    c.value = value
    c.font = Font(name=font, size=size, bold=bold, italic=italic, color=color)
    c.alignment = Alignment(wrap_text=wrap, vertical="top")
    if fill:
        c.fill = fill
    return c


def main() -> int:
    wb = Workbook()

    # ------------------------------------------------------------ Read me
    rm = wb.active
    rm.title = "Read me"
    rm.column_dimensions["A"].width = 110
    lines = [
        ("SICKLE-Guard field report: Tamil translation", 14, True, None),
        ("", 10, False, None),
        ("What this is", 11, True, None),
        ("Every sentence a farmer can see in the SICKLE-Guard field report, in "
         f"the order it appears. {sum(1 for x in rows() if x[5])} headings "
         "and labels already have a Tamil version; every sentence below "
         "them, including every safety warning, is still in English. The "
         "report will not be offered in "
         "Tamil until a native speaker has translated and checked it.", 10,
         False, None),
        ("", 10, False, None),
        ("What to fill in", 11, True, None),
        ("On the 'Translate' sheet, write your Tamil in the YELLOW column G "
         "(Your Tamil translation). Use the yellow column H for anything "
         "else: a better wording, a question, or a sentence you think "
         "farmers would misread. Where a Tamil version already exists "
         "(column E), check it and write your corrected version in G, or "
         "copy it if it is right.", 10, False, YELLOW),
        ("Row 2 is an EXAMPLE showing the expected format; real rows start at "
         "row 3.", 10, False, None),
        ("", 10, False, None),
        ("Rules that keep the report safe", 11, True, None),
        ("1. Rows marked SAFETY-CRITICAL (pink) must keep their full meaning "
         "and strength. In particular, the report must never sound as if it "
         "says the crop is fine, that a claim is valid, or that a loss is "
         "confirmed.", 10, False, SAFE),
        ("2. Keep every number, helpline number and website address exactly "
         "as written.", 10, False, None),
        ("3. Keep {rank} and {total} (row D10) exactly; the program fills "
         "them in.", 10, False, None),
        ("4. Keep the square-bracket markers [UNVERIFIED] in brackets, "
         "translated or not, so they stay visible.", 10, False, None),
        ("5. Translate meaning, not word by word. Plain, spoken Tamil that a "
         "farmer understands is the goal. Capitals in English show emphasis; "
         "show the same emphasis in Tamil.", 10, False, None),
        ("6. Line breaks do not matter here: the program will wrap the "
         "Tamil text.", 10, False, None),
        ("", 10, False, None),
        ("Context", 11, True, None),
        ("The 'Report (English)' sheet shows the full report exactly as the "
         "program prints it, for a field with a harvest record that is "
         "flagged and for a field with no record that is not flagged. The "
         "'Report (Tamil)' sheet shows what a Tamil reader sees today. Both "
         "use illustrative inputs: field 999 is not a real field.", 10,
         False, None),
        ("Only one of the two Flag blocks appears in any one report (B for "
         "flagged, C for not flagged), and the Trust section shows D2 or "
         "D3-D7 depending on whether the field has a harvest record (column "
         "C says when each row is shown).", 10, False, None),
    ]
    for i, (txt, sz, bold, fill) in enumerate(lines, 1):
        cell(rm, f"A{i}", txt, size=sz, bold=bold, fill=fill)

    # ------------------------------------------------------------ Translate
    ws = wb.create_sheet("Translate")
    head = ["ID", "Section", "Shown when", "Priority", "English (exact)",
            "Current Tamil", "Your Tamil translation",
            "Your comments / better wording", "Notes for the translator"]
    widths = [6, 11, 18, 15, 48, 30, 48, 32, 40]
    for i, (h, wd) in enumerate(zip(head, widths), 1):
        col = chr(64 + i)
        ws.column_dimensions[col].width = wd
        c = cell(ws, f"{col}1", h, bold=True, color="FFFFFF", fill=HEAD)
        c.border = BOX
    # Reorder so the yellow columns sit beside the English
    # (E English, F current Tamil, G yours, H comments, I notes).
    ex = ["EX", "Header", ALWAYS, "", "FIELD REPORT", TA["title"],
          TA["title"], "Example row: the existing Tamil, copied as correct.",
          "EXAMPLE ONLY: shows the expected format."]
    for j, v in enumerate(ex, 1):
        col = chr(64 + j)
        c = cell(ws, f"{col}2", v, font=TAMIL if j in (6, 7) else ARIAL,
                 italic=True, color="666666", fill=GREY)
        c.border = BOX
    r = 3
    for rid, sec, when, prio, en, ta, note in rows():
        vals = [rid, sec, when, prio, en, ta, "", "", note]
        for j, v in enumerate(vals, 1):
            col = chr(64 + j)
            fill = YELLOW if j in (7, 8) else (SAFE if prio == SAFETY and
                                               j in (1, 4, 5) else None)
            c = cell(ws, f"{col}{r}", v,
                     font=TAMIL if j in (6, 7, 8) else ARIAL,
                     bold=(j == 4 and prio == SAFETY), fill=fill)
            c.border = BOX
        r += 1
    last = r - 1
    ws.freeze_panes = "F2"
    ws.auto_filter.ref = f"A1:I{last}"
    for rr in range(2, last + 1):
        ws.row_dimensions[rr].height = 48
    dv = DataValidation(type="textLength", operator="lessThanOrEqual",
                        formula1="2000", allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"G3:H{last}")

    # ------------------------------------------------------------ reports
    for title, lang in (("Report (English)", "en"), ("Report (Tamil)", "ta")):
        rs = wb.create_sheet(title)
        rs.column_dimensions["A"].width = 95
        row = 1
        for name, variant in (("Field WITH a harvest record, FLAGGED, rank "
                               "3 of 20", "record"),
                              ("Field with NO harvest record, NOT flagged",
                               "cold")):
            cell(rs, f"A{row}", f"{name}  (illustrative inputs; field 999 is "
                 "not a real field)", bold=True, size=10.5)
            row += 1
            for ln in report_text(lang, variant):
                cell(rs, f"A{row}", ln, font=TAMIL if lang == "ta" else MONO,
                     size=9.5, wrap=False)
                row += 1
            row += 2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    n = len(rows())
    n_ta = sum(1 for x in rows() if x[5])
    n_safe = sum(1 for x in rows() if x[3] == SAFETY)
    print(f"wrote {OUT}")
    print(f"  {n} sentences to translate ({n_ta} have a Tamil version to "
          f"check, {n - n_ta} have none); {n_safe} safety-critical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
