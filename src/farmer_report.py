"""SICKLE-Guard: a farmer-facing field report.

    python src/farmer_report.py --uid 2150
    python src/farmer_report.py --uid 2150 --lang ta
    python src/farmer_report.py --village-list --budget 0.33

THE ONE DESIGN RULE
-------------------
The costs of the two errors are not symmetric. Telling a farmer their crop
looks fine when it is not may stop them filing a claim they are entitled
to, inside a 72-hour window that cannot be reopened. Telling them to go and
check when nothing is wrong costs a phone call.

So this report **never outputs a negative finding**. There is no "no loss
detected" state and no reassuring summary. A field is either flagged for
attention or not flagged, and "not flagged" is printed with an explicit
statement that it is not evidence of no loss.

WHAT THE MODEL IS ALLOWED TO SAY HERE, AND WHY
-----------------------------------------------
Measured on this dataset (see THREE_VALIDATIONS.md, forward-in-time split):

  satellite yield estimate   MAE 345.9 vs 355.9 for a global constant,
                             r = 0.252          -> NOT shown as a headline
  satellite loss flag        AUROC 0.657, CI [0.528, 0.773]
                             -> nearly chance    -> NOT used alone
  bad-season detection       AUROC 0.59, CI spans chance
                             -> NOT offered at all
  history-first ranking      AUROC 0.904, CI [0.832, 0.964], catching 25
                             of 28 losses at a 33 % budget. TWO caveats:
                             causal history covered 100 % of that test set,
                             so the number belongs to the history table and
                             not to the satellite model; and 60 % of the
                             labelled losses sit at plots below threshold in
                             EVERY season, so the metric is largely
                             identifying chronically poor land rather than
                             crop damage.   -> used, with that wording
  satellite, where it could
  actually matter            Tested on grouped CV, where 41 % of fields are
                             cold starts: adding satellite moves ranking
                             AUROC from 0.838 to 0.824, difference -0.014
                             [-0.058, +0.022]. It adds nothing measurable.

The report is therefore built on the field's own past harvests where they
exist, and says so. The satellite estimate appears only for a field with no
record, labelled as weak, because in that case it is the only field-specific
number available. It is not shown to beat the area average for such a field
(-14.3 kg/acre, 95 % CI [-30.4, +3.6], results/metrics/does_satellite_add.json),
so the report prints it with "NO RELIABLE ESTIMATE" and "Do not act on the
number itself".

WHAT IT IS NOT
--------------
Not connected to PMFBY. Not an insurer. Produces no eligibility decision,
no payout, no Insurance Unit figure. Contact details come from
configs/scheme_contacts.yaml and are printed as UNVERIFIED until somebody
checks them against an official source.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# The Tamil strings below are unprintable on a Windows console left at
# cp1252, which raised UnicodeEncodeError rather than showing the report.
# Reconfiguring stdout is the fix; `errors="replace"` keeps a terminal that
# still cannot render a glyph from losing the whole report.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import make_rf                      # noqa: E402
from reliability import cols_all, matrix        # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
# dataset root; SICKLE_DATA_ROOT as in sickle-project/.env.example
ROOT = Path(__import__("os").environ.get(
    "SICKLE_DATA_ROOT", "D:/FPD/sickledataset extracted"))
CONTACTS = PROJ / "configs" / "scheme_contacts.yaml"
SEEDS = [0, 1, 2]

STR = {
    "en": {
        "title": "FIELD REPORT", "field": "Field", "season": "Season",
        "est": "Estimated yield", "unit": "kg/acre", "basis": "Based on",
        "hist": "this field's own past harvests",
        "sat": "satellite imagery (no past harvest on record)",
        "vs": "Compared with nearby fields",
        "flag": "FLAGGED FOR ATTENTION",
        "noflag": "Not flagged by this system",
        "whatnow": "WHAT TO DO NOW",
        "conf": "HOW MUCH TO TRUST THIS",
    },
}
# Headings and labels, keyed as in the translation workbook.
HEAD_IDS = {"title": "A1", "field": "A2", "season": "A3", "est": "A4",
            "unit": "A5", "basis": "A6", "hist": "A7", "sat": "A8",
            "vs": "A9", "flag": "B1", "noflag": "C1", "conf": "D1",
            "whatnow": "E1"}

# Every sentence of the report body, as one string per workbook ID. The
# English report prints these exactly as before, from REPORT_LINES; a test
# checks that each block's lines rejoin to these sentences.
EN_SENT = {
    "B2": "This field is among the lower estimates in its area. That is a "
          "reason to look, not a finding of loss.",
    "B3": "Most fields flagged this way yield below average MOST YEARS. Being "
          "flagged is usually a statement about the land, not about this "
          "season.",
    "C2": "THIS IS NOT EVIDENCE THAT YOUR CROP IS FINE.",
    "C3": "This system misses losses. If you think your crop was damaged, "
          "report it anyway.",
    "D2": "Moderate. Built from this field's own past harvests, which on test "
          "data predicted yield far better than the satellite model (error 200 "
          "vs 346 kg/acre).",
    "D3": "NO RELIABLE ESTIMATE.",
    "D4": "No past harvest on record, so the figure above is a satellite "
          "guess.",
    "D5": "On test data it was NOT shown to beat simply using the area average "
          "(gain 14 kg/acre, interval -30 to +4, includes zero).",
    "D6": "Do not act on the number itself.",
    "D7": "This warning means there is NO RECORD for this field. It does NOT "
          "mean this estimate is worse than the ones without the warning: "
          "tested, the difference in error is +41 kg/acre, interval -58 to "
          "+130, which includes zero.",
    "D8": "This system CANNOT tell whether this particular season was bad for "
          "this field.",
    "D9": "That was tested and did not work: on fields that are sometimes below "
          "and sometimes above the threshold, ranking AUROC is 0.626 [0.386, "
          "0.887], which includes chance.",
    "D9b": "In plain terms: this score does not reliably pick out which fields "
           "need more attention.",
    "D10": "Rank in this list: {rank} of {total} (lowest first).",
    "E2": "1. If your crop was damaged, report it NOW. Do not wait for any "
          "report, including this one.",
    "E7": "[UNVERIFIED]",
    "E8": "2. Contact one of these to report or to check status:",
    "E13": "[UNVERIFIED - confirm first]",
    "E14": "3. Confirm the numbers above with your bank, the Common Service "
           "Centre, or the District Agriculture Officer. They could not be "
           "verified against an official page.",
}

# The English report's line breaks, unchanged. A block prints its IDs'
# sentences together, because the English wraps some sentences across
# shared lines (D3-D6, D8-D9).
REPORT_LINES = {
    ("B2",): ["      This field is among the lower estimates in its area.",
              "      That is a reason to look, not a finding of loss."],
    ("B3",): ["      Most fields flagged this way yield below average MOST",
              "      YEARS. Being flagged is usually a statement about the",
              "      land, not about this season."],
    ("C2",): ["      THIS IS NOT EVIDENCE THAT YOUR CROP IS FINE."],
    ("C3",): ["      This system misses losses. If you think your crop was",
              "      damaged, report it anyway."],
    ("D2",): ["      Moderate. Built from this field's own past harvests,",
              "      which on test data predicted yield far better than the",
              "      satellite model (error 200 vs 346 kg/acre)."],
    ("D3", "D4", "D5", "D6"): [
        "      NO RELIABLE ESTIMATE. No past harvest on record, so",
        "      the figure above is a satellite guess. On test data it",
        "      was NOT shown to beat simply using the area average",
        "      (gain 14 kg/acre, interval -30 to +4, includes zero).",
        "      Do not act on the number itself."],
    ("D7",): ["      This warning means there is NO RECORD for this field.",
              "      It does NOT mean this estimate is worse than the ones",
              "      without the warning: tested, the difference in error is",
              "      +41 kg/acre, interval -58 to +130, which includes zero."],
    ("D8", "D9"): [
        "      This system CANNOT tell whether this particular season was",
        "      bad for this field. That was tested and did not work:",
        "      on fields that are sometimes below and sometimes above the",
        "      threshold, ranking AUROC is 0.626 [0.386, 0.887], which",
        "      includes chance."],
    ("D9b",): ["      In plain terms: this score does not reliably pick out",
               "      which fields need more attention."],
    ("E2",): ["   1. If your crop was damaged, report it NOW. Do not wait for",
              "      any report, including this one."],
    ("E8",): ["   2. Contact one of these to report or to check status:"],
    ("E14",): ["   3. Confirm the numbers above with your bank, the Common",
               "      Service Centre, or the District Agriculture Officer.",
               "      They could not be verified against an official page."],
}

TRANSLATIONS = {"ta": Path(__file__).resolve().parents[1] / "configs"
                / "report_translations_ta.yaml"}


def parse(d):
    for f in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(d.strip(), f)
        except Exception:
            continue
    return None


def load():
    feats = list(csv.DictReader((PROJ / "results/features.csv").open(encoding="utf8")))
    tab = {r["UNIQUE_ID"]: r for r in csv.DictReader(
        (ROOT / "sickle_dataset_tabular.csv").open(encoding="utf8"))}
    for r in feats:
        r["yield"] = float(r["yield"])
        r["plot"] = float(r["plot"])
        t = tab[r["uid"]]
        r["sow"] = parse(t["SOWING_DATE"])
        r["harv"] = parse(t["HARVESTING_DATE"])
    return feats


def causal_history(feats):
    byplot = defaultdict(list)
    for i, r in enumerate(feats):
        byplot[r["plot"]].append(i)
    hist = np.full(len(feats), np.nan)
    for _, v in byplot.items():
        for i in v:
            prior = [j for j in v if feats[j]["harv"] < feats[i]["sow"]]
            if prior:
                hist[i] = float(np.mean([feats[j]["yield"] for j in prior]))
    return hist


def satellite(feats):
    cols = cols_all()
    X = matrix(feats, cols)
    med = np.nanmedian(X, 0)
    med[~np.isfinite(med)] = 0.0
    i = np.where(~np.isfinite(X))
    X[i] = np.take(med, i[1])
    tr = np.array([r["split"] in ("train", "val") for r in feats])
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    yv = np.array([r["yield"] for r in feats])
    return np.vstack([make_rf(s).fit(Z[tr], yv[tr]).predict(Z)
                      for s in SEEDS]).mean(0)


def contacts():
    if not CONTACTS.exists():
        return None
    return yaml.safe_load(CONTACTS.read_text(encoding="utf8"))


def _norm(t: str) -> str:
    return " ".join(str(t).split())


def current_english(cfg=None) -> dict:
    """The English each translatable ID currently says, from the code and
    from configs/scheme_contacts.yaml. A translation is valid only while
    the English it was made from equals this."""
    cfg = cfg if cfg is not None else contacts()
    en = {HEAD_IDS[k]: STR["en"][k] for k in HEAD_IDS}
    en["C1"] = STR["en"]["noflag"] + "."
    en.update(EN_SENT)
    if cfg:
        dl = cfg.get("deadlines", [])
        for (a, b), d in zip((("E3", "E4"), ("E5", "E6")), dl):
            en[a], en[b] = _norm(d["applies_to"]), _norm(d["limit"])
        for k, c in zip(("E9", "E10", "E11", "E12"), cfg.get("contacts", [])):
            en[k] = f"{c['name']}: {c['value']}"
        notes = cfg.get("mandatory_notices", {})
        for k, key in (("E15", "never_discourage"), ("E16", "not_official")):
            if notes.get(key):
                en[k] = _norm(notes[key])
    return en


def translations(lang: str, cfg=None) -> dict:
    """{ID: translated text} for every entry still matching the English."""
    f = TRANSLATIONS.get(lang)
    if not f or not f.exists():
        return {}
    ents = (yaml.safe_load(f.read_text(encoding="utf8")) or {}).get(
        "entries", {})
    now = current_english(cfg)
    return {k: v["ta"] for k, v in ents.items()
            if k in now and v.get("ta") and _norm(v.get("en", "")) == _norm(now[k])}


def untranslated_fraction(lang: str = "ta", cfg=None) -> tuple[int, int]:
    """(IDs with no valid translation, all translatable IDs).

    Derived from the translation file and the current English, never kept
    by hand: changing an English sentence makes its translation invalid,
    and the count rises by itself."""
    now = current_english(cfg)
    have = translations(lang, cfg)
    return sum(1 for k in now if k not in have), len(now)


def report(r, est, basis, pct, flagged, cfg, lang, rank=None, total=None):
    import textwrap as _tw
    tr = translations(lang, cfg) if lang != "en" else {}
    S = dict(STR["en"])
    for key, rid in HEAD_IDS.items():
        if rid in tr:
            S[key] = tr[rid].rstrip(".") if key == "noflag" else tr[rid]

    def block(ids, first="      ", rest="      ", fmt=None):
        """Print one block: the translation if every sentence in it has
        one, otherwise the English lines exactly as they always were."""
        if all(i in tr for i in ids):
            text = " ".join(tr[i] for i in ids)
            if fmt:
                text = text.format(**fmt)
            for ln in _tw.wrap(text, width=64, initial_indent=first,
                               subsequent_indent=rest):
                print(ln)
        else:
            for ln in REPORT_LINES[tuple(ids)]:
                print(ln.format(**fmt) if fmt else ln)

    def cfg_text(rid, english):
        return tr.get(rid, english)

    # A report whose headings are translated and whose warnings are not is
    # worse than one that is wholly in English: the headings signal that
    # the document was prepared for this reader, which invites trust in
    # cautions they cannot read. Wherever a sentence has no valid
    # translation it stays in English, and the report says so first.
    if lang != "en":
        _miss, _tot = untranslated_fraction(lang, cfg)
        if _miss:
            print()
            print("  " + "!" * 60)
            print(f"  THIS REPORT IS NOT FULLY TRANSLATED. {_miss} of {_tot} "
                  f"sentences")
            print("  below are in English, possibly including safety")
            print("  warnings. Ask someone to read the English text with")
            print("  you, and do not rely on the headings alone.")
            print("  " + "!" * 60)
    w = 66
    print("=" * w)
    try:                     # the data stores the year as 2019.0
        year = f"{int(float(r['year']))}"
    except (TypeError, ValueError):
        year = r["year"]
    print(f"  {S['title']}   {S['field']} {r['plot']:.0f}   "
          f"{S['season']} {r['season']} {year}")
    print("=" * w)
    print(f"  {S['est']}: {est:.0f} {S['unit']}")
    print(f"  {S['basis']}: {S['hist'] if basis == 'history' else S['sat']}")
    print(f"  {S['vs']}: {pct:.0f}%")
    print()
    if flagged:
        print(f"  >>> {S['flag']} <<<")
        block(["B2"])
        block(["B3"])
    else:
        print(f"  {S['noflag']}.")
        block(["C2"])
        block(["C3"])
    print()
    print(f"  {S['conf']}")
    if basis == "history":
        block(["D2"])
    else:
        block(["D3", "D4", "D5", "D6"])
        block(["D7"])
    block(["D8", "D9"])
    block(["D9b"])
    if rank is not None:
        if "D10" in tr:
            block(["D10"], fmt={"rank": rank, "total": total})
        else:
            print(f"      Rank in this list: {rank} of {total} (lowest first).")
    print()
    print(f"  {S['whatnow']}")
    block(["E2"], first="   ")
    if cfg:
        unver = cfg_text("E7", EN_SENT["E7"])
        ids = (("E3", "E4"), ("E5", "E6"))
        for (ia, il), d in zip(ids, cfg.get("deadlines", [])):
            mark = "" if d.get("verified") else f"  {unver}"
            applies = cfg_text(ia, _norm(d["applies_to"]))
            limit = cfg_text(il, _norm(d["limit"]))
            width = 60 if ia in tr else 56
            lines = _tw.wrap(applies, width=width)
            for k, ln in enumerate(lines):
                print(f"      - {ln}" if k == 0 else f"        {ln}")
            print(f"        {limit}{mark}")
        block(["E8"], first="   ")
        unc = cfg_text("E13", EN_SENT["E13"])
        for k, c in zip(("E9", "E10", "E11", "E12"),
                        cfg.get("contacts", [])):
            mark = "" if c.get("verified") else f"  {unc}"
            print(f"      - {cfg_text(k, c['name'] + ': ' + str(c['value']))}"
                  f"{mark}")
        block(["E14"], first="   ")
        print()
        for rid, key in (("E15", "never_discourage"), ("E16", "not_official")):
            txt = _norm(cfg.get("mandatory_notices", {}).get(key, ""))
            if txt:
                for line in _tw.wrap(cfg_text(rid, txt), width=62):
                    print("   " + line)
                print()
    print("=" * w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--uid")
    ap.add_argument("--village-list", action="store_true")
    ap.add_argument("--budget", type=float, default=0.33)
    ap.add_argument("--lang", default="en", choices=sorted(set(STR) | set(TRANSLATIONS)))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    feats = load()
    hist = causal_history(feats)
    sat = satellite(feats)
    est = np.where(np.isfinite(hist), hist, sat)
    basis = np.where(np.isfinite(hist), "history", "satellite")
    bench = float(np.mean([r["yield"] for r in feats if r["split"] == "train"]))
    cfg = contacts()

    if args.uid:
        k = next((i for i, r in enumerate(feats) if r["uid"] == args.uid), None)
        if k is None:
            raise SystemExit(f"no field-season with id {args.uid}")
        pool = [i for i, r in enumerate(feats)
                if r["split"] == feats[k]["split"]]
        order = sorted(pool, key=lambda i: est[i])
        kk = max(1, int(round(args.budget * len(pool))))
        flagged = k in set(order[:kk])
        report(feats[k], est[k], basis[k], 100 * est[k] / bench, flagged,
               cfg, args.lang, order.index(k) + 1, len(pool))
        return 0

    pool = [i for i, r in enumerate(feats) if r["split"] == args.split]
    order = sorted(pool, key=lambda i: est[i])
    kk = max(1, int(round(args.budget * len(pool))))
    print(f"\nFIELDS TO VISIT FIRST  ({kk} of {len(pool)}, "
          f"{args.budget:.0%} budget)")
    print("Ranked lowest estimate first. Ranking, not diagnosis.\n")
    print(f"{'#':>3s} {'field':>7s} {'season':<9s} {'est':>7s} {'%area':>6s} "
          f"{'basis':>10s}")
    for n, i in enumerate(order[:kk], 1):
        print(f"{n:>3d} {feats[i]['plot']:>7.0f} {feats[i]['season']:<9s} "
              f"{est[i]:>7.0f} {100*est[i]/bench:>6.0f} {basis[i]:>10s}")
    nh = sum(1 for i in order[:kk] if basis[i] == "history")
    print(f"\n{nh} of {kk} use the field's own past harvests; "
          f"{kk-nh} are satellite-only and much less reliable.")
    print("Fields not listed are NOT cleared. This system misses losses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
