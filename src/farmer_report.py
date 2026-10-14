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
record, labelled as weak, because in that case it is the only thing
available and it does beat knowing nothing.

WHAT IT IS NOT
--------------
Not connected to PMFBY. Not an insurer. Produces no eligibility decision,
no payout, no Insurance Unit figure. Contact details come from
configs/scheme_contacts.yaml and are printed as UNVERIFIED until somebody
checks them against an official source.
"""

from __future__ import annotations

import argparse
import ast
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
        "est": "Estimated yield", "basis": "Based on",
        "hist": "this field's own past harvests",
        "sat": "satellite imagery (no past harvest on record)",
        "vs": "Compared with nearby fields",
        "flag": "FLAGGED FOR ATTENTION",
        "noflag": "Not flagged by this system",
        "whatnow": "WHAT TO DO NOW",
        "conf": "HOW MUCH TO TRUST THIS",
    },
    "ta": {
        "title": "வயல் அறிக்கை", "field": "வயல்", "season": "பருவம்",
        "est": "மதிப்பிடப்பட்ட விளைச்சல்", "basis": "அடிப்படை",
        "hist": "இந்த வயலின் முந்தைய அறுவடைகள்",
        "sat": "செயற்கைக்கோள் படம் (முந்தைய அறுவடை பதிவு இல்லை)",
        "vs": "அருகிலுள்ள வயல்களுடன் ஒப்பிடுகையில்",
        "flag": "கவனத்திற்கு குறிக்கப்பட்டுள்ளது",
        "noflag": "இந்த அமைப்பால் குறிக்கப்படவில்லை",
        "whatnow": "இப்போது என்ன செய்வது",
        "conf": "இதை எவ்வளவு நம்பலாம்",
    },
}


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


def untranslated_fraction() -> tuple[int, int]:
    """How much of the report is NOT covered by the translation table.

    Counted from the source rather than maintained by hand, so it cannot
    go stale: a sentence added in English and never translated changes
    this number immediately.
    """
    src = ast.parse(Path(__file__).read_text(encoding="utf8"))
    fn = next(n for n in ast.walk(src)
              if isinstance(n, ast.FunctionDef) and n.name == "report")
    via = lit = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
            txt = ast.unparse(node)
            if "S[" in txt:
                via += 1
            elif any(isinstance(c, ast.Constant) and isinstance(c.value, str)
                     and len(c.value.strip()) > 12 for c in ast.walk(node)):
                lit += 1
    return lit, via + lit


def report(r, est, basis, pct, flagged, cfg, lang, rank=None, total=None):
    S = STR.get(lang, STR["en"])
    # A report whose headings are translated and whose warnings are not is
    # worse than one that is wholly in English: the headings signal that
    # the document was prepared for this reader, which invites trust in
    # cautions they cannot read. Rather than machine-translate safety text
    # -- a mistranslated "report damage now, do not wait" is a real harm --
    # the report says plainly that it is not localised.
    if lang != "en":
        _lit, _tot = untranslated_fraction()
        print()
        print("  " + "!" * 60)
        print(f"  THIS REPORT IS NOT TRANSLATED. {_lit} of {_tot} lines "
              f"below are in")
        print("  English, including every safety warning. The headings are")
        print("  translated and the warnings are not, which can make the")
        print("  report look prepared for you when it is not.")
        print("  Ask someone to read the English text with you, and do not")
        print("  rely on the headings alone.")
        print("  " + "!" * 60)
    w = 66
    print("=" * w)
    print(f"  {S['title']}   {S['field']} {r['plot']:.0f}   "
          f"{S['season']} {r['season']} {r['year']}")
    print("=" * w)
    print(f"  {S['est']}: {est:.0f} kg/acre")
    print(f"  {S['basis']}: {S['hist'] if basis == 'history' else S['sat']}")
    print(f"  {S['vs']}: {pct:.0f}%")
    print()
    if flagged:
        print(f"  >>> {S['flag']} <<<")
        print("      This field is among the lower estimates in its area.")
        print("      That is a reason to look, not a finding of loss.")
        print("      Most fields flagged this way yield below average MOST")
        print("      YEARS. Being flagged is usually a statement about the")
        print("      land, not about this season.")
    else:
        print(f"  {S['noflag']}.")
        print("      THIS IS NOT EVIDENCE THAT YOUR CROP IS FINE.")
        print("      This system misses losses. If you think your crop was")
        print("      damaged, report it anyway.")
    print()
    print(f"  {S['conf']}")
    if basis == "history":
        print("      Moderate. Built from this field's own past harvests,")
        print("      which on test data predicted yield far better than the")
        print("      satellite model (error 200 vs 346 kg/acre).")
    else:
        print("      NO RELIABLE ESTIMATE. No past harvest on record, so")
        print("      the figure above is a satellite guess. On test data it")
        print("      was NOT shown to beat simply using the area average")
        print("      (gain 14 kg/acre, interval -30 to +4, includes zero).")
        print("      Do not act on the number itself.")
        print("      This warning means there is NO RECORD for this field.")
        print("      It does NOT mean this estimate is worse than the ones")
        print("      without the warning: tested, the difference in error is")
        print("      +41 kg/acre, interval -58 to +130, which includes zero.")
    print("      This system CANNOT tell whether this particular season was")
    print("      bad for this field. That was tested and did not work:")
    print("      on fields that are sometimes below and sometimes above the")
    print("      threshold, ranking AUROC is 0.626 [0.386, 0.887], which")
    print("      includes chance.")
    if rank is not None:
        print(f"      Rank in this list: {rank} of {total} (lowest first).")
    print()
    print(f"  {S['whatnow']}")
    print("   1. If your crop was damaged, report it NOW. Do not wait for")
    print("      any report, including this one.")
    if cfg:
        for d in cfg.get("deadlines", []):
            mark = "" if d.get("verified") else "  [UNVERIFIED]"
            import textwrap as _tw
            for ln in _tw.wrap(d["applies_to"], width=56):
                print(f"      - {ln}" if ln == _tw.wrap(d["applies_to"], 56)[0]
                      else f"        {ln}")
            print(f"        {d['limit']}{mark}")
        print("   2. Contact one of these to report or to check status:")
        for c in cfg.get("contacts", []):
            mark = "" if c.get("verified") else "  [UNVERIFIED - confirm first]"
            print(f"      - {c['name']}: {c['value']}{mark}")
        print("   3. Confirm the numbers above with your bank, the Common")
        print("      Service Centre, or the District Agriculture Officer.")
        print("      They could not be verified against an official page.")
        print()
        import textwrap
        for k in ("never_discourage", "not_official"):
            txt = " ".join(cfg.get("mandatory_notices", {}).get(k, "").split())
            if txt:
                for line in textwrap.wrap(txt, width=62):
                    print("   " + line)
                print()
    print("=" * w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--uid")
    ap.add_argument("--village-list", action="store_true")
    ap.add_argument("--budget", type=float, default=0.33)
    ap.add_argument("--lang", default="en", choices=sorted(STR))
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
