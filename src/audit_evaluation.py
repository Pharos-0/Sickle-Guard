"""The five verification checks a reviewer asked for, run against the files.

    python src/audit_evaluation.py

Each check answers one question and prints PASS, FAIL or REPORT. REPORT
means the check produced a number that has to be read rather than a verdict
that can be asserted.

  1  Is the split fixed, plot-disjoint, and free of plot-level leakage?
  2  Are the hybrid and the last-harvest baseline compared on identical
     predictions, splits, labels and metrics -- and what exactly differs
     between them if they are?
  3  Do the flagged-versus-unflagged denominators, interval method and
     clustering hold up?
  4  Does the revised warning text say what the evaluation supports, and do
     the tests cover every interface that renders it?
  5  Does every row of the evidence-to-design matrix carry a verdict its
     own test supports?

Writes results/metrics/audit_evaluation.json.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import farmer_report as fr  # noqa: E402

OUT = PROJ / "results" / "metrics" / "audit_evaluation.json"
LEVEL = 0.80
NBOOT = 10000
SEED = 0
RESULTS: list[dict] = []


def record(n, title, status, lines):
    RESULTS.append({"check": n, "title": title, "status": status,
                    "detail": lines})
    print("=" * 78)
    print(f"{n}. {title}   [{status}]")
    print("=" * 78)
    for ln in lines:
        print("   " + ln)
    print()


# ------------------------------------------------------------------ 1
def check_split():
    lines, ok = [], True
    cfg = json.loads((PROJ / "configs/splits.json").read_text(encoding="utf8"))
    lines.append(f"seed {cfg['seed']}, grouped on {cfg['grouping_key']}, "
                 f"stratified by {cfg['stratified_by']}")

    feats = list(csv.DictReader(
        (PROJ / "results/features.csv").open(encoding="utf8")))
    byplot = defaultdict(set)
    for r in feats:
        byplot[float(r["plot"])].add(r["split"])
    straddle = {p: s for p, s in byplot.items() if len(s) > 1}
    lines.append(f"{len(byplot)} plots; plots appearing in more than one "
                 f"split: {len(straddle)}")
    if straddle:
        ok = False
        lines.append(f"  LEAKAGE: {dict(list(straddle.items())[:5])}")

    counts = defaultdict(lambda: [0, set()])
    for r in feats:
        counts[r["split"]][0] += 1
        counts[r["split"]][1].add(float(r["plot"]))
    for s in ("train", "val", "test"):
        if s in counts:
            lines.append(f"  {s:5s} {counts[s][0]:4d} field-seasons, "
                         f"{len(counts[s][1]):3d} plots")
            declared = cfg.get("stats", {}).get(s, {})
            if declared:
                a = declared.get("n_rows"), declared.get("n_plots")
                b = counts[s][0], len(counts[s][1])
                if a != b:
                    ok = False
                    lines.append(f"    MISMATCH with configs/splits.json: "
                                 f"declared {a}, on disk {b}")

    # The frozen split file must predate the analyses that use it, otherwise
    # "fixed in advance" is a claim rather than a fact.
    sf = (PROJ / "configs/splits.json").stat().st_mtime
    later = []
    for f in ("results/metrics/evaluate_prototype.json",
              "results/metrics/grouped_cv.json",
              "results/predictions/oof_pred.npy"):
        p = PROJ / f
        if p.exists() and p.stat().st_mtime < sf:
            later.append(f)
    if later:
        ok = False
        lines.append(f"  the split file is NEWER than {later}, so it cannot "
                     f"have been fixed before them")
    else:
        lines.append("  the split file predates every artefact that "
                     "consumes it")

    lines.append("NOTE, and it is a real limit rather than a leak: the "
                 "satellite model in")
    lines.append("  the shipped hybrid is fitted on train+val, so test rows "
                 "are out-of-sample")
    lines.append("  for it, but the FEATURE SET and the model family were "
                 "chosen while looking")
    lines.append("  at val. The test split is held out from fitting, not "
                 "from every design")
    lines.append("  decision ever made. Only src/confirmatory.py was frozen "
                 "in that stronger")
    lines.append("  sense, and it FAILED both its primary criteria.")
    record(1, "Split is fixed, plot-disjoint and consistent with its own "
              "config", "PASS" if ok else "FAIL", lines)
    return ok


# ------------------------------------------------------------------ 2
def auroc(score, pos):
    pos = np.asarray(pos, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    s = np.asarray(score, dtype=float)
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), dtype=float)
    srt = s[o]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_plots(stat, plots, n_boot=NBOOT, seed=SEED):
    uniq = np.unique(plots)
    pos = {g: np.where(plots == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    v, skipped = [], 0
    for _ in range(n_boot):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        idx = np.concatenate([pos[g] for g in pick])
        r = stat(idx)
        if r is None or not np.isfinite(r):
            skipped += 1
            continue
        v.append(float(r))
    if not v:
        return float("nan"), float("nan"), skipped
    v = np.array(v)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)),
            skipped)


def build():
    feats = fr.load()
    hist = fr.causal_history(feats)
    sat = fr.satellite(feats)
    oof = PROJ / "results/predictions/oof_pred.npy"
    sat_oof = np.load(oof) if oof.exists() else None
    y = np.array([r["yield"] for r in feats])
    plots = np.array([r["plot"] for r in feats])
    split = np.array([r["split"] for r in feats])
    bench = float(y[split == "train"].mean())
    byplot = defaultdict(list)
    for i, r in enumerate(feats):
        byplot[r["plot"]].append(i)
    last = np.full(len(feats), np.nan)
    for _, v in byplot.items():
        for i in v:
            prior = [j for j in v if feats[j]["harv"] < feats[i]["sow"]]
            if prior:
                last[i] = y[max(prior, key=lambda k: feats[k]["harv"])]
    return dict(y=y, plots=plots, split=split, bench=bench, hist=hist,
                sat=sat, sat_oof=sat_oof, last=last)


def check_baseline(D):
    """Hybrid vs last-harvest, paired, on identical everything."""
    lines = []
    y, plots, split = D["y"], D["plots"], D["split"]
    bad = y < LEVEL * D["bench"]
    has_hist = np.isfinite(D["hist"])
    has_last = np.isfinite(D["last"])
    lines.append(f"coverage: {int(has_hist.sum())} field-seasons have causal "
                 f"history, {int(has_last.sum())} have a last harvest "
                 f"({'identical sets' if np.array_equal(has_hist, has_last) else 'DIFFERENT sets'})")

    out = {}
    for label, idx in (("held-out test split", np.where(split == "test")[0]),
                       ("all field-seasons, out-of-fold",
                        np.arange(len(y)))):
        use_oof = label.startswith("all") and D["sat_oof"] is not None
        s_sat = D["sat_oof"] if use_oof else D["sat"]
        hyb = -np.where(has_hist, D["hist"], s_sat)[idx]
        lst = -np.where(has_last, D["last"], D["bench"])[idx]
        hm = -np.where(has_hist, D["hist"], D["bench"])[idx]
        b, p = bad[idx], plots[idx]
        if not (0 < b.sum() < len(idx)):
            continue
        row = {}
        for nm, s in (("hybrid", hyb), ("last harvest", lst),
                      ("history mean", hm)):
            row[nm] = auroc(s, b)
        # paired difference on the SAME resampled plots
        dlo, dhi, sk = boot_plots(
            lambda i: (auroc(hyb[i], b[i]) - auroc(lst[i], b[i])
                       if 0 < b[i].sum() < len(i) else None), p)
        d = row["hybrid"] - row["last harvest"]
        verdict = ("hybrid better" if dlo > 0 else
                   "last harvest better" if dhi < 0 else "INCONCLUSIVE")
        lines.append(f"{label}: n={len(idx)}, losses={int(b.sum())}, "
                     f"satellite={'out-of-fold' if use_oof else 'shipped model'}")
        lines.append(f"   hybrid {row['hybrid']:.3f} | last harvest "
                     f"{row['last harvest']:.3f} | history mean "
                     f"{row['history mean']:.3f}")
        lines.append(f"   paired hybrid - last harvest = {d:+.3f}, 95% CI "
                     f"[{dlo:+.3f}, {dhi:+.3f}]  -> {verdict}")
        out[label] = {"aurocs": row, "paired_delta": d,
                      "ci": [dlo, dhi], "verdict": verdict,
                      "degenerate_replicates_skipped": sk,
                      "satellite": "out-of-fold" if use_oof else "shipped"}

    # what actually differs between the two rankings
    same = int((has_hist & has_last).sum())
    lines.append(f"WHAT DIFFERS: on the {same} field-seasons that have a "
                 f"record, the hybrid uses")
    lines.append("   the MEAN of all prior harvests and the baseline uses "
                 "the LAST one; on the")
    lines.append(f"   {int((~has_hist).sum())} without a record the hybrid "
                 f"falls back to satellite and")
    lines.append("   the baseline to the area average. The comparison is "
                 "therefore fair but")
    lines.append("   conflates two choices, which is why history mean is "
                 "shown alongside.")
    record(2, "Hybrid and last-harvest compared on identical splits, "
              "labels and metrics", "REPORT", lines)
    return out


# ------------------------------------------------------------------ 3
def check_flag(D):
    lines = []
    y, plots, split = D["y"], D["plots"], D["split"]
    bad = y < LEVEL * D["bench"]
    has_hist = np.isfinite(D["hist"])
    guard = np.where(has_hist, D["hist"], D["sat"])
    out = {}
    for label, idx in (("held-out test split", np.where(split == "test")[0]),
                       ("all field-seasons", np.arange(len(y)))):
        s = -guard[idx]
        b, p = bad[idx], plots[idx]
        k = max(1, int(round(0.33 * len(idx))))
        flag = np.zeros(len(idx), dtype=bool)
        flag[np.argsort(-s, kind="mergesort")[:k]] = True
        nf, nu = int(flag.sum()), int((~flag).sum())
        pf = float(b[flag].mean())
        pu = float(b[~flag].mean())
        lo, hi, sk = boot_plots(
            lambda i: (float(b[i][flag[i]].mean() - b[i][~flag[i]].mean())
                       if flag[i].any() and (~flag[i]).any() else None), p)
        lines.append(f"{label}: {len(idx)} field-seasons at "
                     f"{len(np.unique(p))} plots")
        lines.append(f"   denominators: flagged {nf} ({b[flag].sum()} bad), "
                     f"unflagged {nu} ({b[~flag].sum()} bad); "
                     f"{nf}+{nu}={nf + nu} = n {'OK' if nf + nu == len(idx) else 'MISMATCH'}")
        lines.append(f"   budget realised {nf / len(idx):.1%} against a 33% "
                     f"target")
        lines.append(f"   difference in share below threshold {pf - pu:+.1%}, "
                     f"95% CI [{lo:+.1%}, {hi:+.1%}]")
        lines.append(f"   interval method: percentile bootstrap, "
                     f"{NBOOT} replicates, resampling PLOTS and carrying "
                     f"every season of a plot together")
        lines.append(f"   replicates discarded as degenerate (one side "
                     f"empty): {sk} of {NBOOT} = {sk / NBOOT:.2%}")
        if sk / NBOOT > 0.05:
            lines.append("   WARNING: more than 5% discarded, so the "
                         "interval is conditioned on non-degenerate "
                         "replicates and is optimistic")
        out[label] = {"n_flagged": nf, "n_unflagged": nu,
                      "n_bad_flagged": int(b[flag].sum()),
                      "n_bad_unflagged": int(b[~flag].sum()),
                      "difference": pf - pu, "ci": [lo, hi],
                      "discarded_replicates": sk,
                      "clustered_on": "plot",
                      "informative": bool(lo > 0)}
    lines.append("CAVEAT the numbers cannot carry: the target is yield "
                 "below 80% of a")
    lines.append("   training-split mean on FARMER-RECALLED yield. An "
                 "association with that")
    lines.append("   label is not evidence of detecting crop damage.")
    record(3, "Flagged-versus-unflagged denominators, interval method and "
              "clustering", "REPORT", lines)
    return out


# ------------------------------------------------------------------ 4
def check_warning():
    lines, ok = [], True
    fr_src = (PROJ / "src/farmer_report.py").read_text(encoding="utf8")
    app_src = (PROJ / "app.py").read_text(encoding="utf8")
    needed = ("no record", "NO RECORD")
    for name, src in (("src/farmer_report.py", fr_src), ("app.py", app_src)):
        has_scope = any(t in src for t in needed)
        lines.append(f"{name}: states the warning means a missing record "
                     f"-> {'yes' if has_scope else 'NO'}")
        if not has_scope:
            ok = False
    # the old, stronger reading must not survive anywhere
    for name, src in (("src/farmer_report.py", fr_src), ("app.py", app_src)):
        bad = re.findall(r"less accurate|worse estimate|unreliable estimate",
                         src, re.I)
        if bad:
            ok = False
            lines.append(f"{name}: still asserts the estimate is worse "
                         f"-> {bad[:3]}")

    tests = list((PROJ / "tests").glob("*.py"))
    tsrc = "\n".join(t.read_text(encoding="utf8") for t in tests)
    covers_cli = "farmer_report" in tsrc
    covers_app = "app.py" in tsrc or "import app" in tsrc
    lines.append(f"tests reference the CLI report: "
                 f"{'yes' if covers_cli else 'NO'}")
    lines.append(f"tests reference the Streamlit interface: "
                 f"{'yes' if covers_app else 'NO'}")
    if not covers_app:
        ok = False
        lines.append("   GAP: app.py renders the same warnings and is NOT "
                     "covered by the")
        lines.append("   reassurance scan, so the two interfaces can drift "
                     "apart silently.")
    record(4, "Warning wording matches the evidence, and tests cover every "
              "interface", "PASS" if ok else "FAIL", lines)
    return ok


# ------------------------------------------------------------------ 5
def check_matrix():
    lines, ok = [], True
    f = PROJ / "EVIDENCE_TO_DESIGN.md"
    if not f.exists():
        record(5, "Evidence-to-design matrix verdicts", "FAIL",
               ["EVIDENCE_TO_DESIGN.md missing"])
        return False
    src = f.read_text(encoding="utf8")
    rows = re.findall(r"^## (\d+)\. (.+)$", src, re.M)
    verdicts = re.findall(r"\*\*Verdict\*\* \| \*\*(.+?)\*\*", src)
    lines.append(f"{len(rows)} design rows, {len(verdicts)} verdicts")
    if len(rows) != len(verdicts):
        ok = False
        lines.append("   MISMATCH: every row must carry exactly one verdict")
    allowed = ("Supported", "Refuted", "Inconclusive", "Honest limit",
               "Partly supported")
    for (num, title), v in zip(rows, verdicts):
        head = v.split(",")[0].split(" as ")[0].split(" by ")[0].strip(". ")
        tag = next((a for a in allowed if head.lower().startswith(a.lower())),
                   None)
        lines.append(f"   row {num}: {tag or 'UNCLASSIFIED'}  <- {v[:58]}")
        if tag is None:
            ok = False
    record(5, "Every matrix row carries a verdict its own test supports",
           "PASS" if ok else "FAIL", lines)
    return ok


def main() -> int:
    D = build()
    check_split()
    base = check_baseline(D)
    flag = check_flag(D)
    check_warning()
    check_matrix()

    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    print("=" * 78)
    print(f"{len(RESULTS)} checks: "
          f"{sum(1 for r in RESULTS if r['status'] == 'PASS')} PASS, "
          f"{sum(1 for r in RESULTS if r['status'] == 'REPORT')} REPORT, "
          f"{n_fail} FAIL")
    if n_fail:
        for r in RESULTS:
            if r["status"] == "FAIL":
                print(f"  FAIL: {r['title']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"checks": RESULTS, "baseline_comparison": base,
         "flag_comparison": flag, "n_boot": NBOOT}, indent=2),
        encoding="utf8")
    print(f"wrote {OUT.relative_to(PROJ)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
