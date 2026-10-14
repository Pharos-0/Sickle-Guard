"""Does the prototype work as a DECISION PROCEDURE, not just as software?

    python src/evaluate_prototype.py

WHY THIS FILE EXISTS
--------------------
tests/ proves that SICKLE-Guard behaves as specified: the report never
reassures, the thresholds are applied as written, the numpy models match
scikit-learn. That is a statement about the software. It is not a statement
about whether using the software leads to better decisions than not using
it, and the two are routinely confused.

This script asks the second question, on the held-out test split only --
plots that were never used to fit anything, choose anything or tune
anything. It compares the prototype against the alternatives a verification
officer could actually use instead, including several that need no model at
all.

WHAT IT CANNOT ASK, AND SAYS SO
-------------------------------
No farmer and no officer has ever used this. There is no measurement of
comprehension, trust, time saved, claims filed, claims paid, or any outcome
for any person. Every number below is about ranking and error on recorded
data. The gap between "the ranking is better" and "people are better off"
is not bridged here and cannot be bridged without a field study. The
closing section of the output states that explicitly rather than leaving it
to be inferred.

THE THREE QUESTIONS
-------------------
  A  Under a fixed verification budget, does the prototype find more
     below-threshold fields than the alternatives?
  B  When the report says it has no reliable estimate, is the error
     actually worse? A warning that does not separate error is noise.
  C  What does the no-reassurance rule cost? If every field carries a
     warning, does the flag still carry information on top of it?

Writes results/metrics/evaluate_prototype.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import farmer_report as fr  # noqa: E402

OUT = PROJ / "results" / "metrics" / "evaluate_prototype.json"
LEVEL = 0.80          # below this share of the training mean counts as a loss
BUDGETS = (0.25, 0.33, 0.50)
NBOOT = 10000
SEED = 0


# ----------------------------------------------------------------- stats
def auroc(score, pos):
    """Higher score should mean more likely below threshold. Ties count half."""
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


def at_budget(score, bad, frac):
    n = len(score)
    k = max(1, int(round(frac * n)))
    sel = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")[:k]
    m = np.zeros(n, dtype=bool)
    m[sel] = True
    caught = int((m & bad).sum())
    return {"n_visited": k, "budget_realised": k / n, "n_caught": caught,
            "recall": caught / max(1, int(bad.sum())),
            "precision": caught / k}


def boot_plots(stat, plots, n_boot=NBOOT, seed=SEED):
    """Resample PLOTS, carrying all of a plot's seasons together.

    Seasons at one plot share soil, geometry and management. Resampling
    field-seasons independently would understate every interval here.
    """
    uniq = np.unique(plots)
    pos = {g: np.where(plots == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        idx = np.concatenate([pos[g] for g in pick])
        r = stat(idx)
        if r is not None and np.isfinite(r):
            v.append(float(r))
    if not v:
        return (float("nan"), float("nan"))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ------------------------------------------------------------------ data
def build():
    feats = fr.load()
    hist = fr.causal_history(feats)
    # fr.satellite() fits on train+val, so its predictions for those rows are
    # in-sample. That is correct for the shipped product, which only ever
    # predicts rows it was not fitted on, but it makes any comparison that
    # INCLUDES train or val rows meaningless: a model scored on its own
    # training data wins by memorising it. The secondary analysis over all
    # field-seasons therefore uses the grouped-CV out-of-fold predictions
    # from src/grouped_cv.py, where every row was predicted by a model that
    # never saw its plot.
    sat = fr.satellite(feats)
    oof_f = PROJ / "results/predictions/oof_pred.npy"
    sat_oof = np.load(oof_f) if oof_f.exists() else None
    if sat_oof is not None and len(sat_oof) != len(feats):
        raise RuntimeError(
            f"oof_pred.npy has {len(sat_oof)} rows for {len(feats)} "
            f"field-seasons; refusing to align them by assumption")
    y = np.array([r["yield"] for r in feats])
    plots = np.array([r["plot"] for r in feats])
    area = np.array([float(r["area"]) if r.get("area") not in (None, "")
                     else np.nan for r in feats])
    split = np.array([r["split"] for r in feats])
    bench = float(y[split == "train"].mean())

    # the prototype as shipped: the field's own past harvests where they
    # exist, the satellite guess where they do not
    guard = np.where(np.isfinite(hist), hist, sat)
    cold = ~np.isfinite(hist)

    # the single most recent prior harvest, the simplest record-based rule
    byplot = defaultdict(list)
    for i, r in enumerate(feats):
        byplot[r["plot"]].append(i)
    last = np.full(len(feats), np.nan)
    for _, v in byplot.items():
        for i in v:
            prior = [j for j in v if feats[j]["harv"] < feats[i]["sow"]]
            if prior:
                j = max(prior, key=lambda k: feats[k]["harv"])
                last[i] = y[j]

    guard_oof = (np.where(np.isfinite(hist), hist, sat_oof)
                 if sat_oof is not None else None)
    return dict(feats=feats, y=y, plots=plots, area=area, split=split,
                bench=bench, hist=hist, sat=sat, sat_oof=sat_oof,
                guard=guard, guard_oof=guard_oof, cold=cold, last=last)


def rankings(D, idx, rng, oof=False):
    """Every ranking an officer could use. Higher score = visit sooner.

    `oof=True` swaps the satellite component for grouped-CV out-of-fold
    predictions, which is the only honest option when the rows being scored
    include the ones the satellite model was fitted on.
    """
    g = lambda a: -np.asarray(a)[idx]           # noqa: E731  low yield first
    sat = D["sat_oof"] if (oof and D["sat_oof"] is not None) else D["sat"]
    guard = (D["guard_oof"] if (oof and D["guard_oof"] is not None)
             else D["guard"])
    out = {
        "SICKLE-Guard (as shipped)": g(guard),
        "history only, no imagery": g(np.where(np.isfinite(D["hist"]),
                                               D["hist"], D["bench"])),
        "satellite only": g(sat),
        "last harvest only": g(np.where(np.isfinite(D["last"]),
                                        D["last"], D["bench"])),
        "smallest field first": -D["area"][idx],
        "largest field first": D["area"][idx],
        "arbitrary order": rng.permutation(len(idx)).astype(float),
    }
    for k, v in out.items():
        v = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(v)):
            v = np.where(np.isfinite(v), v, np.nanmin(v[np.isfinite(v)]) - 1)
        out[k] = v
    return out


# ------------------------------------------------------------------- A
def question_a(D, idx, label, res, oof=False):
    y, plots = D["y"][idx], D["plots"][idx]
    bad = y < LEVEL * D["bench"]
    rng = np.random.default_rng(SEED)
    rk = rankings(D, idx, rng, oof=oof)

    bar = "=" * 78
    print(bar)
    print(f"A. VERIFICATION PRIORITISATION on {label}")
    print(f"   {len(idx)} field-seasons at {len(np.unique(plots))} plots; "
          f"{int(bad.sum())} below {LEVEL:.0%} of the training mean "
          f"({LEVEL * D['bench']:.0f} kg/acre)")
    print("   satellite component: "
          + ("grouped-CV out-of-fold, so no row was predicted by a model "
             "that saw its plot" if oof else
             "the shipped model, which never saw these rows"))
    print(bar)
    if not (0 < bad.sum() < len(idx)):
        print("   degenerate: no contrast to score on this subset")
        return
    print(f"{'ranking':<28s}{'AUROC':>8s}{'95% CI':>18s}"
          f"{'rec@33%':>9s}{'prec':>7s}{'vs random':>11s}")

    entry = {}
    for name, s in rk.items():
        a = auroc(s, bad)
        lo, hi = boot_plots(
            lambda i, s=s: (auroc(s[i], bad[i])
                            if 0 < bad[i].sum() < len(i) else None), plots)
        b33 = at_budget(s, bad, 0.33)
        # paired excess over what random visiting would catch, inside each
        # bootstrap replicate, because both move together under resampling
        elo, ehi = boot_plots(
            lambda i, s=s: (at_budget(s[i], bad[i], 0.33)["recall"]
                            - at_budget(s[i], bad[i], 0.33)["budget_realised"]
                            if bad[i].sum() else None), plots)
        entry[name] = {
            "auroc": a, "auroc_ci": [lo, hi],
            "informative": bool(lo > 0.5),
            "budgets": {str(b): at_budget(s, bad, b) for b in BUDGETS},
            "recall_excess_at_33": b33["recall"] - b33["budget_realised"],
            "recall_excess_ci": [elo, ehi],
            "beats_random": bool(elo > 0)}
        print(f"{name:<28s}{a:>8.3f}{f'[{lo:.2f}, {hi:.2f}]':>18s}"
              f"{b33['recall']:>9.2f}{b33['precision']:>7.2f}"
              f"{('yes' if elo > 0 else 'no'):>11s}")
    print("   'vs random' is the paired excess of recall over arbitrary "
          "visiting at the")
    print("   same realised budget, with the interval excluding zero.")
    res[label] = entry

    guard = entry["SICKLE-Guard (as shipped)"]
    better = [k for k, v in entry.items()
              if k != "SICKLE-Guard (as shipped)" and v["auroc"] > guard["auroc"]]
    print()
    if better:
        print(f"   NOTE: {', '.join(better)} rank at least as well as the "
              f"prototype here.")
    if not guard["beats_random"]:
        print("   The prototype does NOT beat arbitrary visiting at the "
              "interval level on this subset.")


# ------------------------------------------------------------------- B
def question_b(D, idx, label, res):
    """Is the 'no reliable estimate' warning calibrated?"""
    err = np.abs(D["guard"] - D["y"])[idx]
    cold = D["cold"][idx]
    plots = D["plots"][idx]
    bar = "=" * 78
    print()
    print(bar)
    print(f"B. IS THE WARNING CALIBRATED? ({label})")
    print(bar)
    print("   The report tells a farmer there is NO RELIABLE ESTIMATE when "
          "the field has")
    print("   no past harvest on record. If that warning does not separate "
          "real error,")
    print("   it is noise, and a warning that is noise erodes the ones that "
          "are not.")
    if cold.sum() == 0 or (~cold).sum() == 0:
        print(f"   not applicable here: {int(cold.sum())} warned, "
              f"{int((~cold).sum())} not warned")
        return
    d = float(err[cold].mean() - err[~cold].mean())
    lo, hi = boot_plots(
        lambda i: (float(err[i][cold[i]].mean() - err[i][~cold[i]].mean())
                   if cold[i].any() and (~cold[i]).any() else None), plots)
    print(f"   warned (no history)   n={int(cold.sum()):3d}   mean |error| "
          f"{err[cold].mean():7.1f} kg/acre")
    print(f"   not warned            n={int((~cold).sum()):3d}   mean |error| "
          f"{err[~cold].mean():7.1f} kg/acre")
    print(f"   difference            {d:+7.1f}   95% CI "
          f"[{lo:+.1f}, {hi:+.1f}]")
    ok = lo > 0
    print(f"   -> the warning {'IS' if ok else 'is NOT'} calibrated at the "
          f"interval level on this subset")
    res.setdefault("warning_calibration", {})[label] = {
        "n_warned": int(cold.sum()), "n_not_warned": int((~cold).sum()),
        "mae_warned": float(err[cold].mean()),
        "mae_not_warned": float(err[~cold].mean()),
        "difference": d, "ci": [lo, hi], "calibrated": bool(ok)}


# ------------------------------------------------------------------- C
def question_c(D, idx, label, res):
    """What does the no-reassurance rule cost, and does the flag still say
    anything on top of a warning everybody gets?"""
    y, plots = D["y"][idx], D["plots"][idx]
    bad = y < LEVEL * D["bench"]
    s = -D["guard"][idx]
    k = max(1, int(round(0.33 * len(idx))))
    flag = np.zeros(len(idx), dtype=bool)
    flag[np.argsort(-s, kind="mergesort")[:k]] = True

    bar = "=" * 78
    print()
    print(bar)
    print(f"C. WHAT THE NO-REASSURANCE RULE COSTS ({label})")
    print(bar)
    print(f"   Every one of the {len(idx)} reports carries a warning: a "
          f"flagged field is")
    print("   told to look, an unflagged field is told this is not evidence "
          "the crop is")
    print("   fine. Nobody is told they are safe. The question is whether "
          "the flag adds")
    print("   anything on top of a warning everyone already has.")
    if not (0 < bad.sum() < len(idx)):
        print("   degenerate on this subset")
        return
    pf = float(bad[flag].mean()) if flag.any() else float("nan")
    pu = float(bad[~flag].mean()) if (~flag).any() else float("nan")
    lo, hi = boot_plots(
        lambda i: (float(bad[i][flag[i]].mean() - bad[i][~flag[i]].mean())
                   if flag[i].any() and (~flag[i]).any() else None), plots)
    print(f"   flagged      {int(flag.sum()):3d} fields, "
          f"{pf:.0%} actually below threshold")
    print(f"   not flagged  {int((~flag).sum()):3d} fields, "
          f"{pu:.0%} actually below threshold")
    print(f"   difference   {pf - pu:+.0%}   95% CI [{lo:+.0%}, {hi:+.0%}]")
    miss = int((bad & ~flag).sum())
    print(f"   MISSED: {miss} of {int(bad.sum())} below-threshold fields are "
          f"NOT flagged.")
    print("   That is the number the no-reassurance rule exists for. A "
          "farmer in that")
    print("   group who read a reassuring message would have had no reason "
          "to act.")
    res.setdefault("flag_adds_information", {})[label] = {
        "n_flagged": int(flag.sum()), "share_bad_flagged": pf,
        "share_bad_unflagged": pu, "difference": pf - pu, "ci": [lo, hi],
        "informative": bool(lo > 0),
        "n_missed": miss, "n_bad": int(bad.sum())}


# ---------------------------------------------------------------- report
def main() -> int:
    D = build()
    res: dict = {"ranking": {}}
    test = np.where(D["split"] == "test")[0]
    allidx = np.arange(len(D["y"]))

    print("SICKLE-Guard evaluated as a decision procedure, not as software.")
    print(f"Loss definition: yield below {LEVEL:.0%} of the TRAINING-split "
          f"mean = {LEVEL * D['bench']:.0f} kg/acre.")
    print("That is a relative cut on farmer-recalled yield, not a "
          "measurement of damage.")
    print()

    question_a(D, test, "held-out test split", res["ranking"])
    question_a(D, allidx, "all field-seasons, out-of-fold (secondary, "
                          "more power)", res["ranking"], oof=True)
    question_b(D, test, "held-out test split", res)
    question_b(D, allidx, "all field-seasons", res)
    question_c(D, test, "held-out test split", res)
    question_c(D, allidx, "all field-seasons", res)

    bar = "=" * 78
    print()
    print(bar)
    print("WHAT THIS EVALUATION DOES NOT ESTABLISH")
    print(bar)
    not_done = [
        "No farmer has used this. Comprehension, trust and whether the "
        "wording changes behaviour are all unmeasured.",
        "No verification officer has used this. Time saved, visits "
        "reordered and decisions changed are all unmeasured.",
        "No outcome for any person is measured: no claim filed, no claim "
        "paid, no income, no yield.",
        "The target is a relative cut on farmer-recalled yield. It is not "
        "crop damage, and src/label_validity.py shows 60 % of the labelled "
        "losses sit at plots that are below threshold every season.",
        "The test split is 27 plots. Every interval here is wide and none "
        "of them establishes an operating threshold.",
        "Ranking better than an alternative is not the same as people "
        "being better off. Nothing here bridges that gap, and only a field "
        "study can.",
    ]
    for i, t in enumerate(not_done, 1):
        print(f"  {i}. {t}")
    res["not_established"] = not_done
    res["loss_definition"] = {
        "level": LEVEL, "threshold_kg_per_acre": LEVEL * D["bench"],
        "note": ("relative cut on the training-split mean of "
                 "farmer-recalled yield; not a measurement of crop damage")}
    res["n_boot"] = NBOOT
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf8")
    print()
    print(f"wrote {OUT.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
