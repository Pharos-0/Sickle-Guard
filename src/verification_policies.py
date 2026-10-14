"""Does the workflow make verification more effective than what a team has?

    python src/verification_policies.py

Executes configs/verification_protocol.yaml, frozen before any result. The
protocol is loaded, not restated, and the run aborts on any disagreement.

Two experiments, reported separately:

  A  four policies on the same held-out plots at three budgets, with the
     register a verification office already keeps as the competitor that
     matters, not random referral
  B  reserving part of the budget for fields with no record, and the
     trade-off that buys

The target throughout is "below the study's yield threshold" on
farmer-recalled yield. It is not crop damage and not an insurance claim,
and nothing here is reported as either.

Writes results/metrics/verification_policies.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import farmer_report as fr  # noqa: E402

PROTOCOL = PROJ / "configs" / "verification_protocol.yaml"
OUT = PROJ / "results" / "metrics" / "verification_policies.json"
LEVEL = 0.80


def load_protocol():
    if not PROTOCOL.exists():
        raise SystemExit(f"frozen protocol missing: {PROTOCOL}")
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf8"))
    if not p.get("frozen_before_results"):
        raise SystemExit("protocol is not marked frozen_before_results")
    if p["experiment_a"]["budgets"] != [0.10, 0.20, 0.33]:
        raise SystemExit("experiment A budgets differ from this script")
    if "0.80" not in p["target"]["definition"]:
        raise SystemExit("target threshold is not 0.80")
    return p


# ------------------------------------------------------------------ data
def build():
    feats = fr.load()
    hist = fr.causal_history(feats)
    sat_ship = fr.satellite(feats)
    oof = np.load(PROJ / "results/predictions/oof_pred.npy")
    if len(oof) != len(feats):
        raise SystemExit("oof_pred.npy is misaligned with the feature table")
    y = np.array([r["yield"] for r in feats])
    plots = np.array([r["plot"] for r in feats])
    split = np.array([r["split"] for r in feats])
    bench = float(y[split == "train"].mean())

    # causal minimum of prior harvests: what a register can know
    byplot = defaultdict(list)
    for i, r in enumerate(feats):
        byplot[r["plot"]].append(i)
    reg_min = np.full(len(feats), np.nan)
    for _, v in byplot.items():
        for i in v:
            prior = [j for j in v if feats[j]["harv"] < feats[i]["sow"]]
            if prior:
                reg_min[i] = float(np.min(y[prior]))
    return dict(y=y, plots=plots, split=split, bench=bench, hist=hist,
                reg_min=reg_min, sat_ship=sat_ship, sat_oof=oof,
                bad=y < LEVEL * bench, has_rec=np.isfinite(hist))


def policies(D, idx, oof: bool):
    """Higher score = inspect sooner. All defined as in the protocol."""
    rng = np.random.default_rng(0)
    sat = D["sat_oof"] if oof else D["sat_ship"]
    has = D["has_rec"][idx]
    n = len(idx)
    # register: recorded fields by lowest prior yield; unrecorded after all
    # recorded ones, in a fixed random order
    reg = np.where(has, -D["reg_min"][idx], -np.inf)
    cold_order = rng.permutation(n).astype(float)
    reg = np.where(has, reg, np.nanmin(np.where(has, reg, np.nan)) - 1e6
                   - cold_order) if has.any() else -cold_order
    hist_only = -np.where(has, D["hist"][idx], D["bench"])
    workflow = -np.where(has, D["hist"][idx], sat[idx])
    return {"register": reg, "history_only": hist_only,
            "workflow": workflow}


def visit(score, budget):
    n = len(score)
    k = max(1, int(round(budget * n)))
    sel = np.zeros(n, dtype=bool)
    sel[np.argsort(-np.asarray(score, float), kind="mergesort")[:k]] = True
    return sel


def outcome(sel, bad, has):
    return {"n_inspections": int(sel.sum()),
            "n_found": int((sel & bad).sum()),
            "n_missed": int((bad & ~sel).sum()),
            "recall": float((sel & bad).sum() / max(1, bad.sum())),
            "n_missed_with_record": int((bad & ~sel & has).sum()),
            "n_missed_cold_start": int((bad & ~sel & ~has).sum())}


def boot_pairs(fn, plots, n_boot, seed=0):
    uniq = np.unique(plots)
    pos = {g: np.where(plots == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        j = np.concatenate([pos[g] for g in pick])
        r = fn(j)
        if r is not None and np.isfinite(r):
            v.append(float(r))
    return np.array(v)


def interval(v, level):
    a = (1 - level) / 2
    return [float(np.percentile(v, 100 * a)),
            float(np.percentile(v, 100 * (1 - a)))]


# ------------------------------------------------------------ experiment A
def experiment_a(D, P, label, idx, oof):
    nb = P["bootstrap"]["replicates"]
    budgets = P["experiment_a"]["budgets"]
    adj = 1 - 0.05 / 6
    bad, has, plots = D["bad"][idx], D["has_rec"][idx], D["plots"][idx]
    pol = policies(D, idx, oof)
    bar = "=" * 86
    print(bar)
    print(f"EXPERIMENT A -- {label}")
    print(f"   {len(idx)} field-seasons at {len(np.unique(plots))} plots; "
          f"{int(bad.sum())} below threshold, {int((bad & ~has).sum())} of "
          f"them with no record")
    print(bar)
    rng = np.random.default_rng(0)
    out = {"n": len(idx), "n_bad": int(bad.sum()),
           "n_bad_cold_start": int((bad & ~has).sum()), "budgets": {}}

    print(f"{'budget':>7s} {'policy':<14s}{'inspect':>8s}{'found':>7s}"
          f"{'missed':>8s}{'(rec/cold)':>12s}{'recall':>8s}")
    for b in budgets:
        row = {}
        # random as a distribution
        rr = []
        for _ in range(nb):
            sel = visit(rng.permutation(len(idx)).astype(float), b)
            rr.append(outcome(sel, bad, has)["recall"])
        rr = np.array(rr)
        row["random"] = {"recall_mean": float(rr.mean()),
                         "recall_p97_5": float(np.percentile(rr, 97.5))}
        print(f"{b:>7.2f} {'random':<14s}{'':>8s}{'':>7s}{'':>8s}"
              f"{'':>12s}{rr.mean():>8.2f}  (p97.5 {np.percentile(rr, 97.5):.2f})")
        for name in ("register", "history_only", "workflow"):
            o = outcome(visit(pol[name], b), bad, has)
            row[name] = o
            split_miss = (f"({o['n_missed_with_record']}/"
                          f"{o['n_missed_cold_start']})")
            print(f"{'':>7s} {name:<14s}{o['n_inspections']:>8d}"
                  f"{o['n_found']:>7d}{o['n_missed']:>8d}"
                  f"{split_miss:>12s}{o['recall']:>8.2f}")
        # paired differences, workflow minus each simple alternative
        cmp = {}
        for other in ("register", "history_only"):
            def stat(j, other=other, b=b):
                if bad[j].sum() == 0:
                    return None
                pj = policies(D, idx[j], oof)
                rw = outcome(visit(pj["workflow"], b), bad[j],
                             has[j])["recall"]
                ro = outcome(visit(pj[other], b), bad[j], has[j])["recall"]
                return rw - ro
            v = boot_pairs(stat, plots, nb)
            d = row["workflow"]["recall"] - row[other]["recall"]
            ci95, ciadj = interval(v, 0.95), interval(v, adj)
            cmp[other] = {"delta": d, "ci95": ci95, "ci_adjusted": ciadj,
                          "adjusted_level": adj,
                          "better": bool(ciadj[0] > 0),
                          "worse": bool(ciadj[1] < 0)}
            print(f"{'':>7s}   workflow - {other:<12s} {d:+.3f}  95% "
                  f"[{ci95[0]:+.3f}, {ci95[1]:+.3f}]  adj "
                  f"[{ciadj[0]:+.3f}, {ciadj[1]:+.3f}]  -> "
                  + ("BETTER" if ciadj[0] > 0 else
                     "WORSE" if ciadj[1] < 0 else "not separable"))
        row["comparisons"] = cmp
        out["budgets"][str(b)] = row
        print()

    better_both = [b for b in budgets
                   if all(out["budgets"][str(b)]["comparisons"][o]["better"]
                          for o in ("register", "history_only"))]
    worse_any = [b for b in budgets
                 if any(out["budgets"][str(b)]["comparisons"][o]["worse"]
                        for o in ("register", "history_only"))]
    ok = bool(better_both) and not worse_any
    out["verdict"] = ("advantage demonstrated" if ok
                      else "advantage NOT demonstrated")
    out["better_than_both_at"] = better_both
    out["worse_than_either_at"] = worse_any
    print(f"VERDICT ({label}): {out['verdict'].upper()}")
    print(f"   better than BOTH register and history-only at budgets: "
          f"{better_both or 'none'}")
    print(f"   significantly worse than either at budgets: "
          f"{worse_any or 'none'}")
    print()
    return out


# ------------------------------------------------------------ experiment B
def experiment_b(D, P, label, idx, oof):
    nb = P["bootstrap"]["replicates"]
    bad, has, plots = D["bad"][idx], D["has_rec"][idx], D["plots"][idx]
    sat = (D["sat_oof"] if oof else D["sat_ship"])[idx]
    hist = D["hist"][idx]
    n = len(idx)
    share_cold = float((~has).mean())
    fracs = []
    for f in P["experiment_b"]["reserve_fraction"]:
        fracs.append(("proportional", share_cold) if f == "proportional"
                     else (f"{float(f):.2f}", float(f)))
    rng = np.random.default_rng(0)
    rand_cold = rng.permutation(n).astype(float)

    def allocate(budget, frac, cold_key, sub=None):
        """Reserve frac of the budget for cold starts; history for the rest."""
        ix = np.arange(n) if sub is None else sub
        h, b_, hs = has[ix], bad[ix], hist[ix]
        k = max(1, int(round(budget * len(ix))))
        k_cold = min(int(round(frac * k)), int((~h).sum()))
        k_rec = min(k - k_cold, int(h.sum()))
        k_cold = min(k - k_rec, int((~h).sum()))   # spill unused capacity
        rec_ix = np.where(h)[0]
        cold_ix = np.where(~h)[0]
        rec_pick = rec_ix[np.argsort(hs[rec_ix], kind="mergesort")[:k_rec]]
        ck = cold_key[ix][cold_ix]
        cold_pick = cold_ix[np.argsort(ck, kind="mergesort")[:k_cold]]
        sel = np.zeros(len(ix), dtype=bool)
        sel[rec_pick] = True
        sel[cold_pick] = True
        found_rec = int((sel & b_ & h).sum())
        found_cold = int((sel & b_ & ~h).sum())
        return {"recall_recorded": found_rec / max(1, int((b_ & h).sum())),
                "recall_cold_start": found_cold / max(1, int((b_ & ~h).sum())),
                "recall_total": (found_rec + found_cold) / max(1, int(b_.sum())),
                "n_found_recorded": found_rec,
                "n_found_cold_start": found_cold,
                "n_cold_inspected": int(k_cold), "n_rec_inspected": int(k_rec)}

    bar = "=" * 86
    print(bar)
    print(f"EXPERIMENT B -- {label}")
    print(f"   {n} field-seasons; {int((bad & has).sum())} below threshold "
          f"with a record, {int((bad & ~has).sum())} without; cold starts "
          f"are {share_cold:.0%} of all field-seasons")
    print(bar)
    out = {"n": n, "n_bad_recorded": int((bad & has).sum()),
           "n_bad_cold_start": int((bad & ~has).sum()),
           "share_cold": share_cold, "budgets": {}}
    for budget in P["experiment_b"]["budgets"]:
        print(f"budget {budget:.2f}")
        print(f"   {'reserve':<14s}{'order':<11s}{'cold insp':>10s}"
              f"{'rec recall':>12s}{'cold recall':>13s}{'total':>8s}")
        rows = {}
        for lab, f in fracs:
            for order, key in (("satellite", sat), ("random", rand_cold)):
                r = allocate(budget, f, key)
                rows[f"{lab}|{order}"] = r
                print(f"   {lab:<14s}{order:<11s}{r['n_cold_inspected']:>10d}"
                      f"{r['recall_recorded']:>12.2f}"
                      f"{r['recall_cold_start']:>13.2f}"
                      f"{r['recall_total']:>8.2f}")
        # dominance: is any allocation at least as good on both axes?
        sat_rows = {k: v for k, v in rows.items() if k.endswith("satellite")}
        dom = []
        for k, v in sat_rows.items():
            if all(v["recall_recorded"] >= o["recall_recorded"] - 1e-12
                   and v["recall_cold_start"] >= o["recall_cold_start"] - 1e-12
                   for kk, o in sat_rows.items() if kk != k):
                dom.append(k.split("|")[0])
        # satellite versus random ordering inside the cold-start pool, at the
        # proportional reserve, paired on plots
        def stat(j, budget=budget):
            if (bad[j] & ~has[j]).sum() == 0:
                return None
            s1 = allocate(budget, share_cold, sat, sub=j)["recall_cold_start"]
            s2 = allocate(budget, share_cold, rand_cold, sub=j)["recall_cold_start"]
            return s1 - s2
        v = boot_pairs(stat, plots, nb)
        d = (rows["proportional|satellite"]["recall_cold_start"]
             - rows["proportional|random"]["recall_cold_start"])
        ci = interval(v, 0.95)
        rows["_within_cold"] = {"delta": d, "ci95": ci,
                                "satellite_helps": bool(ci[0] > 0)}
        rows["_dominant"] = dom
        print(f"   satellite minus random ordering within cold starts "
              f"(proportional reserve): {d:+.3f}  95% [{ci[0]:+.3f}, "
              f"{ci[1]:+.3f}] -> "
              + ("satellite helps" if ci[0] > 0 else "NOT demonstrated"))
        print(f"   allocation at least as good on both axes: "
              f"{dom if dom else 'none -- the result is the trade-off'}")
        print()
        out["budgets"][str(budget)] = rows
    return out


def exploratory_reserve(D, P, idx, oof):
    """EXPLORATORY, not pre-specified: is a small cold-start reserve free?

    Experiment B's table suggested that at a 33 % budget a 10 % reserve
    gave up no recorded recall while finding some cold-start losses. That
    was noticed AFTER the results were seen, so it is tested here as an
    exploratory question, with its own label, and it does not alter either
    pre-registered verdict.
    """
    nb = P["bootstrap"]["replicates"]
    bad, has, plots = D["bad"][idx], D["has_rec"][idx], D["plots"][idx]
    hist = D["hist"][idx]
    rng = np.random.default_rng(0)
    rand_cold = rng.permutation(len(idx)).astype(float)

    def pick(sub, budget, frac):
        h, hs = has[sub], hist[sub]
        k = max(1, int(round(budget * len(sub))))
        k_cold = min(int(round(frac * k)), int((~h).sum()))
        k_rec = min(k - k_cold, int(h.sum()))
        k_cold = min(k - k_rec, int((~h).sum()))
        rec_ix = np.where(h)[0]
        cold_ix = np.where(~h)[0]
        sel = np.zeros(len(sub), dtype=bool)
        sel[rec_ix[np.argsort(hs[rec_ix], kind="mergesort")[:k_rec]]] = True
        ck = rand_cold[sub][cold_ix]
        sel[cold_ix[np.argsort(ck, kind="mergesort")[:k_cold]]] = True
        return sel

    def metrics(sub, budget, frac):
        sel = pick(sub, budget, frac)
        b_, h = bad[sub], has[sub]
        return ((sel & b_).sum() / max(1, b_.sum()),
                (sel & b_ & h).sum() / max(1, (b_ & h).sum()))

    out = {}
    print("=" * 86)
    print("EXPLORATORY (noticed after the results; not pre-registered)")
    print("is a 10 % cold-start reserve free? random order within cold "
          "starts, all field-seasons out-of-fold")
    print("=" * 86)
    for budget in (0.20, 0.33):
        res = {}
        for lab, ix in (("total recall", 0), ("recorded recall", 1)):
            def stat(j, budget=budget, ix=ix):
                sub = idx[j] if False else j
                if bad[sub].sum() == 0 or (bad[sub] & has[sub]).sum() == 0:
                    return None
                return (metrics(sub, budget, 0.10)[ix]
                        - metrics(sub, budget, 0.0)[ix])
            v = boot_pairs(stat, plots, nb)
            full = np.arange(len(idx))
            d = metrics(full, budget, 0.10)[ix] - metrics(full, budget, 0.0)[ix]
            ci = interval(v, 0.95)
            res[lab] = {"delta": float(d), "ci95": ci}
            print(f"   budget {budget:.2f}  {lab:<16s} reserve 10% minus "
                  f"none: {d:+.3f}  95% [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        out[str(budget)] = res
    print("   Read as: where the recorded-recall interval sits at or near "
          "zero, a small")
    print("   reserve costs little on recurring problems. Exploratory, so "
          "it informs the")
    print("   next protocol and is not a finding.")
    print()
    return out


def main() -> int:
    P = load_protocol()
    D = build()
    test = np.where(D["split"] == "test")[0]
    allx = np.arange(len(D["y"]))
    print("Target: yield below 80 % of the training mean, on farmer-recalled "
          "yield.")
    print("This is the study's threshold, not crop damage and not an "
          "insurance claim.")
    print()
    res = {"protocol_version": P["version"],
           "protocol_frozen": str(P["frozen_on"]),
           "threshold_kg_per_acre": LEVEL * D["bench"]}
    res["experiment_a"] = {
        "held_out": experiment_a(D, P, "held-out test split (VERDICT)",
                                 test, oof=False),
        "all_oof": experiment_a(D, P, "all field-seasons, out-of-fold "
                                "(secondary)", allx, oof=True)}
    res["experiment_b"] = {
        "all_oof": experiment_b(D, P, "all field-seasons, out-of-fold "
                                "(PRIMARY for B)", allx, oof=True),
        "held_out": experiment_b(D, P, "held-out test split "
                                 "(UNDERPOWERED: 4 cold-start losses)",
                                 test, oof=False)}
    res["exploratory_reserve"] = {
        "label": "EXPLORATORY -- noticed after results, not pre-registered",
        "all_oof": exploratory_reserve(D, P, allx, oof=True)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf8")
    print(f"wrote {OUT.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
