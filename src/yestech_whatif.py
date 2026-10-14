"""Experiment D: a YES-TECH what-if. EVERY NUMBER HERE IS SIMULATED.

    python src/yestech_whatif.py

Executes experiment_d of configs/extension_protocol.yaml.

No YES-TECH output is public, so nothing here is a statement about how
accurate YES-TECH is. What can be worked out without it is the MECHANISM:
how an error in a technology yield estimate travels into an Insurance
Unit's claim under PMFBY's area approach, at the weightages states actually
use, and which re-verification rule recovers the units that end up wrongly
unpaid.

Sourced inputs (configs/extension_protocol.yaml, experiment_d.sources):
  claim      max(0, (TY - AY) / TY) x Sum Insured
  weights    30 % national minimum; Maharashtra 50 % and Madhya Pradesh
             100 % in 2025-26 (Lok Sabha Starred Q.25, 21 July 2026)
  error      up to 0.30, the largest IU-level nRMSE the January 2023
             YES-TECH manual permitted; reading it as a 1-sigma relative
             error is this study's ASSUMPTION
Assumed inputs, swept rather than fixed: crop-cutting error, bias, the share
of units in true loss, independence of the two errors.

Everything is in units of the threshold yield (TY = 1) and of Sum Insured.
No rupee figure is produced.

Writes results/metrics/yestech_whatif.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import yaml

PROJ = Path(__file__).resolve().parents[1]
PROTOCOL = PROJ / "configs" / "extension_protocol.yaml"
OUT = PROJ / "results" / "metrics" / "yestech_whatif.json"
POP_SD = 0.15


def load_protocol():
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf8"))
    if not p.get("frozen_before_results"):
        raise SystemExit("extension protocol is not marked frozen")
    d = p["experiment_d"]
    if d["label"] != "SIMULATED":
        raise SystemExit("experiment D must be labelled SIMULATED")
    if d["weights"] != [0.30, 0.50, 1.00]:
        raise SystemExit("weights differ from the sourced 30/50/100 %")
    if "0.15" not in d["guard"]["population"]:
        raise SystemExit("population spread differs from this script")
    return p


def used_yield(R, w, sd_t, bias, sd_c, rng):
    """The yield the scheme would use, in units of the threshold yield."""
    y_tech = R * (1.0 + bias + sd_t * rng.standard_normal(R.shape))
    y_cce = R * (1.0 + sd_c * rng.standard_normal(R.shape))
    return w * y_tech + (1.0 - w) * y_cce


def conditional_curves(d, rng):
    """For a unit truly short by s: chance it is paid nothing, and the share
    of its due claim it receives on average."""
    cc = d["conditional_curves"]
    n = int(cc["draws"])
    out = {}
    for w in d["weights"]:
        for sd_t in d["tech_error_sd"]:
            for bias in d["tech_bias"]:
                for sd_c in d["cce_error_sd"]:
                    key = f"w={w:.2f}|sd_t={sd_t:.2f}|bias={bias:+.2f}|sd_c={sd_c:.2f}"
                    row = {}
                    for s in cc["true_shortfall"]:
                        R = np.full(n, 1.0 - s)
                        Y = used_yield(R, w, sd_t, bias, sd_c, rng)
                        paid = np.maximum(0.0, 1.0 - Y)
                        row[f"{s:.2f}"] = {
                            "p_paid_nothing": float((Y >= 1.0).mean()),
                            "paid_over_due": float(paid.mean() / s)}
                    out[key] = row
    return out


def guard(d, rng):
    g = d["guard"]
    n = int(g["n_units"])
    prim = d["primary"]
    out = {}
    for share in g["loss_share"]:
        mu = 1.0 - POP_SD * NormalDist().inv_cdf(share)
        for w in d["weights"]:
            for sd_t in d["tech_error_sd"]:
                for bias in d["tech_bias"]:
                    sd_c = prim["cce_error_sd"]
                    R = rng.normal(mu, POP_SD, n)
                    Y = used_yield(R, w, sd_t, bias, sd_c, rng)
                    due = R < 1.0
                    paid = Y < 1.0
                    wrong_unpaid = due & ~paid
                    claim_due = np.maximum(0.0, 1.0 - R)
                    under0 = np.maximum(0.0, claim_due - np.maximum(0.0, 1.0 - Y)).sum()
                    recheck = R * (1.0 + sd_c * rng.standard_normal(n))
                    key = f"loss={share:.2f}|w={w:.2f}|sd_t={sd_t:.2f}|bias={bias:+.2f}"
                    cell = {"n_due": int(due.sum()),
                            "n_wrongly_unpaid": int(wrong_unpaid.sum()),
                            "n_wrongly_paid": int((~due & paid).sum()),
                            "budgets": {}}
                    for b in g["recheck_budget"]:
                        k = int(round(b * n))
                        order = {
                            "random": rng.permutation(n),
                            "near_threshold": np.concatenate([
                                np.where(~paid)[0][np.argsort(Y[~paid], kind="mergesort")],
                                np.where(paid)[0]]),
                            "lowest_estimate": np.argsort(Y, kind="mergesort")}
                        pol = {}
                        for name, o in order.items():
                            sel = np.zeros(n, dtype=bool)
                            sel[o[:k]] = True
                            Yf = np.where(sel, recheck, Y)
                            paid_f = Yf < 1.0
                            rec = int((wrong_unpaid & sel & paid_f).sum())
                            under1 = np.maximum(
                                0.0, claim_due - np.maximum(0.0, 1.0 - Yf)).sum()
                            pol[name] = {
                                "recovered": rec,
                                "recovered_share": rec / max(1, int(wrong_unpaid.sum())),
                                "rechecks_per_recovery": (k / rec) if rec else None,
                                "underpayment_removed_share":
                                    float(1.0 - under1 / under0) if under0 > 0 else None,
                                "correctly_unpaid_flipped_to_paid":
                                    int((~due & ~paid & sel & paid_f).sum()),
                                "due_units_unpaid_by_recheck":
                                    int((due & paid & sel & ~paid_f).sum())}
                        cell["budgets"][f"{b:.2f}"] = {"k": k, "policies": pol}
                    out[key] = cell
    return out


def verdict(d, G):
    prim = d["primary"]
    fails = []
    for w in d["weights"]:
        for sd_t in d["tech_error_sd"]:
            key = (f"loss={prim['loss_share']:.2f}|w={w:.2f}|sd_t={sd_t:.2f}"
                   f"|bias={prim['tech_bias']:+.2f}")
            for b, row in G[key]["budgets"].items():
                p = row["policies"]
                if not p["near_threshold"]["recovered"] > p["random"]["recovered"]:
                    fails.append(f"{key} budget {b}")
    return {"near_threshold_is_efficient_guard": not fails,
            "cells_failing": fails,
            "qualifier": "UNDER THE STATED ASSUMPTIONS; SIMULATED; says nothing "
                         "about YES-TECH's actual accuracy"}


def main() -> int:
    P = load_protocol()
    d = P["experiment_d"]
    rng = np.random.default_rng(P["bootstrap"]["seed"])
    print("EVERY NUMBER BELOW IS SIMULATED. No YES-TECH output was used; none "
          "is public.")
    print()
    curves = conditional_curves(d, rng)
    G = guard(d, rng)
    V = verdict(d, G)

    prim = d["primary"]
    print("P(unit paid nothing | truly 10 % below threshold), bias 0, CCE "
          "error 0.10")
    print(f"   {'tech error':<12s}" + "".join(f"{'w=' + format(w, '.0%'):>10s}"
                                             for w in d["weights"]))
    for sd_t in d["tech_error_sd"]:
        cells = [curves[f"w={w:.2f}|sd_t={sd_t:.2f}|bias=+0.00|sd_c=0.10"]
                 ["0.10"]["p_paid_nothing"] for w in d["weights"]]
        print(f"   {sd_t:<12.2f}" + "".join(f"{c:>10.2f}" for c in cells))
    print()
    print("Re-verifying 5 % of units (loss share 0.20, bias 0, CCE error 0.10): "
          "wrongly unpaid units recovered")
    for w in d["weights"]:
        for sd_t in d["tech_error_sd"]:
            key = (f"loss={prim['loss_share']:.2f}|w={w:.2f}|sd_t={sd_t:.2f}"
                   f"|bias=+0.00")
            c = G[key]
            p = c["budgets"]["0.05"]["policies"]
            print(f"   w={w:.0%} sd_t={sd_t:.2f}: {c['n_wrongly_unpaid']:>6d} wrongly "
                  f"unpaid; near-threshold recovers "
                  f"{p['near_threshold']['recovered_share']:.0%} "
                  f"({p['near_threshold']['rechecks_per_recovery']:.1f} checks each), "
                  f"random {p['random']['recovered_share']:.0%}, lowest-first "
                  f"{p['lowest_estimate']['recovered_share']:.0%}")
    print()
    print(f"VERDICT (simulated): near-threshold re-verification is the efficient "
          f"guard: {V['near_threshold_is_efficient_guard']}")
    res = {"label": "SIMULATED",
           "protocol_version": P["version"],
           "protocol_frozen": str(P["frozen_on"]),
           "units": "threshold yield = 1; claims as a fraction of Sum Insured",
           "population_sd": POP_SD,
           "conditional_curves": curves, "guard": G, "verdict": V}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf8")
    print(f"wrote {OUT.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
