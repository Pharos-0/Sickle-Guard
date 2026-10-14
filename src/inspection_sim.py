"""Experiment C: what survives when the inspector is imperfect?

    python src/inspection_sim.py

Executes experiment_c of configs/extension_protocol.yaml.

Experiment A scored a visit as revealing the truth. Real inspectors miss
problems and report ones that are not there. The rankings below are the
OBSERVED rankings from Experiment A, on the same plots; only the inspector
is SIMULATED, by two numbers:

  sensitivity    chance an inspected below-threshold field-season is
                 confirmed
  false alarm    chance an inspected field-season that is not below
                 threshold is reported as a problem anyway

Three things are reported:

  1  confirmed real problems and the share of confirmed findings that are
     real, for every policy, budget and inspector             (SIMULATED)
  2  the break-even sensitivity: how weak a record-guided inspector can be
     and still confirm as many real problems as a PERFECT inspector sent at
     random                                                   (SIMULATED)
  3  how many real misses carry an estimate that would look reassuring if
     a display said so                                         (OBSERVED)

Writes results/metrics/inspection_sim.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import verification_policies as vp  # noqa: E402

PROTOCOL = PROJ / "configs" / "extension_protocol.yaml"
OUT = PROJ / "results" / "metrics" / "inspection_sim.json"
POLICIES = ("register", "history_only", "workflow")


def load_protocol():
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf8"))
    if not p.get("frozen_before_results"):
        raise SystemExit("extension protocol is not marked frozen")
    C = p["experiment_c"]
    if C["label"] != "SIMULATED":
        raise SystemExit("experiment C must be labelled SIMULATED")
    if C["budgets"] != [0.10, 0.20, 0.33]:
        raise SystemExit("experiment C budgets differ from this script")
    if list(C["policies"]) != ["random", *POLICIES]:
        raise SystemExit("experiment C policies differ from this script")
    if C["observed_exposure"]["label"] != "OBSERVED":
        raise SystemExit("the exposure count must be labelled OBSERVED")
    return p


def expected_rows(found, k, n_bad, n, sens, fa):
    """Expected confirmations for one policy at one budget and inspector."""
    real = sens * found
    false = fa * (k - found)
    return {"confirmed_real": real, "confirmed_false": false,
            "precision_of_confirmed": (real / (real + false)
                                       if real + false > 0 else None)}


def run_set(D, P, label, idx, oof):
    C = P["experiment_c"]
    nb = P["bootstrap"]["replicates"]
    sens_grid = [float(s) for s in C["inspector"]["sensitivity"]]
    fa_grid = [float(f) for f in C["inspector"]["false_alarm"]]
    bad, has, plots = D["bad"][idx], D["has_rec"][idx], D["plots"][idx]
    n, n_bad = len(idx), int(bad.sum())
    pol = vp.policies(D, idx, oof)

    bar = "=" * 86
    print(bar)
    print(f"EXPERIMENT C (SIMULATED inspector, OBSERVED ranking) -- {label}")
    print(f"   {n} field-seasons, {n_bad} below threshold")
    print(bar)
    out = {"n": n, "n_bad": n_bad, "budgets": {}}

    for b in C["budgets"]:
        k = max(1, int(round(b * n)))
        rand_found = k * n_bad / n           # exact hypergeometric mean
        row = {"k": k, "random": {"mean_found": rand_found}}
        for name in POLICIES:
            o = vp.outcome(vp.visit(pol[name], b), bad, has)
            row[name] = {"n_found": o["n_found"],
                         "break_even_sensitivity": (rand_found / o["n_found"]
                                                    if o["n_found"] else None)}
        grid = {}
        for s in sens_grid:
            for f in fa_grid:
                cell = {"random": expected_rows(rand_found, k, n_bad, n, s, f)}
                for name in POLICIES:
                    cell[name] = expected_rows(row[name]["n_found"], k,
                                               n_bad, n, s, f)
                grid[f"s={s:.2f}|f={f:.2f}"] = cell
        row["grid"] = grid
        out["budgets"][str(b)] = row
        be = row["register"]["break_even_sensitivity"]
        print(f"budget {b:.2f}: k={k}; random finds {rand_found:.2f} on "
              f"average; register {row['register']['n_found']}, history "
              f"{row['history_only']['n_found']}, workflow "
              f"{row['workflow']['n_found']}")
        print(f"   break-even sensitivity (register vs PERFECT random "
              f"inspector): {be:.2f}")
        g = grid["s=0.70|f=0.10"]
        print(f"   inspector s=0.70, f=0.10: confirmed real "
              f"register {g['register']['confirmed_real']:.2f} vs random "
              f"{g['random']['confirmed_real']:.2f}; share of confirmed "
              f"findings that are real: register "
              f"{g['register']['precision_of_confirmed']:.2f}, random "
              f"{g['random']['precision_of_confirmed']:.2f}")

    # ---- Monte Carlo verdict: does register beat random once inspection is
    # noisy? Joint draws over plots, the random referral and the inspector.
    rng = np.random.default_rng(P["bootstrap"]["seed"])
    uniq = np.unique(plots)
    pos = {g: np.where(plots == g)[0] for g in uniq}
    found_reg = {b: np.empty(nb, dtype=int) for b in C["budgets"]}
    found_rnd = {b: np.empty(nb, dtype=int) for b in C["budgets"]}
    for r in range(nb):
        pick = uniq[rng.integers(0, len(uniq), len(uniq))]
        j = np.concatenate([pos[g] for g in pick])
        pj = vp.policies(D, idx[j], oof)
        perm = rng.permutation(len(j)).astype(float)
        for b in C["budgets"]:
            found_reg[b][r] = (vp.visit(pj["register"], b) & bad[j]).sum()
            found_rnd[b][r] = (vp.visit(perm, b) & bad[j]).sum()
    verdict = {}
    for b in C["budgets"]:
        rows = {}
        for s in sens_grid:
            cr = rng.binomial(found_reg[b], s)
            cn = rng.binomial(found_rnd[b], s)
            d = cr - cn
            ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            rows[f"{s:.2f}"] = {"mean_difference": float(d.mean()),
                                "ci95": ci, "survives": bool(ci[0] > 0)}
        surv = [float(s) for s in rows if rows[s]["survives"]]
        verdict[str(b)] = {"by_sensitivity": rows,
                           "lowest_surviving_sensitivity":
                               min(surv) if surv else None}
        print(f"   budget {b:.2f}: register beats random on confirmed real "
              f"problems down to sensitivity "
              f"{min(surv) if surv else 'none'} (95 % interval excludes 0)")
    out["monte_carlo_register_vs_random"] = verdict
    print()
    return out


def observed_exposure(D, P, idx, oof, label):
    """OBSERVED: real misses whose estimate would look reassuring."""
    C = P["experiment_c"]["observed_exposure"]
    bad, has = D["bad"][idx], D["has_rec"][idx]
    sat = (D["sat_oof"] if oof else D["sat_ship"])[idx]
    est = np.where(has, D["hist"][idx], sat)
    thr = vp.LEVEL * D["bench"]
    pol = vp.policies(D, idx, oof)
    out = {"threshold_kg_per_acre": thr, "budgets": {}}
    print("=" * 86)
    print(f"OBSERVED EXPOSURE -- {label}: below-threshold field-seasons the "
          f"workflow did not inspect,")
    print("and how many carry an estimate at or above the threshold (what a "
          "'looks fine' display")
    print("would have shown)")
    print("=" * 86)
    for b in P["experiment_c"]["budgets"]:
        sel = vp.visit(pol["workflow"], b)
        missed = bad & ~sel
        miss_re = missed & (est >= thr)
        insp_re = sel & bad & (est >= thr)
        row = {"n_bad": int(bad.sum()), "n_missed": int(missed.sum()),
               "n_missed_estimate_at_or_above_threshold": int(miss_re.sum()),
               "n_missed_cold_start": int((missed & ~has).sum()),
               "n_inspected_bad_estimate_at_or_above_threshold":
                   int(insp_re.sum()),
               "missed_estimates_kg_per_acre": sorted(
                   float(round(v, 1)) for v in est[missed]),
               "anchoring_SIMULATED": {
                   f"{a:.1f}": float(a * insp_re.sum())
                   for a in C["anchoring_sweep"]}}
        out["budgets"][str(b)] = row
        print(f"   budget {b:.2f}: {row['n_missed']} of {row['n_bad']} "
              f"missed; {row['n_missed_estimate_at_or_above_threshold']} of "
              f"them at or above threshold; {row['n_missed_cold_start']} "
              f"cold starts; inspected-but-reassuring "
              f"{row['n_inspected_bad_estimate_at_or_above_threshold']}")
    print()
    return out


def main() -> int:
    P = load_protocol()
    D = vp.build()
    test = np.where(D["split"] == "test")[0]
    allx = np.arange(len(D["y"]))
    print("SIMULATED inspector on OBSERVED rankings. The target is the study's "
          "yield threshold on")
    print("farmer-recalled yield, not crop damage.")
    print()
    res = {"label": "SIMULATED inspector; rankings and exposure OBSERVED",
           "protocol_version": P["version"],
           "protocol_frozen": str(P["frozen_on"]),
           "sets": {
               "held-out test split (VERDICT)": run_set(
                   D, P, "held-out test split (VERDICT)", test, oof=False),
               "all field-seasons out-of-fold (secondary)": run_set(
                   D, P, "all field-seasons out-of-fold (secondary)", allx,
                   oof=True)},
           "observed_exposure": {
               "held-out test split": observed_exposure(
                   D, P, test, False, "held-out test split"),
               "all field-seasons out-of-fold": observed_exposure(
                   D, P, allx, True, "all field-seasons out-of-fold")}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf8")
    print(f"wrote {OUT.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
