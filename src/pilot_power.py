"""How large must a field pilot be to settle what this data could not?

    python src/pilot_power.py

PILOT_PROTOCOL.md proposes a one-season pilot. Its size should come from
the effects actually observed here, not from a round number. For each
comparison the pilot would make, this takes the observed paired difference
in recall at a 20 % budget and the observed share of discordant
below-threshold fields (found by one policy and not the other), and returns
the number of below-threshold fields a paired (McNemar) comparison needs for
80 % power at a two-sided 5 % level, using the standard normal
approximation:

    n = ( z_a * sqrt(p_d) + z_b * sqrt(p_d - d^2) )^2 / d^2

where d is the difference in recall and p_d the discordant share. The
number of fields to enrol follows from the observed base rate of
below-threshold field-seasons.

These are planning numbers from small samples. Every input is printed so a
reader can redo them with their own assumptions.

Writes results/metrics/pilot_power.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import verification_policies as vp  # noqa: E402

OUT = PROJ / "results" / "metrics" / "pilot_power.json"
BUDGET = 0.20
ALPHA, POWER = 0.05, 0.80


def n_pairs(d: float, p_d: float) -> float | None:
    za = NormalDist().inv_cdf(1 - ALPHA / 2)
    zb = NormalDist().inv_cdf(POWER)
    if d == 0 or p_d <= d * d:
        return None
    return (za * math.sqrt(p_d) + zb * math.sqrt(p_d - d * d)) ** 2 / d ** 2


def main() -> int:
    D = vp.build()
    out = {"budget": BUDGET, "alpha": ALPHA, "power": POWER,
           "method": "paired proportions (McNemar), normal approximation",
           "sets": {}}
    for label, idx, oof in (
            ("held-out test split", np.where(D["split"] == "test")[0], False),
            ("all field-seasons out-of-fold", np.arange(len(D["y"])), True)):
        bad = D["bad"][idx]
        pol = vp.policies(D, idx, oof)
        sel = {k: vp.visit(v, BUDGET) for k, v in pol.items()}
        nb = int(bad.sum())
        base_rate = nb / len(idx)
        rows = {}
        # workflow against the register, and against history only: both
        # lists exist, so discordance is counted directly
        for a, b in (("workflow", "register"), ("workflow", "history_only")):
            fa, fb = sel[a] & bad, sel[b] & bad
            d = (fa.sum() - fb.sum()) / nb
            p_d = ((fa & ~fb).sum() + (fb & ~fa).sum()) / nb
            n = n_pairs(d, p_d)
            rows[f"{a} vs {b}"] = {
                "difference": float(d), "discordant_share": float(p_d),
                "below_threshold_fields_needed": n,
                "fields_to_enrol": (n / base_rate) if n else None}
        # register against random referral: a random list has no fixed
        # membership, so discordance is its expectation under independence
        r = float((sel["register"] & bad).sum() / nb)
        q = BUDGET
        p_d = r * (1 - q) + q * (1 - r)
        n = n_pairs(r - q, p_d)
        rows["register vs random"] = {
            "difference": r - q, "discordant_share": p_d,
            "below_threshold_fields_needed": n,
            "fields_to_enrol": (n / base_rate) if n else None}
        out["sets"][label] = {"n": int(len(idx)), "n_bad": nb,
                              "base_rate": base_rate, "comparisons": rows}
        print(f"{label}: {nb} of {len(idx)} below threshold "
              f"(base rate {base_rate:.3f})")
        for k, v in rows.items():
            nn = v["below_threshold_fields_needed"]
            print(f"   {k:<26s} d={v['difference']:+.3f}  "
                  f"p_d={v['discordant_share']:.3f}  ->  "
                  + (f"{math.ceil(nn)} below-threshold fields, about "
                     f"{math.ceil(v['fields_to_enrol'])} fields"
                     if nn else "no difference observed; not estimable"))
    OUT.write_text(json.dumps(out, indent=2), encoding="utf8")
    print(f"wrote {OUT.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
