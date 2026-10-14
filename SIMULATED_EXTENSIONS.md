# Imperfect inspectors, and a YES-TECH what-if

`src/inspection_sim.py` (Experiment C) and `src/yestech_whatif.py`
(Experiment D), both under `configs/extension_protocol.yaml`, frozen before
either was run. **Every number on this page is simulated.** Neither is
field performance, and neither may be quoted beside an observed result
without saying so.

## C. What if the inspector is imperfect?

Experiment A scored a visit as revealing the truth. Here the rankings are
the real, observed ones on the same plots; only the inspector is
simulated, by two numbers: the chance it confirms a real problem
(sensitivity) and the chance it reports one that is not there (false
alarm).

**Break-even.** At a 20 % budget on the held-out plots, a register-guided
inspector with a sensitivity of 0.30 confirms, on average, as many real
problems as a perfect inspector sent to fields at random. Record-guided
visits tolerate a very weak inspector.

**What a finding is worth.** With an inspector at sensitivity 0.70 and a
10 % false-alarm rate, 0.91 of the problems a register-guided team
confirms are real, against 0.60 for random visits. Targeting raises the
base rate in the visited pool, so it protects against false findings as
well as finding more true ones.

**The pre-registered verdict.** On the 11 held-out problem field-seasons,
the interval for register minus random touches zero even for a perfect
inspector: the held-out split is too small to settle the question at any
sensitivity, and that is the verdict. Over all 53 problem field-seasons
(out-of-fold, secondary), the register still confirms more real problems
than random visits down to a sensitivity of 0.3 at a 20 % budget.

What this does not show: how accurate real inspectors are. The rankings
are observed; the inspector is a modelling assumption.

## D. A YES-TECH what-if

No YES-TECH output is public, so nothing here says how accurate YES-TECH
is. What can be worked out without it is the mechanism. Under PMFBY's area
approach a unit's claim is (threshold − actual) / threshold of the sum
insured, and the yield used blends the technology estimate with
crop-cutting. The weights are sourced: 30 % is the national minimum, and
in 2025-26 Maharashtra used 50 % and Madhya Pradesh 100 % (Lok Sabha
Starred Question 25, 21 July 2026). The error sizes are assumptions; 0.30
is the largest unit-level nRMSE the January 2023 YES-TECH manual tolerated,
and reading it as a one-standard-deviation relative error is this study's
choice. Crop-cutting error is set at 0.10 and swept.

**Weight decides how much the model's error reaches the farmer.** For a
unit truly 10 % below its threshold, with a technology error of 0.20, the
chance of being paid nothing is 29 % at 100 % weight against 11 % at the
30 % minimum. With an error of 0.30 the figures are 36 % and 16 %. Blending
with crop-cutting averages some error away; at full weight nothing does,
and the payout is exactly as accurate as the model.

**Error cuts both ways.** Unbiased error also pays units that are not due,
and in every unbiased run more of them than it wrongly leaves unpaid. A
model that overestimates yield by 10 % shifts the balance against the
farmer.

**Which re-verification rule recovers the wrongly unpaid.** Rechecking 5 %
of units at 100 % weight and an error of 0.20:

| rule | rechecks per unit recovered |
|---|---:|
| units just above the threshold, closest first | 5.4 |
| random units | 19.5 |
| lowest estimates first | none recovered |

Rechecking near the threshold recovers 14 % of the wrongly unpaid units
with a 5 % budget, about four times as efficiently as random rechecks. It
is not free: each recheck is itself a noisy crop-cut, and the same
rechecks pay 658 units that were not due. Rechecking the lowest estimates
recovers nothing, because those units are already being paid.

**The pre-registered verdict.** Near-threshold re-verification recovers
more wrongly unpaid units than random rechecks at every weight, every
error size and both budgets, at the primary settings. That holds under
these assumptions only. The design consequence is a rule a scheme could
test with its own data: where the model's weight is high, re-verify the
units whose estimate lands just above the threshold.

What this does not show: that YES-TECH makes these errors, that any farmer
missed a claim, or that this rule would perform the same on real units.
Validating it needs matched unit-level technology and crop-cutting yields,
which are not public.
