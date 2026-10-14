# Acceptance criteria — frozen before the confirmatory test run

Written 2026-10-11, after Phase-2 diagnosis and after deriving the operating
numbers on **validation only**, and before the upgraded system is scored on
the 27 held-out test plots.

Every threshold below is motivated from validation evidence or from the
measurement context. None was chosen to make the current system pass, and
none may be weakened after the test result is seen. A failed criterion is
reported as failed.

---

## 0. What changed, and why these criteria differ from the brief

The brief's primary objective was to reduce variance compression so that
threshold decisions improve. `ROOT_CAUSE_REPORT.md` shows that path is
capped: compression is partly the statistically correct response to a weak
signal (optimal SD ratio = r), and inflating it trades missed losses for
false alarms roughly one for one while degrading MAE.

The binding constraint is the **ranking**, which rescaling cannot change.
So the primary criterion is ranking quality and budgeted verification, with
MAE demoted to a guard against degeneracy.

## 1. Evidence these criteria are set from (validation, 24 plots, 60 seasons)

| quantity | validation value |
|---|---:|
| MAE | 326.0 kg/acre |
| train-mean floor MAE | 346.2 kg/acre |
| SD ratio | 0.213 |
| correlation r | +0.446 |
| loss-ranking AUROC, 80 % level | 0.712 |
| lowest-predicted-first recall @ 33 % budget, 80 % level | 78 % (9 true losses) |

---

## A. Predictive quality — guard criteria, not the objective

**A1 (must pass).** Test MAE must not exceed the train-mean floor. A system
whose point predictions are worse than a constant cannot be defended
regardless of its ranking.

**A2 (must pass).** Mean prediction bias on test within ±10 % of the
benchmark, i.e. |bias| ≤ 184 kg/acre. Motivation: a systematic offset larger
than this shifts every threshold comparison by more than the gap between
adjacent indemnity levels (184 kg/acre separates the 70 % and 80 % cuts).

**A3 (report, no pass/fail).** SD ratio after calibration should move toward
the MSE-optimal value r and must not exceed 1.0. Exceeding r is permitted
only if A1 still passes, and the trade-off must be reported. This is
hygiene; it is explicitly **not** the measure of success.

## B. Ranking quality — the primary criterion

**B1 (must pass, PRIMARY).** For loss detection at the 80 % indemnity level,
the plot-bootstrapped 95 % CI on AUROC must exclude 0.5.

Motivation: validation gives 0.712. A CI excluding chance is the weakest
claim that supports using the ordering at all. A point-estimate threshold
such as "AUROC ≥ 0.70" is deliberately **not** used, because with 27 test
plots the sampling error is large and such a bar would be met or missed by
noise.

**B2 (report).** AUROC at the 70 % and 90 % levels, with intervals. These
are sensitivity analyses, not pass/fail, because the 70 % level has few
positives.

## C. Decision quality at a fixed verification budget

**C1 (must pass, PRIMARY).** At a **33 % verification budget** and the 80 %
indemnity level, recall of true losses by lowest-predicted-first referral
must have a plot-bootstrapped 95 % CI whose lower bound exceeds the
random-referral expectation of 33 %.

Motivation for the budget: 33 % is the middle of the three budgets examined
throughout this project, and the one at which validation showed the clearest
separation (78 % vs 33 %). Fixing it in advance prevents choosing the budget
that flatters the result.

**C2 (must pass).** The upgraded strategy must not be worse than the
previous study's reliability-guided verification at the same budget, judged
by residual wrong decisions. "Not worse" means the paired difference does
not significantly favour the old strategy.

**C3 (report).** Missed losses and false alarms reported separately at every
budget and level. Any claim of improvement must state both.

## D. Reproducibility

**D1 (must pass).** All existing tests pass, plus new tests covering the
calibration step and the ranking evaluation.

**D2 (must pass).** No test label is touched during fitting, calibration or
model selection. Asserted in code, not just claimed.

**D3 (must pass).** The confirmatory test evaluation runs **once**. If it is
re-run after a change, the change and the reason are recorded in this file
and the result is relabelled exploratory.

**D4 (must pass).** Frozen split, machine-readable metrics, and a single
documented command reproduce the analysis.

---

## E. What failure means

If **B1** fails, the ordering is not demonstrated on held-out plots and the
system has no defensible decision layer on this dataset. The correct
response is to report that and identify the evidence gap — most likely the
number of independent plots — not to search for a threshold or budget at
which something clears.

If **C1** fails but B1 passes, the ranking carries signal but not enough to
beat random referral at a realistic budget; report the gap between AUROC and
operational usefulness.

If **A1** fails while B1 and C1 pass, the system is a loss *detector* and not
a yield *predictor*, and the paper must say so in those words.

## F. Declared in advance

- Primary confirmatory comparisons: **B1 and C1**. Two comparisons, so a
  Bonferroni-adjusted threshold of 0.025 applies where a p-value is used.
- All other numbers in the final report are secondary or exploratory and are
  labelled as such.
- The test set is scored once, after this file is committed.
