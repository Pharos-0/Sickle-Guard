# Final validation report — upgraded SICKLE yield system

Run 2026-10-11. Confirmatory evaluation scored **once** against criteria
frozen in `ACCEPTANCE_CRITERIA.md` before the run.

---

## Verdict: system NOT accepted. The signal is real; 27 test plots cannot establish it.

Both primary criteria failed, and they failed on precision, not direction.
A better-powered follow-up over all 131 plots then cleared both. The study
is **precision-limited, not signal-limited**, and the missing evidence is
independent plots.

## 1. Confirmatory run — 27 untouched test plots

| Criterion | Result | Status |
|---|---|---|
| A1 MAE ≤ train-mean floor | 253.6 vs 290.7 | **pass** |
| A2 \|bias\| ≤ 184 kg/acre | +18.8 | **pass** |
| A3 SD ratio ≤ 1.0 | 0.571 (MSE-optimal 0.491) | **pass** |
| **B1 AUROC CI excludes 0.5 @ 80 %** | 0.738, CI **[0.434, 0.947]** | **FAIL** |
| **C1 recall CI > 33 % @ 33 % budget** | 55 %, CI **[14 %, 100 %]** | **FAIL** |

Both point estimates are useful — AUROC 0.738, recall 55 % against 33 %
random. Both intervals are too wide to conclude anything, because the test
split holds 27 plots and 11 true losses.

The criteria were not weakened after seeing this. Section E of
`ACCEPTANCE_CRITERIA.md` pre-specified the response: identify the evidence
gap through a separately designed experiment.

## 2. Variance calibration worked, as hygiene

Inflation factor 2.094, fitted on **validation only**, never on test.

| | before | after |
|---|---:|---:|
| SD ratio | 0.274 | **0.571** |
| MAE | 257.8 | **253.6** |
| bias | — | +18.8 |

Free improvement: the spread roughly doubles toward the MSE-optimal value
and MAE improves slightly. It is reported as hygiene, **not** as a fix for
the decision problem — `ROOT_CAUSE_REPORT.md` §2 shows rescaling cannot
change a ranking and trades missed losses for false alarms one for one.

## 3. Better-powered follow-up — all 131 plots, exploratory

4 repeats of grouped 5-fold, every plot held out once per repeat,
preprocessing fitted inside each fold.

| | out-of-fold, 131 plots | confirmatory, 27 plots |
|---|---|---|
| AUROC @ 80 % | **0.717, CI [0.618, 0.812]** — excludes 0.5 | 0.738, CI [0.434, 0.947] |
| recall @ 33 % budget | **64 %, CI [46 %, 76 %]** — excludes random | 55 %, CI [14 %, 100 %] |
| MAE | 277.9 (floor 303.0) | 253.6 (floor 290.7) |
| r | +0.468 | +0.491 |

**The point estimates agree closely. Only the precision differs.** This is
an exploratory estimate — out-of-fold reuses every plot across folds and is
optimistic relative to an untouched split — so it does **not** overturn the
confirmatory verdict. It does locate the gap.

## 4. The evidence gap, quantified

Mean width of the 95 % AUROC interval, subsampling the CV set:

| independent plots | CI width |
|---:|---:|
| 27 (the test split) | 0.376 |
| 50 | 0.311 |
| 80 | 0.245 |
| 110 | 0.215 |
| 131 | 0.194 |

To establish B1 confirmatorily — a lower bound above 0.5 at a point estimate
near 0.72 — the interval must be narrower than about 0.44. That is reached
somewhere between 27 and 50 plots on this curve, so a test split of roughly
**50–60 independent plots** would likely suffice. The present data cannot
supply that without shrinking training below the point where the model works.

## 5. What is established, and what is not

**Established**
- Yield labels are kg/acre, verified against masks on 319/319 rows.
- Variance compression is partly the correct response to a weak signal; the
  MSE-optimal SD ratio is r, not 1.0.
- Calibrating toward that optimum is free (MAE 257.8 → 253.6).
- Rescaling cannot change the ranking, so it cannot fix threshold decisions.
- The model's ordering carries loss-relevant signal: AUROC ≈ 0.72 both in
  the confirmatory run and out-of-fold.

**Not established**
- That the ordering beats chance on an untouched split — B1 failed.
- That budgeted verification beats random on an untouched split — C1 failed.
- Any claim about real payouts, farmer outcomes or operational YES-TECH.
  No Insurance Unit identity exists in this data at all.

## 6. Reproducibility

40 tests pass. Frozen split, machine-readable metrics for every claim,
leakage asserted in code (a test plot in train/val raises). Backend
scikit-learn 1.5.2, with a validated numpy fallback.

`bash reproduce.sh` then `python src/confirmatory.py`.

## 7. Recommended next step

Do not tune further against this test split — the criteria are frozen and
the split is spent. Either acquire more independent plots, or report the
study as it stands: a pre-registered negative with a well-characterised
cause and a quantified sample-size requirement. The second is publishable
and honest; the first needs data that does not currently exist.
