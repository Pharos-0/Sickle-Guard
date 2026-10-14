# Status — SICKLE yield / reliability / insurance-simulation study

Run 2026-10-11. Separate project from the SICKLE audit paper; that
manuscript is untouched.

Everything below was executed. Where something was not run it says so.

---

## The finding

**A yield model that looks competitive on MAE flags almost no crop losses,
because it retains only 28 % of the reference yield variance.**

| | reference | model |
|---|---:|---:|
| SD (kg/acre) | 392 | **109** |
| range (kg/acre) | 1,820 | 568 |
| variance ratio | — | **0.279** |
| correlation with reference | — | **+0.521** |
| calibration slope (1.0 = calibrated) | — | 1.867 |

The model *does* rank fields: r = +0.52. But a predictor that cannot explain
much variance minimises MAE by sitting near the mean, and loss thresholds
live in the tail it has stopped reaching. Consequence, at a train-mean
benchmark of 1,838 kg/acre:

| Indemnity level | Reference flags loss | Model flags loss | Disagreements | Of which **missed losses** |
|---|---:|---:|---:|---:|
| 70 % | 9 / 62 | **0** | 9 (15 %) | **9** |
| 80 % | 11 / 62 | **0** | 11 (18 %) | **11** |
| 90 % | 18 / 62 | 3 | 17 (27 %) | **16** |

**Every disagreement at 70 % and 80 % is a missed loss. Not one is a false
alarm.** The error is entirely in the direction that would deny a farmer a
flagged loss. MAE never reveals this: the model's MAE (256) is better than
the train-mean floor (292).

This is the result worth building the paper around. It is a precise,
quantified, mechanistically explained failure mode of a metric that looks
fine.

## RQ1–RQ2: yield prediction and ablation

Grouped by plot: train 80 plots / 197 seasons, val 24 / 60, test 27 / 62.
Selection on validation only; test scored once afterwards.

| | Variant | Val MAE | Test MAE | Test 95 % CI |
|---|---|---:|---:|---|
| V1 | train mean (floor) | 346.2 | 292.2 | [208, 373] |
| V2 | season mean | 353.1 | 294.3 | [212, 374] |
| V3 | ridge, S2 optical | 384.9 | 289.9 | [225, 354] |
| V4 | + phenology | 392.7 | 276.5 | [213, 339] |
| V5 | + Sentinel-1 radar | 380.6 | 298.0 | [232, 361] |
| V6 | + geometry | 335.4 | 262.3 | [206, 321] |
| **V7** | **random forest, all features** | **325.0** | **258.6** | **[188, 331]** |
| V8 | gradient boosting | 339.3 | 274.9 | [203, 351] |

Paired by plot against the floor — the correct test, since the marginal
intervals above overlap heavily and say nothing either way:

| Variant | Δ MAE vs floor | 95 % CI | perm p | verdict |
|---|---:|---|---:|---|
| V7 | **+33.4** | [+6.1, +64.4] | **0.026** | better |
| V6 | +29.8 | [−18.4, +79.4] | 0.241 | inconclusive |
| V4 | +15.7 | [−17.6, +51.1] | 0.397 | inconclusive |
| others | — | contain zero | — | inconclusive |

**V7 is the only variant that beats the floor, and it does not survive
multiple-comparison correction**: 7 variants were tested, so the Bonferroni
threshold is 0.0071 and p = 0.026. Reported as a nominal improvement, not an
established one.

Adjacent steps: only V3→V4 (adding phenology) helps, +13.5 [+2.5, +24.3],
p = 0.026 — also inside the correction. Radar *hurt* in this setup
(V4→V5, −21.5) though inconclusively.

## RQ3: reliability screening

Predictor V7. Signals computed at inference with no label: forest tree
spread, seed spread, cloud-free acquisition count, plot pixel count,
distance from the training distribution, predicted yield.

Pre-declared headline — split at the test median APE (9.2 %), 31 of 62
positive:

| Signal | AUROC | 95 % CI | verdict |
|---|---:|---|---|
| rf_spread | 0.672 | contains 0.5 | not demonstrated |
| low_n_clear | 0.575 | contains 0.5 | not demonstrated |
| ood_distance | 0.569 | contains 0.5 | not demonstrated |
| quantile_width | 0.531 | contains 0.5 | not demonstrated |
| others | ≤ 0.559 | contain 0.5 | not demonstrated |

**No signal is demonstrated at the pre-declared threshold.**

Exploratory, across the pre-declared threshold grid: at APE > 20 % and
> 25 % (13 positives), `rf_spread` reaches AUROC 0.801 [0.550, 0.959] and
the **quantile-regression interval width** 0.714 [0.519, 0.859], both
excluding chance. The quantile intervals are well calibrated overall —
nominal 90 %, empirical 90.3 % coverage on test. So ensemble
disagreement looks able to flag *large* errors while failing on
median-sized ones — plausible, and exactly what a pre-registered follow-up
should test. It is **not** promoted to a headline, because picking the
threshold after seeing it would not be confirmatory.

Screening at fixed budgets with `rf_spread` on the median split: 20 %
workload catches 35 % at 92 % precision (random 19 %), 33 % catches 55 % at
85 % precision (random 32 %), 50 % catches 65 % (random 50 %).

## RQ4: insurance-decision simulation

**Read `INSURANCE_ASSUMPTIONS.md` first. No monetary figure is produced
anywhere.** The dataset has no Gram Panchayat identity for any of the 131
yield plots, so Insurance Units cannot be formed and the real PMFBY
procedure cannot be reproduced. The simulation is plot-level against a
hypothetical threshold built from the train-split mean — not a PMFBY
Threshold Yield.

Reliability-guided verification does reduce residual wrong decisions versus
random at the same budget (70 % level, 20 % budget: 3.0 remaining vs 7.2
random vs 4.0 for the low-predicted-yield baseline). At the 90 % level the
low-yield baseline overtakes it at a 50 % budget (4 vs 7). Mixed, and on 27
plots not separable.

## What was not done

| Item | Status |
|---|---|
| Deep temporal / spatiotemporal models | **not run.** 197 training rows and 80 plots do not support them; the proposal's own guidance says not to add capacity without rationale |
| Quantile-regression intervals | **run.** Nominal 90 %, empirical 90.3 % coverage on test; width used as a reliability signal |
| Conformal prediction intervals | **not run.** Grouped conformal needs a calibration set carved from 24 val plots; left as follow-up |
| Insurance Unit aggregation | **impossible.** No geography; see DATA_AUDIT.md §4.1 |
| Monetary payout estimates | **refused.** No Sum Insured exists |
| scikit-learn models | **now used.** 1.9.1 was blocked by Smart App Control; **1.5.2 loads**, so scikit-learn is the backend. The numpy trees remain as a validated fallback (`src/models.py` picks); 12 cross-check tests confirm both give the same headline MAE on this study's data (258.8 vs 258.6) |
| Manuscript | **not written.** Results are frozen and reproducible; the paper is the next step |

## Verification

40 tests pass: 12 on split integrity and target semantics, 16 on the numpy
tree implementations (split finder checked against brute force), and 12
cross-checking those against scikit-learn — including that both backends
agree on the headline MAE for this study's own feature table.

Key integrity properties pinned by test: no plot crosses partitions; no
feature is collinear with the target; every season of a plot sits in one
split; the target behaves like a density not a total.

Reproduce: `bash reproduce.sh`.

## Two corrections to the working assumptions

1. **YIELD is kg per acre, not a field total.** Uncorrelated with AREA
   (r = −0.29). Median 1,800 kg/acre = 4,448 kg/ha.
2. **Mask band 5 holds yield for every plot in the patch.** Summing the
   whole band overstates it by up to 5×; restricting to `band0 == PLOT_ID`
   reproduces the CSV exactly on 319/319 rows.

## Recommended next step

Write the paper around the variance-compression finding, with the ablation
as supporting evidence that even the best model is only nominally better
than the mean. The reliability and simulation sections are honest secondary
results. The contribution is not "we predicted yield well" — we did not —
it is **"a model selected on MAE is systematically blind to the losses an
insurance decision exists to detect, and here is the mechanism and its
magnitude."**
