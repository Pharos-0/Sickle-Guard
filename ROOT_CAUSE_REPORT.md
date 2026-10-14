# Root-cause report: why the predictor misses losses

Phase 2 diagnosis, run 2026-10-11 before any new model was built. The
conclusion changes what the remaining phases should optimise, so it is
recorded before they start.

---

## 1. The premise that needs correcting

The upgrade brief treats variance compression as a defect to engineer away:
"the existing model produces predictions that are too compressed toward the
mean … the project must investigate and, if the available data permit,
correct this failure."

**Compression is partly correct behaviour, not a bug.** Under squared-error
loss the optimal linear predictor has

    SD(prediction) / SD(target) = r

exactly, where r is the correlation between prediction and target. Shrinking
toward the mean is the right response to a weak signal. A predictor that
matched the target's spread while correlating at r = 0.51 would be
*miscalibrated in the opposite direction* and would have higher error.

Measured on the 62 held-out field-seasons:

| quantity | value |
|---|---:|
| observed SD ratio | **0.274** |
| correlation r | **0.509** |
| MSE-optimal SD ratio | **0.509** |

So the model is more compressed than optimal. There is genuine room to
inflate from 0.274 to 0.509 — and doing so is free, MAE even improves
slightly (257.8 → 251.4). But there is **no** justification for pushing
toward 1.0, and the brief's framing implies there is.

## 2. The ceiling test: does fixing compression fix the decisions?

Predictions were rescaled about their own mean to several SD ratios, holding
the ranking fixed, and threshold crossings recounted against the train-mean
benchmark of 1,838 kg/acre.

| SD ratio | MAE | 70 % recall / precision | 80 % recall / precision | 90 % recall / precision |
|---:|---:|---|---|---|
| 0.274 (observed) | 257.8 | 0 % / — | 0 % / — | 11 % / 67 % |
| 0.509 (MSE-optimal) | 251.4 | 0 % / — | 18 % / 100 % | 33 % / 40 % |
| 0.750 | 268.9 | 11 % / 50 % | 36 % / 57 % | 50 % / 43 % |
| 1.000 | 310.1 | 11 % / 25 % | 55 % / 50 % | 50 % / 39 % |

**Inflating the variance trades missed losses for false alarms at roughly
one for one, and degrades MAE.** At SD ratio 1.0 the 80 % level catches 6 of
11 true losses but raises 6 false alarms. That is not a solved problem; it
is the same problem rebalanced.

## 3. The actual binding constraint is the ranking

Rescaling is a monotone transform, so it cannot change how the model
*orders* fields. The order is what caps everything:

| threshold level | true losses | AUROC of predicted yield as a loss ranker |
|---|---:|---:|
| 70 % | 9 | **0.721** |
| 80 % | 11 | **0.743** |
| 90 % | 18 | **0.698** |

**This is the real result, and it is more useful than it first appears.**
AUROC 0.70–0.74 means the model carries genuine, decision-relevant signal
about which fields are low-yielding — far more than any of the reliability
signals tested in Phase 6 of the previous study, which sat between 0.47 and
0.67 with intervals spanning chance.

What destroys that signal is **thresholding a compressed scale**, not the
model's knowledge. The information is in the ordering; the loss is incurred
when the ordering is converted to a yes/no by comparing a shrunken number to
a fixed cut-off.

## 4. What this means for the remaining phases

Three consequences, in order of importance.

**4.1 Stop optimising the point prediction for the decision.** Phases 3, 4
and 7 of the brief aim to improve MAE, reduce compression, and then threshold
the result. The ceiling test shows that path is capped and self-defeating:
every step toward matching variance costs accuracy and buys false alarms.

**4.2 Rank and verify instead of thresholding.** The defensible system takes
the model's *ordering* and refers the lowest-predicted fields for
verification under a budget. That uses the AUROC 0.72 signal directly and
never requires the prediction to cross an absolute threshold. It is also
closer to how a verification budget actually works: an agency can inspect
the n most suspicious fields, not "all fields below a number".

Notably, the previous study already found this: the `low_yield` verification
baseline was competitive with, and at one budget better than, the
reliability-guided strategy. That was reported as a mildly awkward result.
It is actually the signal.

**4.3 Variance calibration is still worth doing, but as hygiene, not as the
fix.** Inflating to the MSE-optimal ratio is free and should be applied,
fitted on validation only. It must not be sold as solving the decision
problem.

## 5. Causes examined and their status

| Candidate cause | Status |
|---|---|
| Shrinkage from weak signal | **Demonstrated and partly correct.** Optimal ratio is r = 0.509; observed 0.274 |
| Over-regularisation beyond the optimum | **Demonstrated.** Gap between 0.274 and 0.509 is real and recoverable |
| Target leakage | **Ruled out.** Asserted at extraction and by test; no feature correlates above 0.99 with the target |
| Wrong target semantics | **Ruled out.** YIELD is kg/acre, verified against masks on 319/319 rows |
| Mask misalignment across sensor grids | **Ruled out.** Mask is resampled onto each band's native grid; 10/20/60 m handled separately |
| Too few pixels per plot | **Plausible, not isolated.** A plot is ~15 S2 pixels; cannot be tested without higher-resolution imagery |
| Label noise (farmer recall) | **Plausible, not isolated.** Would cap r regardless of model; untestable without CCE measurements |
| Too few independent plots | **Plausible and likely binding.** 80 training plots for a between-plot signal |

The last three cannot be separated with this dataset. They are reported as
plausible, not demonstrated, and they all push in the same direction: they
cap r, and r caps everything else.

## 6. Revised objective for the upgraded system

Replace "reduce variance compression so threshold decisions improve" with:

> **Improve the model's ranking of fields by loss risk, and build the
> decision layer as budgeted verification of the worst-ranked fields rather
> than as a threshold on a point prediction.**

Primary metric becomes AUROC / average precision for loss detection, and
recall at a fixed verification budget. MAE stays as a secondary check that
the predictor has not become degenerate. Variance calibration to the
MSE-optimal ratio is applied as hygiene and reported, not claimed as a fix.

This is a change to the brief, made before the experiments it governs, and
it is justified by §2 and §3 above rather than by any test result obtained
after the fact.
