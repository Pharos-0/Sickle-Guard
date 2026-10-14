# Discovery: the model is a site-quality estimator, not a crop monitor

2026-10-11. Found by asking a question the project had not asked, answered
with a counterfactual that needs no new data.

---

## 1. The finding in one line

**83 % of this model's loss-detection skill survives when you feed it
imagery from the wrong season of the right field.** The satellite time
series is functioning as a site fingerprint, not as an observation of what
happened to the crop that season.

## 2. How it was found

The data audit recorded that ~95 % of yield variance is *between* plots
(SD of plot means 413 kg/acre against an overall SD of 433), and that every
plot contributes 2–4 seasons. Those two facts together permit a
decomposition the project had never run: a model evaluated on unseen plots
can score well by learning which **fields** are good without ever learning
which **season** at a field was bad.

Decomposing the out-of-fold predictions over all 131 plots:

| component | statistic |
|---|---|
| between-plot (ranks plots by mean yield) | **r = +0.568** |
| within-plot (ranks seasons inside a plot) | r = +0.153 |
| within-plot season-ordering accuracy | **53.7 %, 95 % CI [47.4, 60.0]** |

Chance is 50 %. The within-plot ability is not distinguishable from it.

Accuracy by the size of the within-plot yield gap is non-monotonic —
51.9 % / 56.6 % / **40.4 %** / 69.7 % across 0–100, 100–200, 200–400 and
>400 kg/acre — and the 200–400 bin sitting *below* chance is the signature
of noise, not of a signal that strengthens with effect size. Only the >400
bin clears chance (n = 33), and it does not survive correction for the four
bins tested.

## 3. The decisive test

Inference, not experiment, had carried the argument so far. So the inputs
were made counterfactual, inside the same grouped cross-validation, on
identical folds:

| condition | what it destroys | MAE | r | loss AUROC |
|---|---|---:|---:|---:|
| 1 real features | nothing | 277.9 | 0.468 | **0.717** |
| 2 **same-plot season swap** | season-specific info only | 282.4 | 0.422 | **0.679** |
| 3 different-plot swap (control) | site info too | 326.5 | 0.043 | 0.460 |
| train-mean floor | — | 303.0 | — | 0.500 |

Condition 2 replaces each field-season's imagery with that of a **different
season of the same plot**, keeping the label. Everything about the site is
preserved; everything about the season is destroyed.

- Cost of destroying the season: **AUROC −0.038, MAE +4.6 kg/acre.**
- Cost of destroying the site as well: **AUROC −0.256**, r collapses to
  0.043, MAE falls below the train-mean floor.

Expressed as skill above chance: 0.217 real, 0.179 after the season swap.
**83 % of the skill is site-only; 17 % is attributable to the season.**

The control behaves exactly as it must — collapsing to chance — which is
what makes the first comparison interpretable rather than a null result from
broken features.

## 4. Why this matters more than the accuracy numbers

The operational task in loss assessment is detecting **a bad year at a
field**. A chronically low-yielding field is not an insurable event; it is a
known property of the land, already priced in.

In this dataset, of 53 loss seasons at the 80 % threshold, **32 occur at 12
plots that are in loss in every season they appear**. Those are chronic
sites, not loss events. The headline AUROC of 0.717 is substantially earned
on them.

So the model that this project — and, the literature suggests, much of this
genre — reports as a "yield predictor" is better described as a **site
index**. That has three consequences:

1. **It could be obtained far more cheaply.** A soil survey or a few years
   of historical yield would give the same ranking without any satellite
   time series. The expensive input is not doing the work attributed to it.
2. **It is the wrong instrument for the stated purpose.** Screening for
   insurance loss needs within-plot, between-season sensitivity, which is
   the component measured here at chance.
3. **It explains every other negative in this project.** Why temporal
   features barely helped, why radar hurt, why phenology gained little, why
   predictions compressed to 28 % of the true variance, and why ensemble
   disagreement failed to flag errors: the model was never reading the
   season, so season-derived features had nothing to add.

## 5. What this does and does not establish

**Established, on 131 plots out-of-fold**
- The between-plot component is real and reasonably strong (r = +0.568).
- The within-plot component is not demonstrated (53.7 %, CI spans chance).
- 83 % of loss-detection skill is reproducible from the wrong season's
  imagery of the same field.
- The control confirms the features carry site information and the
  experiment is not vacuous.

**Not established**
- That no model could read the season from this imagery. This tests *this*
  feature set and *this* model class. A model with higher spatial
  resolution, better cloud handling, or genuinely measured labels might
  differ.
- That the finding generalises beyond SICKLE's Cauvery Delta smallholdings.
- Anything about real payouts or farmer outcomes.

**Confounded, and honestly so**
Farmer-recalled labels may themselves carry less within-plot year-to-year
signal than the true harvest did — recall compresses toward a typical year.
If so, part of the missing within-plot signal is a label limitation rather
than a model limitation. These cannot be separated without measured yields,
and the distinction is important enough that the paper must state it rather
than claim the model is at fault.

## 6. The contribution this supports

Not a new architecture. An **empirical finding plus the diagnostic that
produces it**:

> A season-permutation test that separates site-quality skill from
> season-specific skill in field-level yield models, requiring no new
> labels, applied to a benchmark where it shows that 83 % of apparent
> loss-detection performance is site identification.

The test is cheap, general, and applicable to any yield model trained on
repeated observations of the same fields. Reported performance in this genre
is routinely a single pooled metric over field-seasons; this shows that
metric can be dominated by a component with no operational value for the
decision the model is built to support.

## 7. Evidence

| claim | artefact |
|---|---|
| between/within decomposition | research log, `results/predictions/oof_pred.npy` |
| gap-size analysis | research log |
| season-swap experiment | `src/season_swap.py`, `results/metrics/season_swap.json` |
| chronic vs event loss split | research log |
| 95 % between-plot variance | `DATA_AUDIT.md` §4.2 |
