# Response to the three validation requests

2026-10-11. Each request, what was run, and what it showed.

---

## 1. Validate the history-first strategy without temporal leakage — DONE, and it survived

**The leak was real and I had missed it.** Every one of the 131 plots appears
in both a 2019-sown and a 2020-sown season. The earlier plot-history
baseline averaged each field-season's *other* seasons, so predicting the
2019 harvest used the 2020 harvest. Future information, unavailable at any
real decision point.

Replaced with a strictly causal rule in `src/causal_history.py`: a season's
history is the mean yield of seasons at the same plot whose **harvest date
precedes this season's sowing date**. Dates come from the released index and
are distinct for all 131 plots.

The cost is severe and is the honest picture: causal history exists for only
**188 of 319 field-seasons (59 %)**, against 100 % under the leaky rule. A
plot seen in 2019 and 2020 yields history for the 2020 season only.

| estimator | all | has history | cold start |
|---|---:|---:|---:|
| global mean | 303.0 | 323.4 | 273.8 |
| **causal history** | 201.6 | **201.6** | — |
| satellite model | 277.9 | 290.7 | 259.5 |
| history-first hybrid | **225.3** | 201.6 | 259.5 |

Paired on the 188 with history: **history beats satellite by 89.1 kg/acre,
95 % CI [−129.5, −50.7]**. The conclusion holds. What changes is coverage:
history answers 59 % of cases, not all of them, and satellite remains the
only option for the other 41 %.

A bootstrap indexing bug surfaced during this and was fixed: the resampler
returned positions into a subset array while the statistic indexed the
global arrays, silently yielding NaN intervals.

## 2. Test seasonal deterioration using independently measured yields — BLOCKED

No measured yields exist in or alongside this dataset. SICKLE's labels are
farmer recall, and no crop-cutting measurements, official yield records or
alternative ground truth are available for these 131 plots.

This is the single most consequential gap in the project, because the
failure of bad-season detection (AUROC 0.586–0.598, all intervals spanning
chance) has two explanations that cannot be separated without it:

- the imagery carries no within-plot seasonal signal, or
- farmer recall compresses toward a typical year, so the *label* carries no
  within-plot seasonal signal.

**What would resolve it:** per-field measured yields — crop-cutting
experiment results or harvester/weighbridge records — for a set of fields
observed over at least two seasons. Roughly 50–60 plots with two measured
seasons each would suffice, given the interval widths observed here. The
test is then simply whether within-plot season ordering clears chance on
measured labels where it does not on recalled ones.

Until that exists, every statement about seasonal deterioration in this
project is reported as **not demonstrated**, not as refuted.

## 3. Verification decisions on genuinely independent plots — DONE on a fresh axis

The plot-grouped confirmatory test is spent. But the plot axis is not the
only one, and it is not the one a deployment faces. `src/forward_eval.py`
trains on seasons **sown before 1 June 2020** and tests on those sown after:
184 train, 135 test field-seasons. This split has never been used for any
decision in this project.

| estimator | MAE | 95 % CI | r |
|---|---:|---|---:|
| global mean | 355.9 | [298, 416] | 0.000 |
| **causal history** | **200.1** | **[159, 240]** | **0.770** |
| satellite model | 345.9 | [290, 403] | 0.252 |

Paired: **history beats satellite by 145.7 kg/acre, 95 % CI [−198.4,
−92.8]**. Satellite barely separates from the global mean across a year
boundary (345.9 vs 355.9) and its correlation falls to r = 0.252.

**The decision, 28 loss seasons of 135, 33 % verification budget:**

| strategy | AUROC | 95 % CI | recall | losses caught |
|---|---:|---|---:|---:|
| random | 0.500 | — | 33 % | 9.2 |
| satellite model | 0.657 | [0.528, 0.773] | 50 % | 14 |
| **history-first hybrid** | **0.904** | **[0.832, 0.964]** | **89 %** | **25** |

Hybrid minus satellite: **+0.247 AUROC, 95 % CI [+0.128, +0.374]** — the
interval excludes zero.

**So the decision does beat simple alternatives, and by a wide margin.** It
catches 25 of 28 losses where random catches 9 and the satellite model
catches 14.

### The limit of this result, stated plainly

The test plots were all seen during training, in their earlier season. This
evaluates **a new season at known fields**, not a new field. That is the
right question for the history-first strategy, which exists precisely for
fields with a record — but no claim about unseen fields follows from it. The
plot-grouped evidence remains the authority there, and it failed.

Relatedly, history covers 100 % of this test set only because every plot has
a 2019 season. In a genuine first deployment year the coverage would be 0 %
and the system would fall back entirely to satellite, where AUROC is 0.657
with an interval that nearly touches chance.

---

## What these three together establish

The system makes a decision a simpler baseline cannot — but the simpler
baseline is the satellite model, not the history rule. Verification ranking
at 89 % recall on a fresh temporal split is a real, useful capability. It is
delivered by averaging past harvests, with satellite imagery contributing
only in the cold-start case where nothing else is available.

That is a defensible operational recommendation and an uncomfortable one for
a satellite-monitoring paper, which is why it is stated this way.

---

## 4. Follow-up: does the satellite component earn its place? — NO

Added after review pointed out that §3's headline had not been compared
against a history-only system.

**The headline was mis-attributed.** On the forward-in-time split, causal
history covered **100 % of test field-seasons**, so the history-first hybrid
and a history-only baseline were *the same predictor*. The AUROC 0.904
belongs to the history table. The satellite model contributed nothing to it,
and reporting it as a property of SICKLE-Guard was wrong.

Tested where a fallback can matter — grouped CV, 188 of 319 with history and
**131 genuine cold starts** — against a system using the area average for new
fields and no imagery anywhere (`src/does_satellite_add.py`):

| | AUROC | 95 % CI | recall @ 33 % |
|---|---:|---|---:|
| history-only, no imagery | **0.838** | [0.776, 0.889] | 72 % |
| hybrid (SICKLE-Guard) | 0.824 | [0.745, 0.891] | 72 % |

| paired difference | delta | 95 % CI |
|---|---:|---|
| ranking AUROC | −0.014 | [−0.058, +0.022] |
| MAE, cold-start fields only | −14.3 kg/acre | [−30.4, +3.6] |
| MAE, overall | −5.9 kg/acre | [−12.5, +1.5] |

Every interval spans zero, and on ranking the history-only system is
**nominally ahead**. The satellite component adds nothing measurable.

### What changed in the product as a result

The cold-start branch no longer presents a number as an estimate. It now
reads: *"NO RELIABLE ESTIMATE. No past harvest on record... it was NOT shown
to beat simply using the area average (gain 14 kg/acre, interval −30 to +4,
includes zero). Do not act on the number itself."*

Satellite is retained only because a new field has no alternative, not
because it was shown to help.

### On the verified tick

Review was right that a green tick only reports what the config claims. The
app now prints the **source** beside the entry — the PIB release it was
checked against — and states that a tick in an app proves nothing on its
own and must be confirmed against that source or with the farmer's bank.

### What this leaves

The useful capability is real and survives: **72 % of losses caught at a
33 % verification budget, AUROC 0.838 [0.776, 0.889]**. It is delivered by a
table of past harvests. That is worth building and worth saying plainly.
