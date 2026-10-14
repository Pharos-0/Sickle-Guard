# A standard safeguard framework, checked against this project's measurements

A generic design framework for satellite agriculture proposes three failure
modes, three safeguards and three prototype features. Every one of them is
reasonable a priori. **Five of the nine have already been tested here, and
three of those failed.** One would actively make the prototype worse.

This file records which is which, so design effort goes where the evidence
says it can succeed. Each row cites the file that produced the number.

Legend: **supported** — measured and it holds. **failed** — measured and it
does not. **untestable here** — the data cannot answer it, stated rather
than guessed.

---

## 1. Where predictions fail

### 1a. Small, irregular fields and mixed cropping — **not supported here**

The intuition is that small or awkwardly shaped fields are harder. In this
benchmark field size does not predict per-field accuracy at all:

> Spearman(pixels contributed, per-field IoU) = **−0.043**, 95 % CI
> −0.384 to +0.294 — `sickle-project/results/field_distributions.json`

What field size *does* control is a field's **weight** in the pooled
metric: one validation field supplies 47.3 % of all scored pixels, and
removing it alone moves the published figure by up to 0.261. So size
matters enormously for what the benchmark *reports* and not at all for
whether a given field is predicted well.

Mixed cropping is **untestable here**: SICKLE's crop-type task is binary
paddy against non-paddy, one delta, one cropping system.

### 1b. Cloud and temporal blinding — **supported**

This is the one failure mode the ablation resolves. Adding Sentinel-1
radar, which penetrates cloud, is one of only three rungs whose paired
interval excludes zero:

> v4 to v5, adding radar: **+0.1860** macro IoU, 95 % CI +0.1120 to +0.2600
> — `sickle-project/results/ladder_power.json`

Cloud is a real and measurable cost, and the mitigation the framework
suggests is the one that worked.

### 1c. Spectral signatures that look healthy — **supported, and worse than stated**

The framework describes imagery that looks fine while the crop is not. The
measured version is stronger: the model is largely not reading the season
at all.

> Feeding a **different season** from the same plot keeps 83 % of the
> ranking skill (AUROC 0.717 real, 0.679 after the swap, 0.460 for a
> different-plot control) — `results/metrics/season_swap.json`

So the problem is not only that a green field can hide damage. It is that
the model's apparent skill is mostly site identification, and asked
directly whether a season deviated from the field's own history it returns
AUROC 0.586–0.598 with every interval spanning chance.

---

## 2. Safeguards

All three proposed safeguards have been implemented and measured. **None is
informative on this data.** Nine candidate signals have now been tested on
yield error, the last being a water/vegetation decoupling index at the
transplanting window proposed in a later architecture review
(`src/mndwi_screen.py`, AUROC 0.512, 95 % CI [0.276, 0.690]); `results/metrics/reliability.json` records the verdict as *no signal
demonstrated better than chance*.

| safeguard proposed | signal tested here | AUROC | 95 % CI | verdict |
|---|---|---:|---|---|
| **2a** confidence / uncertainty | `quantile_width` | 0.531 | [0.319, 0.717] | failed |
| **2a** model disagreement | `rf_spread` | 0.672 | [0.480, 0.842] | failed |
| **2a** seed disagreement | `seed_spread` | 0.444 | [0.282, 0.615] | failed |
| **2a** released-ensemble dispersion | yield screen | 0.311 | [0.103, 0.634] | failed, below chance |
| **2b** anomaly vs this field's history | bad season, 3 definitions | 0.586–0.598 | all span 0.5 | failed |
| **2c** too few clear observations | `low_n_clear` | 0.575 | [0.371, 0.756] | failed |
| **2c** too few pixels | `small_n_px` | 0.559 | [0.375, 0.740] | failed |
| **2c** out-of-distribution imagery | `ood_distance` | 0.569 | [0.368, 0.768] | failed |

Two specifics worth separating out:

**Prediction intervals are well calibrated and still useless per field.**
Quantile intervals reach **90.3 %** empirical coverage at a nominal 90 %,
which looks like success, while their width correlates with realised error
at only **r = +0.10**. They are right on average and carry no information
about *which* field is wrong. Coverage is not discrimination, and reporting
coverage alone would have hidden this.

**The ensemble dispersion signal was pre-registered and still failed.** The
protocol in `sickle-project/configs/yield_screen_protocol.yaml` was frozen
before any result was computed, and the result is reported at the same
prominence a positive one would have received. The members correlate at a
median of 0.92, so 24 checkpoints are not 24 independent opinions.

**What did work**, and it is the exception: on the *crop-type* task, seed
disagreement and low confidence do rank unreliable fields above chance
(AUROC 0.806 and 0.823, intervals excluding 0.5) — but they are not
separable at 36 fields from baselines as crude as ranking by field size,
and seed disagreement requires training the model several times. See
`sickle-project/results/triage/screen_validate.json`.

---

## 3. Prototype features

### 3a. Risk-weighted triage toward vulnerable smallholders — **would make it worse here**

This is the recommendation to check carefully before building. Ranked on
the held-out test split:

| triage rule | AUROC | 95 % CI |
|---|---:|---|
| smallest field first | **0.235** | [0.14, 0.34] |
| largest field first | 0.765 | [0.66, 0.86] |

Visiting the smallest fields first is **strongly anti-correlated** with
finding below-threshold yields. In the yield-labelled subset the below-threshold
field-seasons average **0.95 acre** against **0.79** for the rest, and area
correlates with yield per acre at **−0.29**.

Two things must be said about that before it is generalised:

1. **Every field here is a smallholding.** Across the full released index
   of 2,164 labelled rows the mean plot is **0.351 acre**, the median
   0.190, the maximum **4.11**, and 83 % are at or below half an acre.
   Only 0.8 % reach two acres and none reaches five. Within the
   yield-labelled subset used above the mean is 0.79 acre, because plots
   with recorded yields are larger than the dataset as a whole — a
   selection effect worth stating, and the reason the two figures differ.
   Either way, "largest" means about one acre, not a commercial farm. The
   framework's contrast between a vulnerable smallholder and a large
   insured commercial operation **does not exist in this dataset** and
   cannot be tested with it.
2. **Prioritising by consequence is a policy choice, not a detection
   claim.** Choosing to visit smallholders first because the harm of a
   missed loss is greater is entirely defensible. What the numbers forbid
   is justifying it as *finding more problems* — here it finds fewer.

The honest design is to keep the two separate: rank by expected
below-threshold yield, and let a human apply a consequence weighting on
top, with the cost of that weighting shown.

### 3b. A do-no-harm interface — **implemented, and stronger than proposed**

The framework asks for "Insufficient Data – Ground Verification
Recommended" instead of a green *Healthy* badge. The prototype goes
further: **there is no healthy state at all**, even when data is good. A
field is flagged or not flagged, and *not flagged* always states that this
is not evidence of no loss.

The reason is in the numbers. On the held-out test split **3 of 11**
below-threshold fields are not flagged. A reassuring badge would have
reached those farmers.

One correction the framework's wording would have led into: the prototype's
"no reliable estimate" warning was tested and is **not calibrated** —
warned field-seasons are not reliably worse (+41 kg/acre, 95 % CI −58 to
+130, and the sign reverses over all 319). The text now says the warning
means *there is no record for this field*, not *this estimate is worse*.
Both interfaces are covered by `tests/test_app_interface.py`.

### 3c. Dynamic routing for verification — **untestable here**

There are no coordinates, and this was checked rather than assumed. The
tabular index carries `PLOT_ID`, `RIVER_PART`, season, dates, `AREA`,
`VARIETY` and `YIELD`, nothing positional beyond a named sub-region. The
mask rasters are plain TIFFs: reading the tags of `masks/10m/0.tif` returns
**no GeoTIFF tags at all** — no CRS, no model tiepoint, no pixel scale. The
imagery is delivered as per-plot `.npz` cubes already cut to the plot, so
the geometry that would place a plot on the ground was removed upstream.

No road network, no travel times, no agent locations either. Routing cannot
be built or evaluated from this data, and claiming it as future work
without saying that would be misleading.

---

## What this means for the design

Of the framework's three safeguards, **all three fail on yield** and one
partially works on crop type. That is not a reason to abandon safeguards.
It is the reason the prototype is built the way it is:

> Because no confidence signal could be shown to identify an unreliable
> field, the prototype does not show one. It leads with the field's own
> past harvests, refuses to reassure anyone, and names each limit where a
> user will see it.

A system that displayed a confidence score here would be displaying noise
with the authority of a number. The measured absence of a working
safeguard is itself the design constraint, and it is the part of this work
that generalises: **before shipping an uncertainty display, test whether
it discriminates — coverage alone will not tell you.**

---

A follow-on proposal specified a concrete five-component pipeline — cloud
gatekeeper, feature alignment, attention models beside a passive safeguard
engine, a Kafka triage broker and a dispatch map. That is assessed
component by component in `ARCHITECTURE_ASSESSMENT.md`, which also records
two things this file did not check: Sentinel-2 in SICKLE **does** ship
`SCL` and `MSK_CLDPRB`, so a cloud gate is buildable; and the mask rasters
carry **no GeoTIFF tags at all**, so a dispatch map is not.
