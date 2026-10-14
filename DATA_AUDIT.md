# Data audit — SICKLE yield subset

Phase 1 of the yield / reliability / insurance-simulation study. Every number
here was read from the files on 2026-10-11, not taken from the proposal or
from the earlier audit paper. Where the working assumption was wrong it is
corrected and the correction is marked.

Source: `D:/FPD/sickledataset extracted/` — `sickle_dataset_tabular.csv`,
`images/{L8,S1,S2}/npy/<UNIQUE_ID>/*.npz`, `masks/{3m,10m,30m}/<UNIQUE_ID>.tif`.

---

## 1. Headline counts — the proposal's figures are CONFIRMED

| Split (as released) | Yield-labelled field-seasons | Independent plots |
|---|---:|---:|
| train | 282 | 117 |
| val | 37 | 14 |
| test | 0 | 0 |
| **total** | **319** | **131** |

"Yield-labelled" means `YIELD` parses as a number and is strictly positive.
The released test split carries no labels of any kind, so it is unusable and
plays no part in this study.

All 319 rows are **Paddy**. Years 2019–2020 only.

Field-seasons per plot: 101 plots contribute 2, 27 contribute 4, 3
contribute 3. **No plot contributes only one season**, which matters for the
grouped split — every plot carries at least two correlated observations.

No duplicate `(PLOT_ID, STANDARD_SEASON, YEAR)` triples.

## 2. What the yield label actually is — two corrections

### 2.1 YIELD is a per-acre density, NOT a field total

`AREA` spans only 0.50–1.25 acres, and `YIELD` is **uncorrelated with area**:
Pearson −0.290, Spearman −0.252. A total would scale with area.

| | min | median | max |
|---|---:|---:|---:|
| YIELD (kg/acre) | 800 | 1,800 | 3,840 |
| AREA (acres) | 0.50 | 1.00 | 1.25 |

Median 1,800 kg/acre = **4,448 kg/ha**, which sits inside the plausible Tamil
Nadu paddy range. Treating `YIELD` as a total would imply a median 2,040
kg/acre once divided by area and would misstate the physical quantity.

**Consequence for modelling:** the target is a density. Do not multiply by
area, and do not describe predictions as a harvested mass.

### 2.2 Mask band 5 holds yield for EVERY plot in the patch

A patch contains several surveyed plots. Summing band 5 over the whole patch
sums all of them and is wrong.

| UNIQUE_ID | CSV YIELD | whole band-5 sum | sum where band0 == PLOT_ID |
|---|---:|---:|---:|
| 2060 | 1,760 | 1,760 | 1,760 |
| 2066 | 1,920 | **10,400** | 1,920 |
| 2067 | 1,800 | **11,520** | 1,800 |

Restricting to the target plot reproduces the CSV value **exactly on 319 of
319 rows**. The CSV is therefore the authoritative label and the mask is a
spatial rendering of it.

### 2.3 Provenance of the label

The SICKLE paper describes these as farmer-reported productivity figures
recorded in a ground survey, recalled after the season. They are **not**
crop-cutting experiment measurements, and the manuscript must not call them
that. `VARIETY` is populated on all 319 yield rows while blank on 2,051 of
2,370 rows overall, so the yield subset is the detailed-survey block.

## 3. Imagery coverage — complete, no gaps

Every one of the 319 field-seasons has imagery from all three sensors.

| Sensor | Acquisitions per sample (min / median / max) | Samples with none |
|---|---|---:|
| Sentinel-2 | 30 / 60 / 146 | 0 |
| Sentinel-1 | 11 / 17 / 23 | 0 |
| Landsat-8 | 8 / 11 / 22 | 0 |

Bands present per acquisition: S2 has 12 reflectance bands plus `SCL`,
`MSK_CLDPRB`, `AOT`, `WVP`; S1 has `VV`, `VH`, `angle`; L8 has 7 surface
reflectance bands plus thermal and QA layers.

Patch geometry: S2/S1 ≈ 33×33 px at 10 m (≈330 m across), L8 12×12 at 30 m.
**A 0.38-acre plot is about 15 Sentinel-2 pixels and under 2 Landsat
pixels.** For the worked example the target plot occupies 37 of 1,122 pixels,
3.3 % of the patch; the rest is other land. This is the binding physical
constraint of the dataset and it does not change with sample size.

## 4. Two findings that constrain the study design

### 4.1 RIVER_PART is blank for the entire yield subset, and unrecoverable

All 319 yield rows have an empty `RIVER_PART`, and **not one of the 131 plots
has a region recorded on any other row**. The yield plots occupy IDs
300–448; the plots that do carry regions are IDs 1–249, a disjoint block used
for the crop-type task. There is no overlap.

Consequences, both hard:

- **No geographic stratification or regional hold-out for the yield study.**
- **No aggregation to Gram Panchayat or any administrative unit.** The
  insurance simulation therefore cannot reproduce the YES-TECH Insurance Unit
  procedure, which operates at GP level. This is the single biggest
  limitation on Phase 6 and must be stated there, not worked around.

### 4.2 Almost all yield variance is BETWEEN plots

| Component | kg/acre |
|---|---:|
| Overall SD | 433 |
| SD of plot means (between plots) | **413** |
| Median within-plot range across seasons | 160 |

Roughly 95 % of the spread is between plots rather than between seasons of
the same plot. Two implications:

- The grouped split is not a formality. A model evaluated on unseen plots
  must predict the between-plot component, which is the hard part.
- A model that merely learns each plot's level would score well under a
  random row split and fail here. Any random-split result in the literature
  on this data should be read with that in mind.

## 5. Data quality

- No duplicate field-season keys.
- No missing `CROP`, `AREA`, `VARIETY`, sowing or harvest dates on yield rows.
- Yield range 800–3,840 kg/acre; no zeros or negatives among the 319 (rows
  with `YIELD` of 0 are treated as unlabelled, not as a zero harvest).
- Season coverage: sep-feb 178, oct-mar 82, may-sep 51, jun-oct 6, dec-may 2.
  The last two are too small to analyse separately.
- Varieties: BPT 56, CR 1009 53, CR 51, Kalsar 31, ADT 36 25, CO 51 25,
  ADT 37 18, IR 20 14.

## 6. Is 131 plots enough?

For RQ1 and RQ2, yes, with care. A grouped split of roughly 80/25/26 plots
leaves ~26 independent test plots carrying ~63 field-seasons. That supports
comparing a handful of models with grouped bootstrap intervals. It does not
support a large architecture search, and the protocol must say so.

For RQ3, marginal. The number of high-error plots at any sensible threshold
will be small, and the earlier crop-type screening experiment on 14 plots
produced an interval spanning chance. Expect wide intervals and design the
evaluation to report them honestly rather than to produce a headline.

For RQ4, the binding constraint is §4.1, not sample size: with no geography
the simulation cannot operate at the administrative unit the real scheme
uses. It can only compare plot-level decisions under an explicitly
hypothetical rule.

## 7. Verified facts for downstream phases

| Item | Value |
|---|---|
| Target column | `YIELD`, kg per acre, farmer-reported |
| Target range | 800–3,840 kg/acre |
| Grouping key | `PLOT_ID` (131 distinct, IDs 300–448) |
| Field-season key | `UNIQUE_ID` |
| Sensors available | S2, S1, L8 — all present for all 319 |
| Region label | **absent and unrecoverable** |
| Plot area | 0.50–1.25 acres, weakly informative |
| Season field | `STANDARD_SEASON`, 5 values, 2 usable |
