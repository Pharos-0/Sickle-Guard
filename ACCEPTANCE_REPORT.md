# Final acceptance report

2026-10-11. Maps each objective to evidence. Statuses are PASS, PARTIAL,
FAIL or BLOCKED, each justified by a named artefact.

---

## B. What was discovered

The project set out to build a satellite yield predictor. The decisive
finding is that **for this task, on this benchmark, satellite imagery is
the wrong instrument for the job it was given.**

Three experiments, each with its own control, establish it:

1. **Season-swap** (`src/season_swap.py`). Feeding the model the wrong
   season's imagery of the *right* field costs only AUROC 0.038 (0.717 →
   0.679). Feeding it a different field's imagery collapses it to 0.460.
   **83 % of loss-detection skill is site identification, not crop
   observation.**
2. **Plot-history baseline** (`src/site_vs_season.py`). A baseline that
   sees no imagery at all and averages the field's previous harvests beats
   the full satellite pipeline by **80.5 kg/acre MAE, 95 % CI [+40.9,
   +124.8]** over 131 plots.
3. **Deviation modelling** (same script). Imagery cannot predict a field's
   departure from its own history: MAE 195.7 against 197.4 for predicting
   no deviation, paired CI [−8.8, +5.2].

| regime | best available estimator | MAE |
|---|---|---:|
| field with ≥1 prior season | **plot history, no imagery** | **197.4** |
| brand-new field, no history | satellite model | 277.9 |
| no information at all | global mean | 303.0 |

Satellite closes **24 %** of the gap between knowing nothing and knowing the
field. Its cold-start gain is real (−25.1 kg/acre, CI [−41.6, −8.8]) and
that is the only regime where it is the best option.

## C. Quantitative results

**Verification prioritisation**, 53 loss seasons of 319, 131 plots:

| strategy | AUROC | 95 % CI | recall @ 33 % budget |
|---|---:|---|---:|
| random | 0.500 | — | 33 % |
| satellite model | 0.717 | [0.618, 0.812] | 64 % |
| **plot history** | **0.884** | **[0.817, 0.936]** | **92 %** |

History beats satellite by **+0.168 AUROC, 95 % CI [+0.054, +0.283]** —
paired over plots, excluding zero.

**Bad-season detection** (deviation from the field's own history):

| definition | n | AUROC | 95 % CI |
|---|---:|---:|---|
| deviation < −10 % of history | 45 | 0.598 | [0.497, 0.693] |
| deviation < −20 % of history | 22 | 0.590 | [0.451, 0.717] |
| deviation < −1 SD | 29 | 0.586 | [0.460, 0.706] |

Every interval contains chance.

## D. Objective status

| # | Objective | Status | Evidence |
|---|---|---|---|
| A | Yield prediction | **PARTIAL** | Pipeline runs and beats the global mean out-of-fold (277.9 vs 303.0, CI excludes 0). But it loses to plot history by 80.5 kg/acre, and the frozen confirmatory test failed both primary criteria. `results/metrics/confirmatory.json` |
| B | Field-specific reliability | **FAIL** | Intervals are calibrated on average (90.0 % vs nominal 90 %) but near-constant: 7 distinct widths over 62 fields, SD 52 on a median width of 1254, correlated with realised error at r = +0.10. No disagreement signal cleared chance confirmatorily. `results/metrics/reliability.json` |
| C | Bad-season detection | **FAIL** | AUROC 0.586–0.598 across three pre-declared definitions, all CIs spanning chance. Within-plot season ordering 53.7 % [47.4, 60.0]. `results/metrics/site_vs_season.json` |
| D | Verification prioritisation | **PASS, but not by the model** | Plot history reaches AUROC 0.884 [0.817, 0.936] and 92 % recall at a 33 % budget. The satellite model reaches 0.717 and is significantly worse. The objective is met; the satellite component is not what meets it. |
| E | Decision support | **PASS** | `src/assess_field.py` runs per field, selects the estimator by regime, states its source, grades its own evidence, and refuses to assess seasonal deterioration. |
| F | Insurance relevance | **BLOCKED** | No Gram Panchayat identity exists for any of the 131 yield plots; no Threshold Yield, CCE yield or Sum Insured. `INSURANCE_ASSUMPTIONS.md` |
| — | Reproducibility | **PASS** | 40 tests, frozen split, `reproduce.sh`, leakage asserted in code. |

## E. Remaining limitations

- **Labels are farmer recall, not crop-cutting measurements.** Recall
  plausibly compresses toward a typical year, so some of the missing
  within-plot signal may be the label rather than the model. This cannot be
  separated without measured yields and is the single most important
  confound in the study.
- **131 plots.** The confirmatory test (27 plots) failed on precision, not
  direction; the subsampling curve suggests ~50–60 test plots would settle
  it.
- **Plot history needs persistent field identity and a prior season.** It
  is not deployable for an unseen field, which is exactly where satellite
  retains value.
- Findings are one delta, one cropping system, 2019–2020.

## F. Practical demonstration

```
cd D:/FPD/sickle-guard
D:/FPD/sickle-project/.venv/Scripts/python.exe src/assess_field.py --uid 2150
```

```
FIELD-SEASON 2150   plot 345   oct-mar 2020.0
  estimated yield      1440 kg/acre   [source: plot-history]
                       this field's own past harvests beat the satellite model;
                       the model would have said 1661 kg/acre
  90% interval         1200 to 2454   (+/-627)
                       near-constant across fields; NOT specific to this field
  vs benchmark         78% of 1838 kg/acre
  reliability          HIGHER model disagreement
  VERIFY IN FIELD?     YES

  SEASONAL DETERIORATION: not assessed.
```

Model output above; the held-out reference yield for that field-season was
1840 kg/acre, printed separately by the tool and never used in the estimate.

## G. Final verdict

| question | answer |
|---|---|
| Is the pipeline technically complete? | **Yes.** Ingestion → features → model → interval → ranking → export, 40 tests, one command. |
| Is yield magnitude estimation demonstrably useful? | **No.** It loses to the field's own history by 80 kg/acre and failed the frozen confirmatory criteria. Useful only cold-start, and modestly. |
| Can it identify a bad season rather than a poor field? | **No.** AUROC ≈ 0.59, every interval spanning chance. |
| Are field-level reliability assessments informative? | **No.** Calibrated on average, near-constant per field. |
| Does verification prioritisation beat alternatives? | **Yes — using plot history**, AUROC 0.884 vs 0.717 for satellite, difference CI [+0.054, +0.283]. |
| What is needed before deployment? | Measured yields to resolve the label confound; ~50–60 independent test plots; Insurance Unit linkage for any insurance use. |
| Is the manuscript ready? | **No.** Findings are sound and reproducible but exploratory out-of-fold on one benchmark. Needs replication on a second dataset with repeated field observations, ideally with measured labels. |

## The defensible contribution

Not a yield model. **A diagnostic and a negative result with a practical
consequence**: the season-permutation test, the plot-history baseline, and
the finding that on this benchmark a satellite pipeline is outperformed for
both yield estimation and verification ranking by averaging the field's own
past harvests. Work in this genre routinely reports a single pooled metric
over field-seasons without either control. Both are cheap, and both should
be standard.
