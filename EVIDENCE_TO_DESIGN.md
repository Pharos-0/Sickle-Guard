# From evidence to design to evaluation

The thesis has two halves. The first finds out where satellite crop
prediction is unreliable. The second builds something that behaves
sensibly given those findings. A combined thesis is only coherent if each
design decision can be traced back to a measurement that motivated it, and
forward to an evaluation that tests whether it worked.

This file is that trace. Every row names the finding, the file it came
from, the thing the prototype does differently because of it, where that
lives in the code, and the evaluation that supports **or refutes** it.

Three rows end in a refutation. They are kept in the table, in the same
form as the rest, because a design decision that did not survive its own
test is a result.

**Reading the verdicts.** *Supported* means an interval excludes the null.
*Refuted* means the measurement contradicts the reason the feature exists.
*Honest limit* means the design is a reaction to something that could not
be measured at all.

---

## 1. Report per field, never an area aggregate

| | |
|---|---|
| **Finding** | A pixel-pooled IoU of 0.8665 is close to the average field, but the errors are concentrated: 4 of 36 fields fall below 0.50 while most are mapped almost perfectly (median field 0.957, each field scored on the crop it holds). One validation field holds 47.3 % of all scored pixels; removing it alone moves the pooled figure by up to 0.261. |
| **Source** | Audit paper §V-C; `sickle-project/results/field_distributions.json`, `field_size_val.json` |
| **Design response** | The prototype has no district or area summary anywhere. Every output is one field-season, with that field's own number. |
| **Implemented in** | `src/farmer_report.py`, `src/assess_field.py`, `app.py` |
| **Evaluation** | On the held-out test split the flag separates fields that really are below threshold from those that are not: 40 % against 7 %, difference +33 % (95 % CI +1 % to +62 %). |
| **Verdict** | **Supported.** `src/evaluate_prototype.py`, question C |

## 2. Put the field's own history first, satellite second

| | |
|---|---|
| **Finding** | Averaging a field's past harvests beats the satellite model by 145.7 kg/acre. Tested where a fallback could matter, the satellite component adds nothing measurable: ΔAUROC −0.014 (95 % CI −0.058 to +0.022). |
| **Source** | `results/metrics/does_satellite_add.json`, `forward_eval.json` |
| **Design response** | The report uses causal history — harvests that finished before this season was sown — wherever it exists, and falls back to satellite only for a field with no record. |
| **Implemented in** | `src/farmer_report.py::causal_history` |
| **Evaluation** | Out-of-fold over 319 field-seasons, history-only ranks at AUROC 0.838 and the shipped hybrid at 0.824. On the held-out test split: 0.797 against 0.807. Paired on the same resampled plots, hybrid minus last-harvest is +0.004 (95 % CI -0.092 to +0.078) on test and -0.014 (-0.059 to +0.024) out-of-fold. |
| **Verdict** | **Inconclusive for the satellite half.** Putting history first is supported by the MAE gap; whether the satellite fallback helps or hurts is not resolved at 27 held-out plots, and the interval is narrow enough out-of-fold to exclude a large benefit. `src/evaluate_prototype.py` question A, `src/audit_evaluation.py` check 2 |

## 3. Never claim this season was bad for this field

| | |
|---|---|
| **Finding** | 83 % of the ranking skill survives when the model is fed a *different season* from the same plot (AUROC 0.717 real, 0.679 after the swap, 0.460 for a different-plot control). Asked directly whether a season deviates from the field's own history, AUROC is 0.59–0.60 with every interval spanning chance. |
| **Source** | `results/metrics/season_swap.json`, `site_vs_season.json`; `DISCOVERY.md` |
| **Design response** | The report states in plain words that the system **cannot** tell whether this particular season was bad for this field, and says the test was run and failed. |
| **Implemented in** | `src/farmer_report.py`, `app.py` (persistent warning, not a footnote) |
| **Evaluation** | Pre-declared acceptance criterion C, scored across three definitions of a bad season. |
| **Verdict** | **Supported as a limit, FAIL as a capability.** `ACCEPTANCE_REPORT.md` row C |

## 4. Describe the tool as ranking usual yield level, not detecting losses

| | |
|---|---|
| **Finding** | 60 % of labelled losses sit at 12 plots that are below threshold in *every* observed season. On the 20 plots where the answer is not already in the field's record, AUROC falls to 0.626 (95 % CI 0.386 to 0.887), which includes chance. |
| **Source** | `results/metrics/label_validity.json` |
| **Design response** | Every description of the tool — README, sidebar, report text — says it ranks fields by their usual yield level. The phrase "loss detection" was removed from all live claims. |
| **Implemented in** | `README.md`, `app.py` sidebar caption, `src/farmer_report.py` evidence block |
| **Evaluation** | The subset decomposition itself; there is no separate test, because the finding *is* the test. |
| **Verdict** | **Supported.** The honest description is the one the evidence allows. |

## 5. No reassuring state, ever

| | |
|---|---|
| **Finding** | Not a measurement — an asymmetry. A false "your crop is fine" can stop a farmer filing inside a claim window that does not reopen; a false "go and check" costs a phone call. The two errors are not interchangeable, and no metric in this project weighs them. |
| **Source** | Reasoning stated openly in `README.md`; the 72-hour window itself is **unverified** (`configs/scheme_contacts.yaml`) |
| **Design response** | A field is flagged or not flagged. *Not flagged* always prints that this is not evidence of no loss. Step 1 of every report is *report damage now, do not wait for this*. |
| **Implemented in** | `src/farmer_report.py`; enforced by `tests/test_farmer_report.py`, a negation-aware scan for reassuring language plus a meta-test proving the guard bites |
| **Evaluation** | On the held-out test split, **3 of 11** below-threshold fields are not flagged. Those are the farmers a reassuring message would have silenced. |
| **Verdict** | **Honest limit, and the miss count is the justification.** The rule cannot be validated against outcomes here, only shown to be necessary. `src/evaluate_prototype.py`, question C |

## 6. Warn when there is no record for the field

| | |
|---|---|
| **Finding** | A new field has no history, so the estimate falls back to satellite, which was not shown to beat the area average (cold-start MAE gain 14.3 kg/acre, 95 % CI −30.4 to +3.6). |
| **Source** | `results/metrics/does_satellite_add.json` |
| **Design response** | Cold-start reports print NO RELIABLE ESTIMATE and *do not act on the number itself*. |
| **Implemented in** | `src/farmer_report.py`, `app.py` warning state |
| **Evaluation** | Does the warning track real error? On the held-out test split, warned fields average 265.5 kg/acre of error against 224.8 for unwarned — difference +40.7 (95 % CI −57.7 to +129.8). Over all 319 field-seasons the difference runs the **wrong way**, −32.5 (95 % CI −83.3 to +15.4). |
| **Verdict** | **REFUTED as calibration.** The warning is honest about *why* it fires — there is genuinely no record — but it does not identify the field-seasons where the estimate is actually worse. It should be read as a statement about missing evidence, not as a predictor of error, and the text should say so. `src/evaluate_prototype.py`, question B |

## 7. Do not present per-field uncertainty intervals as discriminating

| | |
|---|---|
| **Finding** | Quantile intervals reach 90.3 % empirical coverage at a nominal 90 %, but their width is near-constant and correlates with realised error at only r = +0.10. Coverage without discrimination is a confidence band that is right on average and uninformative for any one field. |
| **Source** | `results/metrics/reliability.json` |
| **Design response** | The analyst view shows the interval with an explicit label that it does not discriminate; the farmer report does not show it at all. |
| **Implemented in** | `src/assess_field.py` |
| **Evaluation** | The coverage-versus-correlation split in the same file. |
| **Verdict** | **Supported as a limit.** |

## 8. Dispersion across models is not a reliability signal here

| | |
|---|---|
| **Finding** | Spread across the released yield checkpoints ranks field-season error at AUROC 0.31 (95 % CI 0.10 to 0.63) — no better than random referral, and the point estimate is below chance. The ensemble members correlate at a median of 0.92, so they are not independent. |
| **Source** | Audit paper §V-I; `sickle-project/results/triage/yield_screen.json`, protocol frozen before results |
| **Design response** | No "model confidence" number is shown to a farmer anywhere in the prototype. |
| **Implemented in** | absence, deliberately — `src/farmer_report.py` has no confidence field |
| **Evaluation** | The pre-registered screen itself, reported as a negative result at the same prominence a positive one would have had. |
| **Verdict** | **Supported by a negative result.** |

## 9. Print the source beside every scheme contact

| | |
|---|---|
| **Finding** | Not from the model. Giving a farmer a wrong helpline number is a direct harm, and a tick in an app only means the app's own config claims verification. |
| **Source** | `configs/scheme_contacts.yaml` |
| **Design response** | Only entries marked verified appear without a warning, and each prints the source so it can be re-checked. Unverified entries print `[UNVERIFIED — confirm first]` and the report tells the farmer to confirm with their bank, Common Service Centre or District Agriculture Officer. |
| **Implemented in** | `src/farmer_report.py::contacts`, `app.py` |
| **Evaluation** | 14447 checked against a PIB release of 08 Feb 2024, read directly. The WhatsApp number, the app and the 72-hour limit remain **unverified**: `pmfby.gov.in` returns an empty body to both a plain fetch and a rendered load. |
| **Verdict** | **Partly supported.** One entry verified, three not, and the interface says which is which. |

---

## What the table does not contain

No row connects a design decision to an outcome for a person, because no
such measurement exists. Nobody has used this. The strongest claim any row
makes is that a ranking or a flag separates recorded yields better than an
alternative does.

Three specific gaps follow, and they are the honest next steps rather than
caveats to be buried:

1. **Comprehension.** The no-reassurance wording is designed to stop a
   false sense of safety. Whether a farmer reading it reaches the intended
   conclusion has never been tested, and the design is a hypothesis about
   reading, evaluated only as a property of the text.
2. **Workflow.** Question A shows the ranking beats arbitrary visiting. It
   does not show that an officer using it visits differently, faster, or
   with better results than one using the register they already have —
   particularly since *looking up the field's last harvest* ranks just as
   well (AUROC 0.838 against 0.824 out-of-fold).
3. **The label.** Everything is scored against yield below 80 % of a
   training-split mean, computed on farmer recall. Separating model error
   from recall error needs crop-cutting measurements this dataset does not
   contain, and that single gap bounds every row above.

## The uncomfortable summary

Of the nine design decisions, **two are responses to the satellite model
not working** (rows 2 and 8), **three are responses to a metric or a label
meaning less than it appears to** (rows 1, 4, 7), **one has been refuted by
its own evaluation** (row 6), and **one cannot be resolved at this sample
size** (row 2's satellite half).

The prototype's main contribution is therefore not that satellite imagery
helps. It is that a system can be built to be useful and honest *while its
central model is weak* — by leading with the field's own record, refusing
to reassure, and naming each limit where a user will see it. That is a
defensible thesis claim. "Satellite imagery enables field-level loss
detection" is not, and nothing in this project supports it.
