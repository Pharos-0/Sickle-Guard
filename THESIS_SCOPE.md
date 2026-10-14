# What goes where: thesis, paper, and what stays out of both

## The objective, as stated

> We identify conditions under which satellite-based agricultural
> predictions can be unreliable, empirically evaluate safeguards for
> detecting those failures, and develop a decision-support prototype to
> help prioritize verification while guarding against false reassurance.

This is the canonical wording. `src/check_claims.py` enforces that it
appears here verbatim and that no document in this project claims anything
stronger.

Revised 2026-10-12 at the author's request. The earlier canonical sentence
ended "to prioritize verification while minimizing the risk of false
reassurance". Two changes, both kept no stronger than the evidence:
"help prioritize" (the prototype supports an inspection office's choice;
Experiment A showed its edge over the office's own register is not
demonstrated) and "guarding against false reassurance" (the same property
of the artefact, now also carried by blind task sheets in the inspection
workflow). "Conditions under which" and "empirically evaluate" were kept
for the reasons below.

Three phrases in it are load-bearing, and each was chosen because an
earlier draft overreached:

- **conditions under which**, not *where*. The audit identifies the
  circumstances that make a prediction unreliable -- evaluation batch size,
  a pooled metric dominated by one field, the absence of a prior record --
  not merely a list of fields that failed.
- **empirically evaluate**, not *test*. Nine yield-reliability signals
  have been evaluated and none demonstrated useful discrimination. The
  contribution is the evaluation, and it is a contribution whether or not a
  safeguard succeeds.
- **guarding against false reassurance**, not *protecting farmers
  from* it. The first is a property of the system and is demonstrated: the
  report has no reassuring state, enforced by tests; inspection task sheets
  withhold the estimate; and 3 of 11 below-threshold fields on the held-out
  split go unflagged, which is exactly who a reassuring message would have
  reached. The second is an outcome for a person and **nobody has used
  this**.

### Evidence status, clause by clause

| clause | status | what carries it |
|---|---|---|
| identify conditions of unreliability | **supported** | recalled labels run below the state's crop-cutting yields, and the satellite model's estimates did not move in a season of delta-wide state crop-damage relief; none of the 24 released yield models is shown to beat a constant (all 24 have higher observed error, but no plot-clustered interval excludes zero); one field holds 47.3 % of the pooled metric and removing it moves the headline by up to 0.261; batch size moves a frozen model by 32.7 IoU points |
| empirically evaluate safeguards | **supported as an investigation** | nine yield signals evaluated, none discriminating; the crop-type screen carries information beyond field size (p = 0.0003) while its standalone ranking gain sits on a lower bound of exactly zero |
| prototype that helps prioritize verification while guarding against false reassurance | **partially supported** | the flag separates 40 % from 7 % below threshold; no reassuring state exists; but the hybrid is not demonstrated to beat looking up the last harvest, and no human outcome is measured |

The third row is the honest limit of the whole project. Ranking better
than an alternative is not the same as anyone being better off, and
nothing here bridges that gap.


Three bodies of work exist. They are not the same contribution and should
not be merged to look larger.

| | what it is | where it lives | status |
|---|---|---|---|
| **A. Benchmark audit** | Reproducibility and field-level reliability of the SICKLE benchmark | `D:/FPD/sickle-project` | manuscript complete, all checks passing, 9 pages |
| **B. SICKLE-Guard** | Field assessment and verification-prioritisation prototype | `D:/FPD/sickle-guard` | prototype complete, now evaluated as a decision procedure |
| **C. Data-regime sweep** | How much labelled data each modelling choice needs | `sickle-project/outputs/crop_type/sweep` | 34 of 90 runs done, paused, resumable |

---

## The thesis: A then B, with the arrow between them

The combined claim is **not** "satellite imagery enables field-level loss
detection." Nothing here supports that, and B exists partly because A
showed it to be false.

The claim the evidence does support:

> Field-level satellite crop prediction fails in specific, measurable ways
> that aggregate benchmark scores conceal. Those failures can be
> characterised; on crop type, though not on yield, some of them carry a
> label-free signal that survives conditioning on field size; and a
> decision-support tool can be built to remain useful and honest **while
> its central model is weak** — by leading with the field's own record,
> refusing to reassure, and naming each limit where a user will see it.

That claim needs all three chapters:

1. **Discovery (A).** Where predictions fail, how far the pooled number is
   from a field, and which safeguards survive testing. Three defects, one
   misreadable metric, and the finding that none of the 24 released yield
   models is shown to beat a constant (corrected 2026-10-13: all 24 have
   higher observed error, but with 14 plots no gap is distinguishable from
   zero).
2. **Response (B).** A prototype whose every design decision traces to a
   measurement from A. The trace is `EVIDENCE_TO_DESIGN.md`.
3. **Verification (B, honestly).** Whether the prototype does what it
   claims, tested against alternatives an officer could use instead, on
   fields never used to build it. `src/evaluate_prototype.py`.

Chapter 3 is what makes the thesis more than two projects stapled
together. It is also where most of the uncomfortable results are.

---

## The IEEE paper: A only

The audit stands alone: a complete reproducibility study with a
pre-registered negative result, declared tolerances, and a one-command
rebuild. Adding the prototype to it would weaken it, for three reasons:

- The prototype's strongest evaluation is still ranking on 27 held-out
  plots. That is a thesis chapter, not a second contribution in a
  conference paper.
- The two use different data regimes — 36 validation fields for crop type,
  319 field-seasons at 131 plots for yield — and the paper would have to
  explain the shift twice.
- The audit is already 9 pages. The prototype would need four more and
  would make the paper about two things.

**If the prototype is published, it is a second paper**, and its honest
framing is a system paper about designing under a weak model, not a
performance paper.

---

## Under which conditions would it help at all?

`SYNTHETIC_CONDITIONS.md` maps that, because the real data cannot: SICKLE
fixes the share of fields with a record, the season-to-site variance ratio
and the recall error, and those are exactly the things that decide whether
triage is worth doing. The generator is calibrated to four measured
quantities and the run aborts if it misses them.

Across **31 settings in seven sweeps**, requiring a policy to beat the
runner-up by 0.02 recall before it counts: **20 are ties**, the register
wins 8, the prototype wins 3, and the history-first policy wins none. Most
differences are inside the noise, and that is the first thing to say. At
the calibrated operating point the register wins 0.73 against 0.64.

The conclusion, with its evidence boundary made explicit:

> Under the evaluated conditions, the prototype is weakest where
> historical records are unavailable — a setting in which its
> satellite-based estimates have not demonstrated sufficient added value
> to reliably prioritize verification.

That is narrower than saying the system is weakest for everyone without
records. It joins the empirical baseline comparison to the simulated
sweeps without treating the simulation as field validation.

Simulated and empirical results are kept apart. `check_claims.py` enforces
that the synthetic page declares itself in its opening line, and that no
document anywhere writes simulated recall as observed.

## Safeguards proposed versus safeguards that work

`SAFEGUARD_FRAMEWORK_MAPPING.md` checks a standard design framework for
satellite agriculture against this project's measurements. Of its nine
recommendations, five had already been tested here and three failed; one
would make the prototype worse on this data. That file is the short answer
to "why does the tool not show a confidence score" — because every
candidate confidence signal was measured and none discriminated.

## What must not be claimed in either

These are the specific overstatements the evidence rules out. Each is
something a reader could reasonably infer if the writing is loose.

| tempting claim | why it is not available |
|---|---|
| "detects crop losses" | 60 % of labelled losses are chronically low plots. On the 20 plots where the answer is not already in the record, AUROC 0.626 with the interval including chance. |
| "72 % of farmers suffering damage would be identified" | The target is yield below 80 % of a training-split mean, on farmer recall. It is a relative cut against the area, not a measurement of damage. |
| "the satellite model powers the tool" | Satellite adds ΔAUROC −0.014 (−0.058 to +0.022) over history alone. Looking up the field's last harvest ranks as well as the whole system. |
| "warns you when the estimate is unreliable" | The warning fires on missing records, not on error. Warned fields are not reliably worse: +41 kg/acre, interval −58 to +130. |
| "validated with farmers" | Nobody has used it. 67 passing tests prove specified behaviour, not outcomes. |
| "ready for PMFBY / YES-TECH" | No Insurance Unit identity exists in this data, so nothing here can reproduce a scheme decision. The audit covers a research benchmark, not any operational model. |

---

## Before submission

Open, and needing a person rather than more computation:

1. **Author name, email, target venue.** The manuscript title block still
   carries placeholders. Venue also decides whether 9 pages is a problem.
2. **Measured yields.** Every yield conclusion in both halves is bounded by
   farmer recall. Crop-cutting measurements would separate model error from
   recall error; this dataset has none.
3. **Independent reproduction.** `build_all.py` shows the manuscript agrees
   with the artefacts in this repository. It cannot show that someone else
   gets the same numbers on their own machine.
4. **Resume the sweep (C)** when the ladder seeds finish — one command, in
   `sickle-project/docs/SWEEP_PAUSED.md`.
5. **A comprehension check**, if the prototype chapter is to claim anything
   about farmers. Even five readers asked what a report means would be
   more evidence about the wording than the entire test suite provides.
