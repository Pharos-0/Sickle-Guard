# SICKLE-Guard

A field-level crop assessment and verification-prioritisation prototype,
built on the SICKLE paddy benchmark (Cauvery Delta, Tamil Nadu).

**It is a research prototype. It is not connected to PMFBY, is not an
insurer, and its output has no official standing.**

---

## What it does, and what the evidence says about each part

Everything below is measured, not asserted. The numbers come from two
held-out designs, neither touched by model selection: a forward-in-time
split (train on seasons sown before June 2020, test on those sown after;
`results/metrics/forward_eval.json`) and grouped cross-validation by plot,
which has more fields with no harvest record (`src/does_satellite_add.py`,
`src/label_validity.py`). See `THREE_VALIDATIONS.md`. The ranking AUROC in
the first row is from grouped cross-validation; the yield-estimate rows and
the 145.7 kg/acre comparison below are from the forward split.

| capability | measured | verdict |
|---|---|---|
| **Ranking fields by usual yield level** | AUROC 0.838 [0.776, 0.889] overall; 0.854 on chronic/never plots | **works** |
| **Detecting a loss event** (which season at a field went wrong) | AUROC **0.626 [0.386, 0.887]** on the 20 plots where the answer is not already in the record | **not demonstrated** |
| Yield estimate, field with history | MAE **200.1** kg/acre, r = 0.77 | usable |
| Yield estimate, new field | MAE 345.9 vs 355.9 for a constant, r = 0.25; not shown to beat the area average | **no reliable estimate** |
| "Was this season bad for this field?" | AUROC 0.59, interval spans chance | **not offered** |
| Per-field uncertainty interval | 90 % coverage but near-constant width; r = +0.10 with realised error | **uninformative, labelled as such** |

The uncomfortable finding, and the reason the tool is built the way it is:
**the satellite model is not what makes this work.** Averaging a field's own
past harvests beats it by 145.7 kg/acre [−198.4, −92.8] and by +0.247 AUROC
[+0.128, +0.374].

Pushed further, the satellite component does not survive at all. On the
forward split causal history covered **100 %** of test fields, so that
split's ranking AUROC of 0.904 (`forward_eval.json`) is the history table's,
and the satellite model contributed nothing to
it. Tested where a fallback can matter — grouped CV, 41 % cold starts —
against a history-only system using the area average for new fields:

| | AUROC | 95 % CI | recall @ 33 % |
|---|---:|---|---:|
| history-only, **no imagery anywhere** | **0.838** | [0.776, 0.889] | 72 % |
| hybrid (SICKLE-Guard) | 0.824 | [0.745, 0.891] | 72 % |

Difference **−0.014, CI [−0.058, +0.022]**; cold-start MAE gain −14.3,
CI [−30.4, +3.6]. Both span zero and history-only is nominally ahead.
**The satellite component adds nothing measurable** (`src/does_satellite_add.py`).
It is retained only because a new field has no alternative, and the report
now says the number should not be acted on.

## The one design rule

The two errors are not symmetric. A false "your crop is fine" can stop a
farmer filing inside a 72-hour window that does not reopen. A false "go and
check" costs a phone call.

So **the report has no reassuring state.** A field is flagged or not
flagged, and *not flagged* always prints that this is not evidence of no
loss. Step 1 of every report is *report damage now, do not wait for this*.

This is enforced by `tests/test_farmer_report.py`, which scans every report
for reassuring or entitlement-deciding language and fails the build if it
appears outside an explicit negation.

## Running it

```bash
# one field, with the full advice block
python src/farmer_report.py --uid 2063

# Tamil
python src/farmer_report.py --uid 2063 --lang ta

# which fields to visit first, under a budget
python src/farmer_report.py --village-list --budget 0.33

# the analyst view, with intervals and reliability reasons
python src/assess_field.py --split test --csv out.csv

# browser interface (needs streamlit)
streamlit run app.py
```

Interpreter: this project shares the venv at
`D:/FPD/sickle-project/.venv`. Substitute
`D:/FPD/sickle-project/.venv/Scripts/python.exe` for `python` on Windows.

## Reproducing the research

```bash
bash reproduce.sh
```

Runs splits → features → tests → ablation → paired tests → reliability →
insurance simulation → causal history → forward evaluation → field
assessments. Each writes machine-readable output to `results/metrics/`.

```bash
python -m pytest tests/ -q     # the full suite
```

## Scheme contact details

`configs/scheme_contacts.yaml`. Each entry carries a `verified` flag and a
source. **Only entries marked verified are shown without a warning.**

| item | status |
|---|---|
| Krishi Rakshak Helpline **14447** | checked against PIB, Ministry of Agriculture & Farmers Welfare, 08 Feb 2024 — the app prints that source so it can be re-checked; a tick in an app is not itself verification |
| PMFBY WhatsApp 7065514447 | unverified — pmfby.gov.in returns an empty body to both fetch and rendered load |
| Crop Insurance app | unverified |
| 72-hour intimation limit | unverified — consistent across secondary sources, not located verbatim on an official page |

Giving a farmer a wrong number is a real harm, so unverified entries print
`[UNVERIFIED — confirm first]` and the report tells the farmer to check with
their bank, Common Service Centre or District Agriculture Officer. Set
`verified: true` only after someone actually checks, and record the source.

## Layout

```
src/
  farmer_report.py     farmer-facing report (the safety rules live here)
  assess_field.py      analyst view with intervals and reasons
  make_splits.py       frozen plot-grouped split
  extract_features.py  satellite time series -> per-field features
  models.py            scikit-learn with a validated numpy fallback
  trees.py             numpy CART/forest/boosting
  run_ablation.py      V1-V8 sequential ablation
  paired_tests.py      plot-paired comparisons
  reliability.py       uncertainty and screening signals
  causal_history.py    history baseline with NO future harvests
  forward_eval.py      train on the past, predict the future
  season_swap.py       site-vs-season permutation test
  site_vs_season.py    two-stage decomposition
  confirmatory.py      the one frozen confirmatory run
  verification_policies.py  Experiments A and B: policies and reserves
  official_records.py  Experiment L: labels against the state's records
  inspection_sim.py    Experiment C: imperfect inspectors (simulated)
  yestech_whatif.py    Experiment D: YES-TECH what-if (simulated)
  inspection_workflow.py  step 4: plan, blind sheets, records, audit chain
  build_console.py     the verification console page
app.py                 Streamlit shell over farmer_report
configs/               frozen split and protocols, scheme contacts
data/official/         Tamil Nadu tables used by Experiment L, with sources
results/               metrics, predictions, figures, workflow, console
tests/                 the full suite (python -m pytest tests -q)
```

## Key documents

- `DISCOVERY.md` — the site-vs-season finding
- `THREE_VALIDATIONS.md` — leakage fix, measured-yield gap, fresh split
- `ACCEPTANCE_REPORT.md` — objective-by-objective status
- `DATA_AUDIT.md` — dataset, units, two corrections to working assumptions
- `INSURANCE_ASSUMPTIONS.md` — what cannot be simulated and why
- `SIMULATED_EXTENSIONS.md` — imperfect inspectors and the YES-TECH what-if
- `configs/extension_protocol.yaml` — the frozen protocol for C, D and L
- `PILOT_PROTOCOL.md` — a proposed one-season field pilot, not run, sized
  by `src/pilot_power.py`

## Limits you should state if you present this

- **131 plots, 319 field-seasons.** One delta, one cropping system,
  2019–2020.
- **Labels are farmer recall, not crop-cutting measurements.** This is the
  largest confound: part of the missing seasonal signal may be the label
  rather than the model, and the two cannot be separated without measured
  yields.
- **The forward-split test plots were seen in training**, in their earlier
  season. That evaluates a new season at known fields, not a new field. The
  plot-grouped confirmatory run, which did test unseen fields, **failed**
  both its primary criteria.
- **No Insurance Unit identity exists in this data**, so nothing here can
  reproduce a PMFBY decision.


## What the headline number does and does not mean

The "72 % of losses caught at a 33 % budget" figure is measured against a
label defined as **yield below 80 % of the training-split mean**, on
**farmer-recalled** yields. That is a relative cut against the area average,
not a measurement of crop damage.

`src/label_validity.py` decomposes it:

| plot type | plots | labelled loss seasons |
|---|---:|---:|
| chronic — below threshold *every* season | 12 | **32** |
| event — sometimes below, sometimes above | 20 | 21 |
| never below | 99 | 0 |

**60 % of labelled losses are at 12 chronically low-yielding plots.** Those
are a property of the land, not an insurable event, and any administrator
with a yield record already knows which fields they are.

Scoring the ranking by subset:

| evaluated on | n | losses | AUROC | 95 % CI |
|---|---:|---:|---:|---|
| everything (the headline) | 319 | 53 | 0.838 | [0.776, 0.889] |
| chronic + never plots | 272 | 32 | 0.854 | [0.809, 0.887] |
| **event plots — the real case** | 47 | 21 | **0.626** | **[0.386, 0.887]** |

On the only subset where the answer is not already in the field's record,
the interval includes chance. Recall at a 33 % budget there is 43 % against
33 % for random.

So, precisely:

- **It does mean** 72 % of field-seasons below 80 % of the area average are
  among the 33 % lowest-ranked.
- **It does not mean** 72 % of farmers suffering crop damage would be
  identified. That has not been shown, and on the subset that bears on it,
  it is not demonstrated.

The defensible description of this tool is **a ranking of fields by their
usual yield level**, which is useful for targeting attention and for
deciding where to sample, and is not a loss detector.

## Is the prototype actually better than the alternatives?

`src/evaluate_prototype.py` asks that, on the held-out test split only (27
plots never used to fit, choose or tune anything), against rankings a
verification officer could use instead. Passing tests prove the software
behaves as specified; they say nothing about whether using it leads to
better decisions.

| ranking, held-out test split | AUROC | 95 % CI | recall @ 33 % | beats arbitrary |
|---|---:|---|---:|---|
| SICKLE-Guard as shipped | 0.807 | [0.56, 1.00] | 0.73 | yes |
| **last harvest only** | 0.803 | [0.62, 0.94] | 0.64 | yes |
| history only, no imagery | 0.797 | [0.61, 0.94] | 0.64 | yes |
| satellite only | 0.734 | [0.42, 0.95] | 0.55 | no |
| largest field first | 0.765 | [0.66, 0.86] | 0.45 | no |
| arbitrary order | 0.494 | [0.24, 0.69] | 0.45 | no |

Out-of-fold over all 319 field-seasons, where there is more power, history
only and last-harvest-only both reach 0.838 against 0.824 for the shipped
hybrid.

Ranked side by side those look like a tie, but a tie is itself a claim, so
it was tested directly. Paired on the same resampled plots
(`src/audit_evaluation.py`, check 2):

| comparison | hybrid minus last harvest | 95 % CI | verdict |
|---|---:|---|---|
| held-out test split | +0.004 | [-0.092, +0.078] | inconclusive |
| all field-seasons, out-of-fold | -0.014 | [-0.059, +0.024] | inconclusive |

**Neither direction is demonstrated.** The honest statement is not that
looking up the last harvest is as good, and not that the hybrid is better;
it is that 27 held-out plots cannot tell them apart. What follows is the
weaker, defensible claim: *the extra machinery has not been shown to earn
its place*, which is consistent with `src/does_satellite_add.py` finding
the satellite component adds nothing measurable.

The two rankings also differ in two ways at once — mean of all prior
harvests against the single most recent, and satellite against area average
for fields with no record — so the comparison is fair but not clean. The
history-mean column is printed beside them for that reason, and it lands on
0.838 as well.

Two further results from the same file:

- **The flag carries information.** Flagged fields are 40 % below threshold
  against 7 % unflagged, difference +33 % (95 % CI +1 % to +62 %). And 3 of
  11 below-threshold fields are *not* flagged — which is exactly why there
  is no reassuring state.
- **The no-reliable-estimate warning is not calibrated.** It fires on
  missing records, not on error: warned fields average +41 kg/acre more
  error, interval −58 to +130. Over all field-seasons the difference runs
  the wrong way. The interface now says the warning means *no record*, not
  *worse estimate*.

What this does not establish is in the closing section of that script, and
the short version is that nobody has used this: no comprehension, no
workflow, no outcome for any person.

See `EVIDENCE_TO_DESIGN.md` for which research finding motivated each
design decision and which evaluation supports or refutes it, and
`THESIS_SCOPE.md` for what belongs in the thesis versus the paper.

## Nine reliability signals tested, none discriminated

The tool shows no confidence score. That is a measured decision, not an
omission. Every candidate signal for "is this particular estimate
unreliable" has been tested against yield error and none discriminates:

| signal | AUROC | 95 % CI |
|---|---:|---|
| prediction-interval width | 0.531 | [0.319, 0.717] |
| random-forest spread | 0.672 | [0.480, 0.842] |
| seed disagreement | 0.444 | [0.282, 0.615] |
| released-ensemble dispersion | 0.311 | [0.103, 0.634] |
| too few clear observations | 0.575 | [0.371, 0.756] |
| too few pixels | 0.559 | [0.375, 0.740] |
| out-of-distribution distance | 0.569 | [0.368, 0.768] |
| low predicted yield | 0.475 | [0.277, 0.650] |
| **water/vegetation decoupling at transplanting** | **0.512** | [0.276, 0.690] |

The last one deserves its own note, because it was the best idea anyone
had proposed and it was tested properly. A paddy field is flooded at
transplanting, so a plot that looks green while showing little standing
water might be green with weeds rather than establishing paddy. MNDWI was
computed from B3 and B11 over a window fixed in advance, 7 days before to
21 days after the recorded transplanting date; 316 of 319 field-seasons
were scorable. The protocol, `configs/mndwi_protocol.yaml`, was frozen
before any result existed.

All three pre-registered criteria failed. The interval includes chance, it
does not beat ranking by the plot's own past harvests (−0.285, 95 % CI
−0.473 to −0.158), and it does not beat either of its own components. The
water term **made the signal worse**: decoupling minus NDVI-alone is
−0.093, 95 % CI −0.142 to −0.040, over all scorable field-seasons.

Prediction intervals deserve a separate warning. They reach **90.3 %**
empirical coverage at a nominal 90 %, which looks like a working
safeguard, while their width correlates with realised error at r = +0.10.
Coverage is not discrimination. Conformal prediction would reproduce the
guarantee that already holds and would not supply what is missing.

**What these nine results do not say.** They do not show that uncertainty
quantification fails in general, or that no signal could work here. Each
interval above is wide, and a wide interval centred near 0.5 is a failure
to demonstrate discrimination, not a demonstration that discrimination is
absent. What they establish is narrower and still useful: on this dataset,
at this sample size, under a pre-specified protocol, these nine signals
did not identify unreliable yield estimates — and a system that displayed
any of them would be showing noise with the authority of a number.

See `ARCHITECTURE_ASSESSMENT.md` for how this bears on a proposed
production pipeline.

## Where the ranking fails

`src/error_breakdown.py` breaks the errors down by groups fixed before any
result was computed: whether the field has a record, how many prior
seasons it has, area tercile, season, and whether the plot is chronically
below threshold, sometimes below, or never. Yield-estimation error and
ranking error are reported separately, because they are different failures.

On the held-out test split the result is unusually clean:

| | n | below threshold | ranking recall | missed | estimation MAE |
|---|---:|---:|---:|---:|---:|
| has a record | 35 | 7 | **1.00** | **0** | 225 [136, 336] |
| cold start | 27 | 4 | **0.25** | **3** | 265 [179, 362] |

**Every ranking miss is a cold start.** Where the field has past harvests
the ranking caught all seven; where it does not, it caught one of four.
The failure is localised to exactly the fields where the system falls back
on the satellite model — the component `src/does_satellite_add.py` found
adds nothing measurable.

The two errors move in **opposite directions**, which is why they are not
pooled. Over all field-seasons, cold starts have the worse ranking recall
(0.69 against 0.92) and the *lower* estimation error (169 against 202
kg/acre). A single accuracy per group would have shown one and hidden the
other. On the held-out split the estimation intervals overlap, so no
estimation gap is demonstrated there in either direction.

One further observation, too small to be a finding: of the 3 misses, 2 are
at *event* plots — fields that are not usually poor. That is the costly
kind of miss, because nothing in the field's record would have prompted a
visit. 9 of 16 subgroups on the held-out split are below the size needed
to interpret, and the output marks each one.

## Does it make verification more effective than what a team already has?

`src/verification_policies.py`, under a protocol frozen before results
(`configs/verification_protocol.yaml`). Four policies on the same 27
held-out plots at three budgets, target as everywhere: below the study's
yield threshold, not verified damage.

| budget | random recall | register | history only | workflow |
|---|---|---:|---:|---:|
| 10 % | 0.10 | 3 of 11 | 4 of 11 | 4 of 11 |
| 20 % | 0.20 | 7 of 11 | 7 of 11 | 8 of 11 |
| 33 % | 0.32 | 7 of 11 | 7 of 11 | 8 of 11 |

**Prioritisation from records is demonstrated; the workflow's advantage
over the register is not.** Every record-based policy far outperforms
random inspection. The workflow finds at most one field-season more than
the register a verification office already keeps, and every
Bonferroni-adjusted interval includes zero. A team with a yield register
would gain nothing measurable from it on these plots.

**Fields with no record.** Every miss is a cold start, so a second
experiment reserved part of the budget for them. At a 33 % budget a 10 %
reserve keeps recorded recall at 97 % and lifts cold-start recall from 0
to 19 %. Within the cold-start pool the satellite estimate does **not**
beat a random order (+0.062, 95 % CI [-0.235, +0.364]): the gain comes
from inspecting some cold-start fields at all, not from the model choosing
which. The design consequence is to set aside a small, stated share of
inspections for fields with no record and not to present a satellite
estimate as evidence for choosing among them.

**Safety under failure.** `tests/test_safety_cases.py` renders reports for
the real held-out field-seasons the workflow missed, for fields with no
record, and for the same insufficient evidence at estimates from 300 to
3,500 kg/acre, and requires the wording to stay identical across those
estimates. A deliberately reassuring mutant was caught. This is a property
of the software, not evidence that anyone understands the report.

**Independent outcomes.** None found that fit. The CSISA repository's
crop-cut data covers rice in Nepal and maize in Odisha with no confirmed
repeat measurement of the same plot; Indian PMFBY damage photographs carry
no stated licence. Recorded as a limitation, not worked around.

## Step 4, verification, as software

`src/inspection_workflow.py` is what an inspection office would run once
the ranking exists: plan the visits, hand each inspector a task sheet,
record what they found, and keep an audit trail nobody can quietly edit.
Each rule traces to a measurement.

| rule | why |
|---|---|
| fields with a record are ranked by their own past harvests | Experiment A: the prioritisation value belongs to the records |
| a stated share is reserved for fields with no record, chosen **at random** with a logged seed | Experiment B: the satellite did not beat a random order among them |
| task sheets are **blind**: the inspector never sees the yield estimate | see below |
| an inspection records an observation, never a decision; claim and damage language is refused | decisions belong to the scheme |
| a finding of a problem needs a photo or a written note | no evidence, no finding |
| every action is a link in a SHA-256 chain; `verify` names the first broken link | tests edit records, edit the log, delete an entry and reorder the plan, and each is caught |
| simulated inspections say so in the record, the log and every summary | a demonstration must never pass for fieldwork |
| an uninspected field is **not inspected, and not cleared** | the farmer's own route to report damage stays open |

**Why the sheets are blind (observed).** At a 20 % budget the workflow
sends inspectors to 8 below-threshold field-seasons on the held-out plots,
and 4 of them carry an estimate above the study threshold. An inspector
shown that number could wave a real problem through. The 3 field-seasons
it does not reach are all cold starts, and all 3 carry satellite estimates
between 1,789 and 1,906 kg/acre against a threshold of 1,470: the
satellite model pulls a field with no record towards the average, so the
farms it misses are exactly the ones a "looks fine" display would have
reassured.

```bash
python src/inspection_workflow.py plan --budget 0.33 --reserve 0.10
python src/inspection_workflow.py sheet --task 1
python src/inspection_workflow.py record --task 1 --finding "no problem observed" --inspector "A. Officer"
python src/inspection_workflow.py verify
python src/inspection_workflow.py demo        # fresh database, one SIMULATED round
python src/build_console.py                   # results/console/verification_console.html
```

The console is a single self-contained page: the queue at each budget and
reserve (the planner's own output), the blind task sheet, the audit chain
re-hashed in the viewer's browser with a button that alters one record and
shows the chain break, and one panel per piece of evidence, each stamped
observed or simulated.

## The labels against Tamil Nadu's own records

Every result here rests on yields farmers recalled. `src/official_records.py`
checks that label against the state's crop-cutting statistics and relief
records (`data/official/`, provenance in `data/official/SOURCES.yaml`),
under `configs/extension_protocol.yaml`. **Observed**; the protocol
discloses that season means were seen before its verdict rules were
written.

**Level, 2019-20.** The state reports paddy "in terms of rice" and its own
conversion table puts cleaned rice at 2/3 of paddy. Converted that way,
recalled Kuruvai yields sit at 0.77 of the crop-cutting yield of
Thanjavur, Thiruvarur and Nagapattinam, below every one of them. Samba sits
at 0.90, inside the district range. Recalled labels run low, and markedly
so in Kuruvai.

**A documented damage season.** In 2020-21 the state allocated about
₹782 crore of agricultural relief across all four core delta districts for
cyclone Burevi and the unprecedented January 2021 rain, against about ₹1
crore for the 2019 monsoon. Every SICKLE Samba crop of 2020-21 was in the
field that January. At the same plots, recalled yields fell -3.9 %
[-7.5, -0.2], but no more than the Kuruvai control (difference -0.3 %), so
an event-specific fall is not demonstrated. **The satellite model's
estimates did not move: 0.0 % [-1.0, +1.0].** It reads the site, not the
season, which is the season-swap finding seen from the outside.

That is evidence about the model's responsiveness, not about these plots.
Relief is paid per district and SICKLE plots carry none, so neither check
can say whether any plot was damaged, or that the model missed damage on
one. Official district yields for 2020-21
were not found in a public table, so the damage season cannot yet be
compared with the state's own crop-cutting figures.

## Imperfect inspectors, and a YES-TECH what-if

Both are simulated and live on their own page,
[`SIMULATED_EXTENSIONS.md`](SIMULATED_EXTENSIONS.md), so their numbers are
never read beside observed ones. The first asks whether record-guided
inspection survives inspectors who miss problems or report false ones; the
second works out how an error in a technology yield estimate travels into
an Insurance Unit's claim at the weightages states use, without any
YES-TECH output, because none is public.

## Keeping the prose honest

`src/check_claims.py` binds every figure quoted in `README.md`,
`EVIDENCE_TO_DESIGN.md`, `ARCHITECTURE_ASSESSMENT.md`,
`THESIS_SCOPE.md` and `SAFEGUARD_FRAMEWORK_MAPPING.md` to the artefact it
came from, and to the exact string the document must contain. Every
claim is checked two ways: the value still matches the artefact, and the
document still says it.

The second check is the one that bites. If a rerun moves a number, the
rendered string changes and every document quoting it fails until it is
rewritten — so the prose cannot quietly fall behind the evidence.
Registry and amendments: `configs/claims.yaml`.

## Data and licence

The SICKLE data, trained weights and reference code belong to their authors
(Sani et al., 2024, WACV) and are distributed by them on request; none of it
is included here. To run this prototype, request SICKLE from its authors and
point `SICKLE_DATA_ROOT` at your copy.

The code in this repository is released under the MIT licence (see
`LICENSE`). The licence covers this code only, not SICKLE.
