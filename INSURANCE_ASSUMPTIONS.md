# Insurance simulation — assumptions, and what cannot be simulated

Read this before any number in `results/metrics/insurance_sim.json`. The
simulation is a sensitivity study of how yield-prediction error propagates
into a threshold decision. **It is not a payout calculation and it does not
reproduce the PMFBY procedure.**

## 1. What the scheme actually does

From the YES-TECH Manual 2023 (MNCFC/DAFW, Document Control Number
DAFW/MNCFC/PMFBY/YES-TECH/Manual_V1.0, Issue 01, published 11 July 2023 —
the only YES-TECH manual published on ncfc.gov.in) and the PMFBY Revised
Operational Guidelines (2020), verified in the companion audit study:

- Clause 3.3.10: "The State/UT shall assign at least 30% weightage to
  modelled yield and 70% weightage to CCE derived yield … The State/UT may
  decide to assign higher weightages to modelled yield". There is a floor of
  30 % and **no upper cap**.
- The stated objective is yield estimation "at the smallest administrative
  unit, i.e., Gram Panchayat (GP)". The decision is made for an **Insurance
  Unit**, not for an individual field.
- Indemnity under PMFBY is conventionally computed from a Threshold Yield,
  itself derived from historical yields and an indemnity level, against the
  Sum Insured for the notified area.

## 2. The four things that make a real simulation impossible here

| Required input | Status |
|---|---|
| Gram Panchayat / Insurance Unit identity per plot | **absent.** `RIVER_PART` is blank on all 319 yield rows and recoverable for none of the 131 plots (DATA_AUDIT.md §4.1) |
| Threshold Yield for the notified area | **absent.** Requires the historical yield series for the IU |
| Sum Insured, scale of finance, premium | **absent** |
| CCE yields to blend with, per the 30/70 rule | **absent** |

The first is fatal to fidelity. **The scheme decides at the Insurance Unit;
this data cannot form Insurance Units at all.** A plot-level simulation
therefore answers a different question from the one the scheme asks, and
every result must be read that way.

## 3. What is simulated instead, and why it is still worth doing

A deliberately simple, explicitly hypothetical rule:

> A field-season is **loss-flagged** when its yield falls below
> `indemnity_level × reference_benchmark`, where the benchmark is the
> **training-split mean yield** and the indemnity level is swept over
> 70 %, 80 % and 90 %.

The benchmark comes from training plots only, so no test information enters
it. This is **not** a Threshold Yield: a real TY is a historical moving
average for a notified area, not a cross-sectional mean of other farms.

The question this can answer is narrow but real:

> If a model-predicted yield replaced a reference yield as the input to a
> threshold decision, how often would the decision change, and does
> reliability-guided verification recover those cases more efficiently than
> random verification at the same budget?

That is a statement about **error propagation into a threshold decision**,
which is the mechanism the real scheme also contains, even though the
thresholds, units and amounts differ.

## 4. Outcome measures, and the unit they are in

- **Decision disagreement rate**: share of field-seasons where the
  prediction-based flag differs from the reference-based flag.
- **False negatives**: reference says loss, prediction does not. These are
  the cases that matter most to a farmer, so they are reported separately.
- **Residual decision error after verification**: verified cases are assumed
  resolved to the reference decision; the metric is the error remaining
  among the unverified.
- **Normalised shortfall difference**: |reference shortfall − predicted
  shortfall| expressed as a **fraction of the benchmark**, never in rupees.

**No monetary figure is produced anywhere.** Sum Insured is unavailable, and
inventing one would turn a sensitivity study into a fabricated payout claim.

## 5. What this does not establish

The simulation does not establish any actual payout, claim outcome, premium,
farmer income, change in compensation, or operational YES-TECH behaviour. It
does not show that any farmer would have been better or worse off. The
verification step assumes human verification recovers the reference value
exactly, which is itself an idealisation: a real crop-cutting experiment has
its own error, and the reference here is farmer recall rather than a
measured CCE.
