# Pilot protocol: one inspection office, one season

**Status: PROPOSED. Not run, and not part of the thesis evidence.** It is
written down before any pilot data exists so that the questions, the
comparison, the ground truth and the verdict rules cannot be chosen after
the results. Running it needs a partner office, an ethics review and the
consent of every farmer whose field is measured.

Everything this project has shown rests on farmer-recalled yield and on 27
held-out plots. A pilot is the step that replaces both with measured
yields from an office's real fields.

## 1. Questions, in order of priority

1. **Records against current practice.** Does ranking fields by their own
   recorded harvests, with a reserved share for fields that have no record,
   find more fields with a measured shortfall per inspection than the way
   the office chooses fields today?
2. **How accurate are inspectors?** How often does a visit confirm a
   measured shortfall, and how often does it report one that is not there?
   Experiment C assumed these numbers; the pilot measures them.
3. **Does a blind task sheet change what inspectors record?** Half the
   inspected fields get a sheet that shows the model estimate, half a sheet
   that withholds it, assigned at random.

The pilot is deliberately **not** designed to test whether the satellite
adds anything over the register. Section 6 shows why.

## 2. Setting and safety

- One district in the Cauvery Delta, one Samba season.
- **Shadow mode.** Nothing the pilot records feeds any claim, relief or
  eligibility decision, and no farmer is told anything about their crop.
- Farmers' own routes to report damage (the 14447 helpline, the crop
  insurance app, the agriculture office) are untouched and are named on
  every consent form.
- Fields are identified by a pseudonymous code in every analysis file.
- Stop rule: if the pilot delays or interferes with any scheme process,
  it stops and the reason is reported.

## 3. Policies compared

Fixed in writing before the season, each at the same budget of 20 % of
enrolled fields:

| policy | how it chooses |
|---|---|
| current practice | the office's own selection, described by the office before the season |
| register | lowest recorded past harvest first, with 10 % of the budget reserved for fields with no record and drawn at random under a logged seed |
| workflow | the register policy with the satellite estimate ordering fields that have no record |

**Union design.** Each policy proposes its list; every field on any list is
inspected once. Inspectors do not know which policy proposed a field. Each
policy is then scored on the fields it proposed, against the same ground
truth, so every comparison is paired.

## 4. Ground truth

- A separate team, blind to every ranking and every model estimate, runs
  the state's standard crop-cutting procedure on **every inspected field
  and on a random audit sample of 10 % of the fields no policy chose**.
- The audit sample is what makes misses measurable: without it, a field
  no policy chose is never measured, and recall cannot be estimated.
- **Primary label:** measured yield below 80 % of the mean measured yield
  of the random audit sample. The reference comes from fields no policy
  selected, so no policy can move it.
- Where the office can supply the Insurance Unit's threshold yield before
  the season, a secondary label uses it instead.

## 5. Measures

| measure | definition |
|---|---|
| recall per policy (primary) | below-label fields a policy proposed, over the estimated number of below-label fields among all enrolled fields (inverse-probability weighted from the union and the audit sample) |
| precision per policy | below-label fields among those the policy proposed |
| inspector sensitivity and false-alarm rate | the inspector's recorded finding against the crop-cut label |
| effect of the blind sheet | the same two rates, blind sheet against shown estimate |
| cost | staff minutes per visit including travel, recorded by the workflow |
| integrity | the audit chain verified at the end of the season |

A false alarm costs a visit; a miss costs a farmer an unexamined loss.
The two are reported separately and are combined into one number only if
the office states its own cost ratio before the season.

## 6. How large

`src/pilot_power.py` sizes each paired comparison from the effects
observed here, for 80 % power at a two-sided 5 % level (McNemar, normal
approximation). Over all field-seasons, out-of-fold:

- **register against random selection:** about 118 fields to enrol;
- **workflow against the register:** the observed difference is +0.019
  in recall, which needs 2,078 below-threshold fields, about 12,505 fields
  to enrol. One district cannot answer that, and the pilot does not try.
  (The held-out split's single discordant field suggests 84
  below-threshold fields, an optimistic figure resting on one field.)

How large "records against current practice" must be depends on how good
current practice already is, which is the thing the pilot measures. If it
is no better than random selection, the same order as above suffices. If the
office already chooses by its records, the difference may be too small to
detect in one district, and that is itself a useful answer: the office
already does what the evidence supports. The proposal is therefore to
enrol about 500 fields, enough for questions 1 and 2 under the first case
and to estimate inspector accuracy to within about a tenth.

## 7. Analysis and verdict rules

- Field-level bootstrap, resampling villages, for every interval.
- Three pairwise policy comparisons, Bonferroni-adjusted.
- A policy is said to **beat** another only if the adjusted interval of
  the paired recall difference excludes zero and it is not significantly
  worse on precision. Otherwise the result is reported as not demonstrated,
  at the same prominence.
- Inspector accuracy is reported with its interval and replaces the
  assumed values in Experiment C, which is then rerun and reported beside
  the original.

## 8. The YES-TECH question needs different data

The pilot cannot test a guard placed after YES-TECH. That needs, for the
same crop, season and Insurance Units: the YES-TECH yield, the crop-cutting
yields with plot locations, the final blended actual yield, the threshold
yield and the weightage applied. With those, two analyses become possible:
how often the technology and crop-cutting yields put a unit on opposite
sides of its threshold, and whether re-verifying units just above the
threshold (Experiment D) would have recovered the units the crop-cutting
yields say were short. Such data is not public and would need a
data-sharing agreement.
