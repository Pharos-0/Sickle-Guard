# A proposed five-component architecture, checked component by component

A follow-on to `SAFEGUARD_FRAMEWORK_MAPPING.md`. The proposal specifies a
pipeline: a cloud gatekeeper, feature alignment, primary attention models
running beside a passive safeguard engine, a Kafka triage broker, and a map
UI that dispatches field agents.

Checked against what SICKLE actually contains and against this project's
measurements, the five components fall into three groups. **Two are
buildable and one of those is already half built. Two have already been
measured here and failed. One cannot be built from this dataset at all.**

---

## Component 1 — Cloud gatekeeper: **buildable, and the data supports it**

I expected to rule this out and could not. Every Sentinel-2 `.npz` in
SICKLE carries, alongside the reflectance bands:

```
SCL            scene classification layer
MSK_CLDPRB     per-pixel cloud probability
AOT, WVP       aerosol optical thickness, water vapour
```

The presence of `AOT` and `WVP` confirms these are **Level-2A
bottom-of-atmosphere** products, as the proposal assumes. An SCL-based mask
for thick cloud, cirrus and shadow, and a "more than 30 % of the plot mask
obscured" gate, are implementable exactly as described. Nothing in this
project has used `SCL` or `MSK_CLDPRB` yet — the feature extractor works
from reflectance alone.

The supporting evidence is the strongest single result in the ablation.
Adding cloud-penetrating radar is one of only three rungs whose paired
interval excludes zero:

> v4 to v5, adding Sentinel-1: **+0.1860** macro IoU, 95 % CI +0.1120 to
> +0.2600 — `sickle-project/results/ladder_power.json`

Cloud costs real accuracy here, and mitigating it is the one intervention
with a resolved effect.

**One caution: a gate is not a safeguard.** "Too few clear observations"
was tested as a reliability *signal* and does not identify unreliable
fields — `low_n_clear`, AUROC **0.575**, 95 % CI [0.371, 0.756]. Dropping
cloudy tiles before inference is sound engineering. Using cloudiness to
flag an output as untrustworthy is not supported.

---

## Component 2 — Feature extraction and alignment: **mostly built already**

`src/extract_features.py` already computes per-plot time-series statistics
for **NDVI, NDWI, NDMI and EVI** — mean, max, min, standard deviation and
area under the curve for each.

It is also resolution-aware in a way worth keeping: Sentinel-2 arrives on
three grids (10 m, 20 m, 60 m), and the extractor resamples the **mask**
onto each band's native grid rather than upsampling the imagery, so no
spectral value is invented to fill a finer grid.

Two genuine additions the proposal identifies:

- **MNDWI**, `(B3 − B11) / (B3 + B11)`, for standing water at
  transplanting, is **not** currently computed, and both bands are
  present. NDMI uses B8 against B11, which measures canopy moisture rather
  than open water, so this is a real gap for a flooded-paddy signal.
- **Whittaker smoothing** of the temporal profile is untested here. The
  current extractor summarises observations where they fall. Whether a
  smoothed, regularly spaced profile helps on a 0.19-acre median plot is
  open and testable.

---

## Component 3A — Attention networks: **measured, and the evidence is against**

The proposal names TSViT or UTAE. Both claims this project can speak to
point the other way:

- **UTAE is already a released SICKLE baseline, and its released crop-type
  checkpoints never trained.** `[L8]_utae` and `[S2]_utae` both score macro
  IoU 0.3812 with **paddy IoU exactly 0.0000** and non-paddy recall 1.000
  — they predict a single class everywhere, and are best at epoch 1.
- **In the controlled ablation, changing the architecture hurt.** The
  v6→v7 rung is one of only three resolved steps and its paired effect is
  **−0.0696** macro IoU, 95 % CI −0.1203 to −0.0189.

With 315 training plots, additional model capacity did not create
additional evidence, whereas an additional sensor did. That is the central
finding of the ablation and it argues for spending effort on Component 1,
not Component 3A.

---

## Component 3B — Passive safeguard engine: **already measured, and it failed**

This is the component to reconsider, because both halves have been built
and evaluated here.

### Conformal prediction guard

The proposal emits an interval and raises a flag when its width crosses a
threshold. Quantile intervals were fitted and **the guarantee already
works**: 90.3 % empirical coverage against a nominal 90 %.

**The flag does not.** Interval width correlates with realised error at
**r = +0.10**, and as a screening signal `quantile_width` scores AUROC
**0.531**, 95 % CI [0.319, 0.717].

This generalises beyond one implementation, and it is the single most
useful thing in this document. Conformal prediction guarantees *marginal
coverage* — that about 90 % of intervals contain the truth. It does not
guarantee that the **wide** intervals land on the **hard** fields.
Swapping a quantile wrapper for a conformal one would reproduce a
guarantee that already holds and would not supply the discrimination that
is missing.

A **locally adaptive or Mondrian** conformal scheme, which conditions the
interval on something observable, is a different and more interesting
proposal — but it is a research question, not a safeguard that can be
assumed to work.

### Historical profile anomaly detector

An Isolation Forest or autoencoder flagging deviation from a plot's own
historical curve is the same task as *did this season go wrong at this
field*, which was pre-specified and tested at three thresholds:

| definition | AUROC | 95 % CI |
|---|---:|---|
| deviation < −10 % of history | 0.598 | [0.497, 0.693] |
| deviation < −20 % of history | 0.590 | [0.451, 0.717] |
| deviation < −1 SD of deviation | 0.586 | [0.460, 0.706] |

Every interval spans chance. A distance-based out-of-distribution signal
was also tried — `ood_distance`, AUROC **0.569** [0.368, 0.768]. Seven
candidate signals were tested in total, and
`results/metrics/reliability.json` records the verdict as *no signal
demonstrated better than chance*.

**The one exception — now tested, and it failed.** The specific example
given, high NDVI during what should be a flooded transplanting phase
indicating weeds rather than crop, was the most promising concrete item in
the whole proposal. It has now been built and evaluated under a protocol
frozen before results (`configs/mndwi_protocol.yaml`,
`src/mndwi_screen.py`).

MNDWI was computed from B3 and B11 over the window from 7 days before to
21 days after the recorded transplanting date, standardised on the
training split, and the decoupling index taken as z(NDVI) − z(MNDWI).
316 of 319 field-seasons were scorable, so this is not a coverage problem.

| signal, held-out test split | AUROC | 95 % CI | informative |
|---|---:|---|---|
| **decoupling (pre-specified)** | **0.512** | [0.276, 0.690] | **no** |
| NDVI alone in the window | 0.597 | [0.266, 0.813] | no |
| MNDWI alone in the window | 0.428 | [0.254, 0.595] | no |
| the plot's own past harvests | 0.797 | [0.610, 0.936] | yes |

All three pre-registered criteria fail: the interval includes chance, it
does not beat the history-only ranking (−0.285, 95 % CI −0.473 to −0.158),
and it does not beat either component on its own.

Worse for the framing than a flat null: **adding the water term made it
worse.** Decoupling minus NDVI-alone is −0.086 on test and −0.093
(95 % CI −0.142 to −0.040) on all scorable field-seasons, where the
interval excludes zero. MNDWI alone sits *below* chance at 0.414. The
hypothesis was that water would sharpen a greenness signal; measured, it
blunted it.

One honest loose end, reported as an observation and not a finding:
window-mean NDVI alone does clear chance on the secondary set
(0.614, 95 % CI 0.520 to 0.699) while failing on the 27 held-out plots.
That is suggestive and underpowered, and under the protocol it changes
nothing.

---

## Component 4 — Kafka triage broker: **three blockers, one fatal**

The message schema is *Primary Model Output + Safeguard Metrics +
Cadastral Plot Ownership Data*, and the triage rule routes on uncertainty
crossed with whether the farmer is marginal or commercial.

**No cadastral or ownership data exists.** The released index carries
`PLOT_ID`, `RIVER_PART`, `STANDARD_SEASON`, `YEAR`, `AREA`, `CROP`,
`VARIETY`, `PADDY_BIN`, three phenology dates, `YIELD` and `SPLIT`. No
owner, no tenure, no holding size beyond the plot, no insurance status.

**The commercial category is empty.** Across 2,164 labelled rows the mean
plot is **0.351 acre**, median **0.190**, maximum **4.11**. 83 % are at or
below half an acre, 0.8 % reach two acres, none reaches five. A triage
matrix whose low-priority arm is "a large commercial block" has an arm with
nothing in it.

**Routing on uncertainty routes at random.** This is the fatal one. The
broker prioritises by safeguard output, and every safeguard signal measured
here sits at chance. Meanwhile the quantity that *does* rank
below-threshold fields — the plot's own past harvests, AUROC **0.838** —
appears nowhere in the architecture.

There is also no stream to broker. SICKLE is a fixed historical archive:
2,370 season-samples over 2018–2020. The infrastructure can be built, but a
thesis cannot evaluate a claim about throughput it never measures.

---

## Component 5 — Map UI and task optimizer: **half buildable**

The hashed-overlay treatment instead of a green *Healthy* badge is the
right instinct, and this prototype goes further: it has **no healthy state
at all**. The reason is measured — on the held-out test split **3 of 11**
below-threshold fields are not flagged, and a reassuring badge would have
reached those farmers.

But *"Prediction Unreliable – Awaiting Verification"* would be driven by
the safeguard engine, and the warning this prototype already shows on that
basis was tested and found **not calibrated**: warned field-seasons are not
reliably worse (+41 kg/acre, 95 % CI −58 to +130, and the sign reverses
over all 319). An overlay labelled *unreliable* that does not track
reliability teaches people to ignore it — which is the opposite of the
do-no-harm intent.

**The dispatch map cannot be built.** Checked rather than assumed: reading
the tags of `masks/10m/0.tif` returns **no GeoTIFF tags at all** — no CRS,
no model tiepoint, no pixel scale. The imagery ships as per-plot `.npz`
cubes already cut to the plot, so the geometry that would place a plot on
the ground was removed upstream. There are no coordinates to dispatch
anyone to, no road network and no agent locations.

---

## The architecture the evidence supports

```
[ S2 L2A cube ] -> [ SCL / MSK_CLDPRB gate ]   supported, not yet built
                          |
                          v
      [ per-plot indices: NDVI NDWI NDMI EVI, + MNDWI to add ]
                          |
        +-----------------+------------------+
        v                                    v
[ the plot's own past harvests ]      [ satellite model ]
  AUROC 0.838 -- the thing              adds nothing measurable
  that actually ranks                   over history alone
        |                                    |
        +-----------------+------------------+
                          v
            [ rank by expected yield level ]
                          |
                          v
   [ report: no healthy state, no confidence score,
     warning scoped to "no record", limits named inline ]
                          |
                          v
 [ human applies consequence weighting, separately and visibly ]
```

No confidence engine, because none of nine candidates discriminated. No
uncertainty-driven routing, because uncertainty does not track error here.
No automated vulnerability weighting, because the data carries no
vulnerability field and ranking by small size finds **fewer**
below-threshold fields, not more — AUROC **0.235**, 95 % CI [0.14, 0.34].

That is a smaller system than the one proposed, and it is the one the
measurements support. The defensible contribution is not a richer pipeline;
it is showing that a pipeline this rich **cannot be justified on this
evidence**, and what to build instead.

---

## Three testable things worth doing next

Ranked by the evidence that they would pay:

1. ~~**MNDWI at the transplanting window.**~~ **Done, and it failed.**
   Tested under a frozen protocol on 316 scorable field-seasons: AUROC
   **0.512**, 95 % CI [0.276, 0.690] on the held-out split, and the water
   term made the signal worse rather than better. It brings the
   reliability signals tested in this project to nine, none of which
   discriminates on
   yield. `src/mndwi_screen.py`, `results/metrics/mndwi_screen.json`.
2. **The SCL cloud gate.** Supported by the radar result, uses data
   already shipped in every `.npz`, and untouched so far. Note it is an
   inference-quality gate, not a reliability flag.
3. **Locally adaptive conformal intervals.** Only worth doing framed as a
   research question — does conditioning the interval on something
   observable buy discrimination that marginal coverage does not? The
   honest prior, after seven failed signals, is that it will not.
