# Under which conditions does triage help a verification team?

Synthetic data cannot show that a farmer is better off. Nobody is in it.
What it can do is vary what SICKLE holds fixed — how many fields have a
record, how much variation is seasonal rather than site-driven, how noisy
farmer recall is, how prevalent genuine events are, whether bad seasons
land on already-poor land, and how large the inspection budget is — and
map where triage helps and where it does not.

`src/synthetic_triage.py`. **Every number on this page is simulated.** It
is a map of conditions, not observed field performance, and it must never
be reported beside the empirical results as though it were one.

## The generator earns its standing first

A generator can be made to say anything, so before any sweep runs it is
checked against four findings from the real data, and **the run aborts if
it misses**. It did abort on the first attempt; the parameters were then
calibrated by grid search until it did not.

| quantity | measured | accepted range | simulated | |
|---|---:|---|---:|---|
| history-only ranking AUROC | 0.838 | [0.788, 0.888] | 0.811 | ok |
| share of losses at chronic plots | 0.600 | [0.500, 0.700] | 0.578 | ok |
| event-plot ranking AUROC | 0.626 | [0.386, 0.887] | 0.505 | ok |
| share of field-seasons with a record | 0.590 | [0.530, 0.650] | 0.585 | ok |

Where a real measurement has a published interval, **the interval is the
target**. Demanding 0.626 from a simulator when that estimate's own 95 %
CI spans [0.386, 0.887] would ask for more precision than the measurement
has. The simulator lands at 0.505, which *agrees* with what was measured:
the finding was never that history predicts which season went wrong, but
that the interval includes chance.

This is a calibration check, not a validation of structure. A different
generator could reproduce the same four quantities and behave differently
in the sweeps.

## The competitor that matters

Not random referral. A verification team already holds a register of which
plots yielded badly before, and 60 % of labelled losses sit at plots below
threshold in every season. So the bar is: **does triage beat visiting the
plots you already know about?**

Scored on recall over **event losses** — a bad season at a field that is
not usually bad. Those are the only ones a prediction can add anything to.

## The result

Across **31 settings in seven sweeps**, with a policy required to beat the
runner-up by 0.02 recall before it counts as better:

| outcome | settings |
|---|---:|
| **tie** | **20** |
| register only wins | 8 |
| prototype wins | 3 |
| history-first wins | 0 |

**Most differences are inside the noise.** That is the dominant finding
and it should be stated before any of the rest.

### At the calibrated operating point, the register wins

With record availability at the measured 0.59, the register recalls
**0.73** of event losses against **0.64** for the prototype — a gap of
0.09, well clear of the 0.02 margin. This is the single comparison the
whole study exists to make, and it goes against the prototype.

### Where the register wins outright

At 0.2, 0.4 and 0.59 record availability; at the real season variance; at
moderate recall error; at a 20 % budget; and when the satellite carries no
information. These are the realistic settings.

### Where the prototype wins outright

Three settings, all of them away from what was measured: when **every**
field has a record, and when the satellite is **twice or three times** as
informative as the measured value of about 0.25.

### The budget sweep, and the design change it motivated

Recall on event losses:

| budget | history-first | prototype | history only | register |
|---:|---:|---:|---:|---:|
| 0.10 | **0.25** | 0.12 | **0.25** | 0.21 |
| 0.20 | 0.55 | 0.45 | 0.55 | **0.59** |
| 0.33 | 0.60 | 0.72 | 0.63 | 0.73 |
| 0.50 | 0.60 | 0.84 | 0.79 | 0.84 |

At a 10 % budget the hybrid recalls **0.12** against **0.25** for history
alone — it is beaten by its own component. The mechanism is in the design:
cold-start fields are ranked by a weak satellite estimate, and when only a
few visits are available those weak picks displace strong ones.

`history-first` — rank the fields that have a record, and place cold
starts behind all of them rather than interleaving guesses — repairs
exactly that, reaching 0.25 at a 10 % budget. But it **ties with plain
history-only** there and never wins outright anywhere. The honest reading
is that history-first fixes a self-inflicted problem rather than adding
anything: at a tight budget, the right move is not to rank cold-start
fields at all.

### Where the shocks land

The new axis, and the one that explains the register's advantage. Bad
seasons were made to fall preferentially on already-poor plots (+1),
independently (0), or on good plots (−1):

| shock_site_bias | prototype | register |
|---:|---:|---:|
| −1.0 (shocks hit good plots) | 0.57 | 0.57 |
| 0.0 (independent) | 0.58 | 0.59 |
| +1.0 (shocks hit poor plots) | 0.59 | 0.61 |

Every setting is a tie at the 0.02 margin. The register's edge grows as
events concentrate on land that is already poor — which is the regime the
real data sits in — and shrinks when they do not. But the prototype does
not gain when the register loses: **both degrade together**. Decoupling
chronic from event losses makes the problem harder for everyone rather
than handing it to the model.

## The conclusion, with the evidence boundary explicit

> Under the evaluated conditions, the prototype is weakest where
> historical records are unavailable — a setting in which its
> satellite-based estimates have not demonstrated sufficient added value
> to reliably prioritize verification.

That is deliberately narrower than "the system is weakest for all farmers
without records". It joins the empirical baseline comparison to the
simulated sweeps without treating the simulation as field validation.

## What this is not

- **Not farmer benefit.** No person is in this simulation.
- **Not a validated model of the real process.** Four measured quantities
  are reproduced; the structure is not thereby confirmed.
- **Not a reason to drop the prototype.** The register needs a maintained
  per-plot yield history. The prototype's case is for fields where none
  exists — which is where the measurements show it performs worst, and
  which is the honest statement of the problem rather than a reason to
  stop.
