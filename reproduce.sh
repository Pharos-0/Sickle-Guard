#!/usr/bin/env bash
# Full pipeline, in dependency order. Uses the sickle-project venv.
set -e
# the shared interpreter lives in the sibling sickle-project folder
PY="${SICKLE_PY:-$(cd "$(dirname "$0")/../sickle-project" && pwd)/.venv/Scripts/python.exe}"
"$PY" src/make_splits.py
"$PY" src/extract_features.py
"$PY" -m pytest tests/ -q
"$PY" src/run_ablation.py
"$PY" src/paired_tests.py
"$PY" src/reliability.py
"$PY" src/insurance_sim.py
"$PY" src/confirmatory.py
"$PY" src/grouped_cv.py
"$PY" src/season_swap.py
"$PY" src/assess_field.py --split test --csv results/predictions/field_assessments.csv
"$PY" src/site_vs_season.py
"$PY" src/causal_history.py
"$PY" src/forward_eval.py
"$PY" src/farmer_report.py --village-list --budget 0.33
"$PY" src/does_satellite_add.py
"$PY" src/label_validity.py

# Evaluates the prototype as a DECISION PROCEDURE rather than as software,
# so it must follow grouped_cv.py, whose out-of-fold predictions are the
# only honest input for the comparison over all field-seasons.
"$PY" src/mndwi_screen.py
"$PY" src/evaluate_prototype.py

# The five verification checks on the evaluation itself:
# split integrity, the baseline comparison, the interval
# method, the warning wording and the matrix verdicts.
"$PY" src/error_breakdown.py

# Pre-registered: the workflow against the register a team
# already keeps, and the cold-start allocation trade-off.
"$PY" src/verification_policies.py

# Maps where triage helps a verification team and where it
# does not. Aborts unless the generator first reproduces four
# measured quantities, so a sweep cannot be reported from a
# simulator that disagrees with the real data.
"$PY" src/synthetic_triage.py
"$PY" src/audit_evaluation.py

# configs/extension_protocol.yaml, frozen before any of these ran.
# L is OBSERVED: SICKLE's recalled yields against Tamil Nadu's own
# crop-cutting and relief records (data/official/). C and D are
# SIMULATED: imperfect inspectors on the observed rankings, and the
# YES-TECH what-if. C must follow verification_policies.py.
"$PY" src/official_records.py
"$PY" src/inspection_sim.py
"$PY" src/yestech_whatif.py
# Planning only: sizes the proposed pilot (PILOT_PROTOCOL.md) from the
# effects observed above.
"$PY" src/pilot_power.py

# Step 4 as software: one fresh plan and a SIMULATED inspection round
# on the held-out split, then the console page built from the artefacts.
"$PY" src/inspection_workflow.py demo
"$PY" src/build_console.py

# Do the documents still agree with the artefacts? Binds every
# quoted figure to its source and to the exact string the prose
# must contain, so a rerun cannot leave stale writing behind.
"$PY" src/check_claims.py
echo "pipeline complete"
