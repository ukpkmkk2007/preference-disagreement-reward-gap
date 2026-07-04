# Preference Pair Quality Diagnostics

This repository contains a compute-constrained research project on preference disagreement and reward-gap diagnostics in human feedback datasets.

The project starts from a small-scale reproduction of a reward-model baseline on MultiPref and extends it into a scenario-conditioned failure analysis of when reward gap fails as a diagnostic signal for human preference disagreement.

## Motivation

Preference pairs are often treated as clean A/B comparison samples. However, human annotators may disagree for reasons such as response verbosity, formatting, safety behavior, factuality, instruction following, task ambiguity, or different preferences for tone and detail.

This project treats disagreement not simply as noise, but as a signal that may reveal ambiguity, multidimensional preferences, or limitations of single-value reward models.

## Research Questions

1. Can a single-value reward model recover the human majority preference on MultiPref-style preference pairs?
2. Can the absolute reward gap between two responses identify cases where human annotators disagree?
3. Does the reliability of reward-gap-based disagreement detection vary across task scenarios?
4. Which scenarios produce overconfident disagreement failures, where humans disagree but the reward model assigns a large score gap?

## Project Overview

### Part 1: Baseline Reproduction

I reproduce a small-scale version of the single-value reward model baseline evaluation pipeline from *Diverging Preferences: When do Annotators Disagree and do Models Know?*

The baseline evaluates:

- **Preference Accuracy**: whether the reward model prefers the same response as the human majority.
- **Diverging ID AUROC**: whether the model's reward gap can identify preference pairs where annotators disagree.

Due to local compute constraints, this is not a full reproduction of the original paper's large-model results.

### Part 2: Scenario-conditioned Failure Analysis

Based on the baseline results, I further analyze whether reward-gap reliability varies across task scenarios.

The pilot study focuses on overconfident disagreement cases, where human annotators disagree but the reward model still assigns a large absolute score gap between two responses.

## Data and Model

- Dataset: MultiPref
- Reward model: `Skywork-Reward-Llama-3.1-8B-v0.2`
- Inference: 4-bit quantized inference
- Training: no model training
- Evaluation scale: small sampled subsets under local GPU constraints
- Length filter: long examples are filtered for local feasibility

This repository does not include raw MultiPref prompts, raw responses, or model weights.

## Baseline Results

On a 100-example subset with clear human majority preferences:

| Metric | Value |
|---|---:|
| Preference Accuracy | 0.7500 |
| Majority-class baseline | 0.6600 |

For Diverging ID, I compare paper-like diverging examples against high-agreement examples using:

```text
diverging_score = -abs(score_a - score_b)
```

| Metric | Value |
|---|---:|
| Diverging ID AUROC | 0.6244 |
| Mean gap, diverging examples | 10.1316 |
| Mean gap, high-agreement examples | 14.6600 |

These results suggest that reward gap contains a weak-to-moderate signal for identifying human preference disagreement, but the signal is limited.

## Scenario-conditioned Failure Analysis

In the second part of the project, I manually label the 100-example pilot set by task scenario and analyze where reward gap becomes unreliable.

Scenario labels include:

- safety/refusal
- technical or expert-advice
- factuality/instruction-following
- format/style
- task ambiguity
- verbosity/concision
- other

Preliminary results suggest that reward-gap reliability is scenario-dependent. In this small pilot, overconfident disagreement failures are more concentrated in **verbosity/concision** and **technical/expert-advice** scenarios.

This result should be interpreted as exploratory rather than conclusive, because the current pilot uses only 100 examples and one reward model.

## Repository Structure

```text
docs/
  baseline_reproduction_methodology.md
  scenario_failure_methodology.md

reports/
  baseline_reproduction_report.md
  case_study_annotation_report.md
  scenario_conditioned_failure_analysis.md

results_summary/
  baseline_metrics.csv
  broad_label_counts_by_type.csv
  broad_label_counts_overall.csv
  case_study_label_counts.csv
  failure_case_summary.csv
  scenario_pilot_summary.csv
  strong_failure_cases_sanitized.csv

scripts/
  02_spyder_parse_multipref_labels.py
  03_spyder_define_diverging.py
  04_spyder_export_case_studies.py
  05_spyder_create_features.py
  06_prepare_skywork_baseline_inputs.py
  10_score_skywork_8b_random100.py
  11_skywork_8b_diverging_id_auroc.py
  12_make_baseline_reproduction_report.py
  13_make_case_study_annotation_report.py
  14_make_advisor_summary.py
  15_scenario_pilot_auto_label.py
  16_extract_overconfident_diverging_cases.py
  17_extract_strong_failure_cases.py
  18_make_scenario_pilot_report.py
```

## Reports

- [Baseline reproduction report](reports/baseline_reproduction_report.md)
- [Case study annotation report](reports/case_study_annotation_report.md)
- [Scenario-conditioned failure analysis](reports/scenario_conditioned_failure_analysis.md)

## Public Release Policy

To keep the repository lightweight and avoid redistributing raw dataset content, this repository only releases scripts, reports, aggregate summaries, and sanitized failure-case metadata.

Not included:

- raw MultiPref data;
- full prompt text;
- full response A or response B text;
- model weights;
- unsanitized case-study CSV files.

Released failure-case metadata may include fields such as:

```text
comparison_id
scenario_manual
failure_strength
score_gap_abs
model_pred
majority_label
n_A
n_B
n_Tie
balanced_disagreement
high_confidence_wrong
```

## Limitations

This is a small-scale exploratory project rather than a full benchmark reproduction or a new reward-model training method.

Main limitations:

1. The evaluation uses Skywork 8B rather than larger reward models.
2. Inference is done with 4-bit quantization due to local GPU constraints.
3. The current analysis uses small sampled subsets instead of full-scale MultiPref evaluation.
4. Scenario labels are manually defined and should be validated on larger samples.
5. Current results use one reward model and one dataset.

## Next Steps

Potential extensions include:

- expanding the sample size;
- validating scenario labels more systematically;
- comparing multiple reward models;
- testing whether the same failure patterns appear on other preference datasets;
- developing a more systematic taxonomy of reward-gap failure cases.
