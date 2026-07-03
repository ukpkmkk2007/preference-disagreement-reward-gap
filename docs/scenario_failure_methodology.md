# Methodology: Scenario-conditioned Reward-gap Failure Analysis

## Goal

This document describes the methodology used for analyzing when reward gap fails as a diagnostic signal for human preference disagreement.

The analysis is built on top of a small-scale MultiPref reward-model baseline reproduction.

## Data

The analysis uses a 100-example subset from MultiPref.

Each example contains:

- a prompt;
- two candidate responses;
- four human preference annotations;
- reward scores assigned by a reward model.

The public GitHub repository does not include raw MultiPref prompts or responses. It only includes aggregate summaries and sanitized case-level metadata.

## Human Disagreement Labels

The main positive label is:

```text
diverging_paper_like
```

This approximates the paper-style definition of human preference disagreement.

The main negative label is:

```text
high_agreement_pref
```

This marks examples where human annotators show relatively high agreement.

## Reward Scores

For each pair, the reward model assigns one score to response A and one score to response B:

```text
score_a
score_b
```

The absolute reward gap is computed as:

```text
score_gap_abs = abs(score_a - score_b)
```

## Reward-gap Diagnostic

The core assumption is:

```text
smaller reward gap -> higher model uncertainty -> more likely human disagreement
larger reward gap -> higher model confidence -> more likely human agreement
```

Therefore, the disagreement-detection score is:

```text
-score_gap_abs
```

A higher value of `-score_gap_abs` means a smaller absolute reward gap and therefore a stronger prediction of human disagreement.

## Scenario Annotation

Each example is manually assigned to one scenario label:

- `safety_refusal`
- `technical/expert-advice`
- `factuality_instruction`
- `format_style`
- `task_ambiguity`
- `verbosity_concision`
- `other`

These scenario labels are intended for exploratory analysis rather than final taxonomy design.

## Scenario-level Analysis

For each scenario, the following statistics are computed:

```text
n
n_diverging
n_high_agreement
mean_gap
mean_gap_diverging
mean_gap_high_agreement
overconfident_count
AUROC
overconfident_rate_among_diverging
```

AUROC is computed within each scenario when both positive and negative classes are present.

## Overconfident-diverging Cases

An overconfident-diverging case is defined as an example where:

```text
diverging_paper_like == True
```

and

```text
score_gap_abs is in the top quartile of the full sample
```

This means that human annotators disagree, but the reward model still assigns a large gap between the two responses.

## Strong Failure Cases

Strong failure cases are overconfident-diverging cases that satisfy at least one of the following conditions.

### 1. Balanced Disagreement

Human preferences are close to evenly split, such as:

```text
2 vs 2
2 vs 1 plus 1 tie
1 vs 2 plus 1 tie
```

In these cases, a large reward gap may indicate model overconfidence even if the model agrees with the plurality or majority label.

### 2. High-confidence Wrong Prediction

The reward model strongly prefers the response that is not preferred by the human majority.

This indicates both high confidence and wrong directional prediction.

## Weak Failure Cases

Weak failure cases are overconfident-diverging cases where human disagreement exists but the human majority is still relatively clear.

For example:

```text
n_A = 3, n_B = 1, n_Tie = 0
```

If the model agrees with the majority and assigns a large reward gap, this is still a disagreement-related failure case, but the evidence is weaker.

## Public Release Policy

The GitHub repository only includes sanitized metadata and aggregate summaries.

The following files should not be released publicly:

- raw MultiPref data;
- full prompt text;
- full response A or response B text;
- model weights;
- unsanitized case-study CSV files.

Released failure-case metadata may include:

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

## Reproducibility Notes

The public repository is intended to show the research pipeline, not to mirror the full dataset.

A reader should be able to understand:

1. how disagreement labels are constructed;
2. how reward gap is computed;
3. how scenario labels are used;
4. how overconfident-diverging cases are defined;
5. which aggregate patterns appear in the pilot analysis.

Full raw examples can be reconstructed by authorized users from the original dataset using `comparison_id`, subject to the dataset license and attribution requirements.

## References

- MultiPref dataset: https://huggingface.co/datasets/allenai/multipref
- Diverging Preferences: When do Annotators Disagree and do Models Know?: https://arxiv.org/abs/2410.14632
- Skywork Reward Model: https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2
