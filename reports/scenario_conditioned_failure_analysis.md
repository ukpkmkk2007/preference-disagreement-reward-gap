# Scenario-conditioned Failure Analysis of Reward-gap Diagnostics

## Overview

This report presents a small-scale exploratory analysis of when reward gap may fail as a diagnostic signal for human preference disagreement.

The analysis extends a baseline reproduction on MultiPref using a single-value reward model. After observing that reward gap provides only a weak-to-moderate signal for identifying diverging human preferences, I further analyze whether the reliability of reward gap differs across task scenarios.

This project should be interpreted as a small-scale reproduction plus exploratory failure analysis, not as a full benchmark or a new reward-model training method.

## Research Question

When does reward gap fail as a diagnostic signal for human preference disagreement across different task scenarios?

The main focus is on **overconfident disagreement cases**:

> Human annotators disagree, but the reward model still assigns a large absolute reward gap between the two responses.

## Experimental Setup

- Dataset: MultiPref
- Reward model: Skywork-Reward-Llama-3.1-8B-v0.2
- Inference setting: 4-bit inference under local GPU constraints
- Sample size: 100 preference pairs
- Positive class: paper-like diverging preference examples
- Negative class: high-agreement preference examples

For each pair, the model assigns a scalar reward score to response A and response B.

The absolute reward gap is computed as:

```text
score_gap_abs = abs(score_a - score_b)
```

The disagreement-detection score is:

```text
-score_gap_abs
```

The intuition is that a smaller reward gap should indicate higher model uncertainty and therefore a higher chance of human preference disagreement.

## Baseline Result

On the 100-example subset, the reward-gap diagnostic obtains:

```text
Diverging ID AUROC = 0.6244
mean_gap_diverging = 10.131574
mean_gap_high_agreement = 14.659980
median_gap_diverging = 7.362305
median_gap_high_agreement = 11.988525
```

This suggests that reward gap contains a weak-to-moderate signal for identifying diverging human preferences. However, the signal is not strong enough to reliably separate all diverging and high-agreement cases.

This motivates the scenario-conditioned analysis below.

## Scenario Categories

Each example is manually assigned to one task scenario:

- `safety_refusal`
- `technical/expert-advice`
- `factuality_instruction`
- `format_style`
- `task_ambiguity`
- `verbosity_concision`
- `other`

The original working label `technical_medical` is interpreted more broadly as `technical/expert-advice`, since the category includes technical, mathematical, engineering, medical, and other expert-advice examples.

## Scenario-level Results

| Scenario | n | n_diverging | n_high_agreement | mean_gap_diverging | mean_gap_high_agreement | AUROC | overconfident_rate_among_diverging |
|---|---:|---:|---:|---:|---:|---:|---:|
| safety_refusal | 21 | 9 | 12 | 9.4142 | 13.1554 | 0.6296 | 0.2222 |
| technical/expert-advice | 20 | 8 | 12 | 18.0391 | 16.8893 | 0.4479 | 0.5000 |
| factuality_instruction | 17 | 13 | 4 | 9.1684 | 6.4473 | 0.3269 | 0.0769 |
| format_style | 16 | 9 | 7 | 4.6035 | 12.9575 | 0.8095 | 0.0000 |
| task_ambiguity | 13 | 6 | 7 | 5.1507 | 11.0200 | 0.8333 | 0.0000 |
| verbosity_concision | 11 | 5 | 6 | 17.2027 | 26.8613 | 0.7000 | 0.8000 |

## Key Observations

Reward gap appears more reliable in `task_ambiguity` and `format_style` examples. In these categories, diverging examples tend to have smaller reward gaps than high-agreement examples, and AUROC is relatively high.

In contrast, `technical/expert-advice` shows a weaker pattern. The AUROC is below 0.5, and half of the diverging examples are overconfident disagreement cases. This suggests that reward gap may be less reliable in expert-advice tasks where surface-level answer quality and domain-specific correctness are harder to judge with a single scalar score.

The `verbosity_concision` category is also notable. Although its AUROC is not low, it contains a high concentration of overconfident disagreement cases. This suggests that reward gap may preserve some average signal while still failing strongly on individual examples where humans disagree over preferred level of detail.

## Failure Case Definition

An **overconfident-diverging case** is defined as an example where:

```text
diverging_paper_like == True
```

and

```text
score_gap_abs is in the top 25% of the full sample
```

This captures cases where human annotators disagree, but the reward model gives a large reward difference between the two candidate responses.

## Strong and Weak Failures

### Strong Failure

Strong failures include at least one of the following:

1. **Balanced disagreement**  
   Human preferences are close to evenly split, such as:
   - 2 vs 2
   - 2 vs 1 plus 1 tie
   - 1 vs 2 plus 1 tie

2. **High-confidence wrong prediction**  
   The reward model strongly prefers the response that is not preferred by the human majority.

These cases are stronger evidence of reward-model overconfidence under human disagreement.

### Weak Failure

Weak failures are overconfident-diverging cases where human disagreement exists, but the human majority is still relatively clear.

For example:

```text
n_A = 3, n_B = 1, n_Tie = 0
```

If the model agrees with the human majority and assigns a large gap, this is still a disagreement-related failure case, but the evidence is weaker than in balanced-disagreement or high-confidence-wrong cases.

## Failure Case Summary

| Scenario | Failure strength | Count |
|---|---|---:|
| factuality_instruction | strong_failure | 1 |
| safety_refusal | weak_failure | 2 |
| technical/expert-advice | strong_failure | 1 |
| technical/expert-advice | weak_failure | 3 |
| verbosity_concision | strong_failure | 3 |
| verbosity_concision | weak_failure | 1 |

The strongest concentration of strong failures appears in `verbosity_concision`, followed by `technical/expert-advice`.

## Interpretation

These preliminary results suggest that reward-gap reliability is scenario-dependent.

Reward gap may be useful as a weak diagnostic signal for human preference disagreement, but it should not be treated as uniformly reliable across task types. In particular, `verbosity_concision` and `technical/expert-advice` examples appear more likely to produce overconfident disagreement failures.

A cautious takeaway is:

> Reward gap can help identify some human preference disagreements, but its reliability depends on the task scenario. It appears more useful in task ambiguity and format/style cases, while overconfident failures are more concentrated in verbosity/concision and technical/expert-advice cases.

## Limitations

This is a small-scale pilot based on 100 examples. The results should be interpreted as exploratory rather than conclusive.

Important limitations include:

- small sample size;
- only one reward model;
- manually defined scenario labels;
- no cross-dataset validation;
- local inference constraints requiring 4-bit inference and length filtering;
- no claim that the observed scenario pattern generalizes to all reward models or preference datasets.

## Next Steps

Future work should:

1. expand the sample size;
2. validate the scenario taxonomy;
3. compare multiple reward models;
4. test whether similar failure patterns appear in other preference datasets;
5. refine the failure taxonomy using more systematic qualitative annotation.

## Public Release Policy

This repository releases aggregate summaries and sanitized failure-case metadata only.

It does not release:

- raw MultiPref data;
- full prompt text;
- full response A or response B text;
- model weights;
- unsanitized case-study CSV files.

The sanitized failure-case metadata may include:

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

## References

- MultiPref dataset: https://huggingface.co/datasets/allenai/multipref
- Diverging Preferences: When do Annotators Disagree and do Models Know?: https://arxiv.org/abs/2410.14632
- Skywork Reward Model: https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2
