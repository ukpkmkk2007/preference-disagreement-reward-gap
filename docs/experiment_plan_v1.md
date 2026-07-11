# Experiment Plan v1

## 1. Research Question

**Does the diagnostic validity of scalar reward-model reward gaps for human preference disagreement vary systematically across different task scenarios and disagreement sources?**

中文表述：

> 单值奖励模型的 reward gap 对人类偏好分歧的诊断效度，是否会随任务场景和分歧来源发生系统性变化？

---

## 2. Motivation

A scalar reward model assigns one reward score to each response. For two responses A and B under the same prompt:

```text
reward_gap = abs(reward_A - reward_B)
```

A smaller reward gap is sometimes treated as a signal that the reward model finds the two responses difficult to distinguish and may therefore indicate human preference disagreement.

However, reward gap is only a preference-level diagnostic proxy. It is not a calibrated estimate of epistemic uncertainty.

The preliminary reproduction obtained an overall Diverging ID AUROC of `0.6244`, suggesting that reward gap contains some disagreement signal, but the signal is limited. This experiment investigates whether the weak overall result hides systematic differences across task scenarios and disagreement sources.

---

## 3. Core Hypotheses

### H1: Scenario heterogeneity

The ability of reward gap to distinguish human-disagreement pairs from high-agreement pairs differs across task scenarios.

### H2: Overconfident disagreement

Some task scenarios or disagreement sources have a higher rate of cases in which human annotators disagree but the reward model assigns a large reward gap.

### H3: Cross-model consistency

If the observed scenario differences represent systematic reward-model behavior rather than noise from one model, at least part of the pattern should appear across multiple scalar reward models.

---

## 4. Experimental Unit

Each experimental unit is one preference comparison containing:

- one prompt;
- response A;
- response B;
- multiple human preference annotations;
- human disagreement information;
- reward score for response A;
- reward score for response B;
- one primary task-scenario label;
- zero or more disagreement-source labels.

---

## 5. Human Preference Labels

The experiment reuses the definitions already implemented in the baseline reproduction code. These definitions are fixed before inspecting the new experiment results.

### 5.1 Supporting quantities

For each preference pair:

- `n_A`: number of annotators preferring response A;
- `n_B`: number of annotators preferring response B;
- `n_Tie`: number of annotators selecting Tie;
- `n_clear`: number of clear A/B preferences;
- `n_slight`: number of slight A/B preferences;
- `n_valid`: number of valid annotations.

The preprocessing code defines:

```text
tie_only =
    (n_A == 0)
    and (n_B == 0)
    and (n_Tie > 0)
```

```text
only_slight_or_tie =
    (n_clear == 0)
    and (n_slight > 0)
```

### 5.2 Diverging pair

The primary positive class is `diverging_paper_like`.

```text
diverging_paper_like =
    (n_A > 0)
    and (n_B > 0)
    and not tie_only
    and not only_slight_or_tie
```

Operational interpretation:

- at least one annotator prefers A;
- at least one annotator prefers B;
- the pair is not tie-only;
- the non-tie preferences are not composed exclusively of slight preferences.

This definition does not require clear preferences on both sides. The stricter field `diverging_substantial` exists but is not the primary positive label.

### 5.3 High-agreement pair

The primary negative class is `high_agreement_pref`.

```text
high_agreement_A =
    (n_A > 0)
    and (n_B == 0)
    and (n_Tie == 0)
```

```text
high_agreement_B =
    (n_B > 0)
    and (n_A == 0)
    and (n_Tie == 0)
```

```text
high_agreement_pref =
    high_agreement_A or high_agreement_B
```

Operational interpretation:

- all non-missing annotators choose the same response;
- there is no opposing A/B vote;
- there is no Tie vote.

### 5.4 Majority label

The three uploaded scripts consume an existing `majority_label` column but do not show the earlier line where it is created. Based on the current outputs and case records, the working definition is:

```text
majority_label = "A"          if n_A > n_B
majority_label = "B"          if n_B > n_A
majority_label = "Ambiguous"  if n_A == n_B
```

Tie votes do not resolve an A/B equality. This rule should be checked once against the preprocessing script that originally creates `majority_label`; if that script differs, the plan must be updated before formal sampling.

Examples:

- `n_A=3, n_B=1, n_Tie=0` gives A;
- `n_A=1, n_B=2, n_Tie=1` gives B;
- `n_A=2, n_B=2, n_Tie=0` gives Ambiguous.

### 5.5 Primary analysis subset

The primary binary Diverging ID analysis includes only pairs satisfying one of:

```text
diverging_paper_like == True
```

or

```text
high_agreement_pref == True
```

Pairs satisfying neither definition are excluded. Invalid rows with no usable annotations are also excluded.

---

## 6. Reward-Gap Signal

For reward model `m` and preference pair `i`:

```text
reward_gap_i,m = abs(reward_A_i,m - reward_B_i,m)
```

For Diverging ID evaluation:

```text
disagreement_score_i,m = -reward_gap_i,m
```

Therefore:

- smaller reward gap means a higher predicted tendency toward human disagreement;
- larger reward gap means a higher predicted tendency toward human agreement.

Reward gap will be described as a **reward-gap disagreement diagnostic** or **preference-margin proxy**, not as calibrated model confidence or epistemic uncertainty.

---

## 7. Two-Level Scenario Taxonomy

Task scenario and disagreement source are treated as separate annotation axes.

### 7.1 Task Scenario

Each pair receives one primary task-scenario label:

- `factual_qa`
- `technical_or_coding`
- `medical_or_health`
- `safety_or_sensitive`
- `general_advice`
- `writing_or_creative`
- `reasoning_or_math`
- `summarization_or_transformation`
- `other`

The label is assigned from the prompt's main task.

### 7.2 Disagreement Source

A pair may receive multiple disagreement-source labels:

- `verbosity_or_concision`
- `format_or_style`
- `task_ambiguity`
- `expertise_or_complexity`
- `refusal_strategy`
- `factuality`
- `instruction_following`
- `value_or_subjective_preference`
- `other`

Labels are assigned from the prompt and the contrast between the responses.

### 7.3 Annotation rule

Scenario labels must be assigned without using:

- reward scores;
- reward gaps;
- model predictions;
- whether the pair later becomes a failure case.

Human vote counts should be hidden during initial scenario annotation where practical.

---

## 8. Experimental Scope

### 8.1 Pipeline test

Use `50–100` pairs to verify:

- data loading;
- label alignment;
- reward scoring;
- scenario annotation format;
- AUROC computation;
- output-file generation.

These results are not the final confirmatory result.

### 8.2 Formal v1 experiment

Target scope:

- dataset: MultiPref;
- total target sample: approximately `400` pairs;
- target class balance: approximately `200` diverging and `200` high-agreement pairs;
- reward models: `2` open scalar reward models;
- no reward-model training;
- no LLM-as-a-Judge requirement;
- no downstream PPO or policy optimization.

Any scope deviation should be documented before examining per-scenario results.

---

## 9. Primary Outcomes

### 9.1 Overall Diverging ID AUROC

For each reward model:

```text
label = 1 for diverging_paper_like
label = 0 for high_agreement_pref
score = -reward_gap
```

Interpretation:

- `0.5`: approximately random ranking;
- greater than `0.5`: some disagreement-diagnostic signal;
- closer to `1.0`: stronger ranking ability.

### 9.2 Per-scenario Diverging ID AUROC

Compute AUROC separately within each task-scenario category.

A category enters the primary comparison only when it has at least:

```text
20 diverging pairs
and
20 high-agreement pairs
```

Smaller categories are reported descriptively.

### 9.3 Overconfident-disagreement rate

Raw gaps are not directly comparable across models.

For each model separately:

1. compute `reward_gap` for all formal-analysis pairs;
2. convert gaps to empirical percentile ranks within that model;
3. define a large gap as percentile rank `>= 0.75`.

```text
overconfident_disagreement =
    diverging_paper_like
    and (modelwise_gap_percentile >= 0.75)
```

Report counts, denominators, and rates by task scenario and disagreement source.

The `0.75` threshold is fixed before final analysis. Other thresholds may appear only as labeled sensitivity analyses.

### 9.4 Preference accuracy

For pairs with a non-ambiguous majority:

```text
model_pred = "A" if reward_A > reward_B
model_pred = "B" if reward_B > reward_A
```

```text
preference_accuracy =
    mean(model_pred == majority_label)
```

Report overall accuracy and, secondarily, accuracy by task scenario.

---

## 10. Statistical Analysis

### 10.1 Confidence intervals

For overall and per-scenario AUROC:

- report point estimates;
- use stratified bootstrap `95%` confidence intervals;
- use `2,000` bootstrap resamples;
- resample diverging and high-agreement pairs separately.

For overconfident-disagreement rates:

- report point estimates;
- report bootstrap or binomial `95%` confidence intervals;
- always report denominators.

### 10.2 Cross-scenario comparisons

For selected pre-specified comparisons, report bootstrap confidence intervals for:

```text
AUROC_scenario_1 - AUROC_scenario_2
```

If intervals are wide or overlap zero, keep conclusions descriptive.

### 10.3 Cross-model consistency

Compare across models:

- direction of scenario AUROC deviations;
- rank ordering of scenario AUROCs;
- overconfident-disagreement rates;
- overlap of failure pairs.

A pattern is called cross-model consistent only when the direction is similar across both models.

---

## 11. Data and Compute Controls

### 11.1 Length filtering

The previous reproduction used:

```text
total_text_chars <= 4000
```

For v1, apply the same limit across models unless a common token-based limit is fixed before result inspection.

### 11.2 Sampling

Use:

```text
random_state = 42
```

Save formal sample IDs before scoring so every model sees the same pairs.

### 11.3 Score-scale comparability

Do not compare raw reward-gap magnitudes across models.

Allowed cross-model comparisons include:

- AUROC;
- within-model gap percentile;
- within-model standardized gap;
- failure rates defined with the same within-model percentile threshold.

---

## 12. Claims This Experiment Cannot Make

This experiment cannot directly claim that:

- reward gap is calibrated epistemic uncertainty;
- human disagreement means equal utility;
- scenario differences are causal;
- one RM represents all RMs;
- MultiPref results generalize to all datasets;
- the taxonomy is complete;
- every large gap is an error;
- the patterns imply downstream policy failure.

This is an empirical diagnostic and failure analysis.

---

## 13. Main Intended Contribution

The intended contribution is not to propose reward gap as a new uncertainty metric.

The intended contribution is:

> to systematically evaluate when an existing scalar reward-gap diagnostic succeeds or fails at identifying human preference disagreement, with particular attention to variation across task scenarios and disagreement sources.

A future extension may test generalization across additional reward models and preference datasets.
