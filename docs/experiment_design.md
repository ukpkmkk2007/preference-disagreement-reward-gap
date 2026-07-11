# Experiment Design

## 1. Objective

This study evaluates whether the absolute score gap from a scalar reward model can serve as a diagnostic signal for human preference disagreement.

The central empirical question is not whether reward gap contains any signal in aggregate, but whether that signal is stable across task scenarios and reward-model checkpoints.

## 2. Research Questions

1. Can scalar reward models recover the strict human majority preference?
2. Can a small absolute reward gap identify examples with annotator disagreement?
3. Does disagreement-detection performance differ across task scenarios?
4. Does the relative performance of reward models change across task scenarios?
5. How often do models produce large reward gaps on examples where humans disagree?

## 3. Dataset

The formal analysis uses a subset of MultiPref.

The project does not redistribute raw MultiPref prompts or responses. Dataset access and use should follow the original source and license terms.

## 4. Formal Sample

The formal sample contains 400 comparison pairs:

- 200 diverging examples;
- 200 high-agreement examples.

The sample is stratified across six task scenarios:

| Task scenario | Diverging | High agreement | Total |
|---|---:|---:|---:|
| Open QA | 40 | 40 | 80 |
| Generation | 35 | 35 | 70 |
| Coding | 35 | 35 | 70 |
| Chat | 35 | 35 | 70 |
| Brainstorm | 35 | 35 | 70 |
| Closed QA | 20 | 20 | 40 |
| **Total** | **200** | **200** | **400** |

The maximum retained text length was approximately 4,000 characters to keep local inference feasible.

## 5. Human Preference Labels

The source table contains annotator vote counts including:

```text
n_A
n_B
n_Tie
```

A strict human majority label is defined as:

```text
A, if n_A > n_B
B, if n_B > n_A
no strict majority, if n_A == n_B
```

`n_Tie` records individual tie votes. It is not itself treated as a row-level exclusion criterion.

Preference accuracy is computed only for rows with a strict human majority.

## 6. Disagreement Labels

The formal evaluation contrasts:

- diverging examples, where annotator preferences are meaningfully split;
- high-agreement examples, where annotators show comparatively strong agreement.

The precise inclusion rules are fixed by the frozen annotation and sample-selection pipeline. Downstream analysis scripts must not alter these labels.

## 7. Reward Models

The formal analysis uses two scalar reward models:

1. `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`
2. `sfairXC/FsfairX-LLaMA3-RM-v0.1`

The project performs inference only. No reward model is trained or fine-tuned.

Each model assigns one scalar score to response A and one scalar score to response B.

## 8. Primary Diagnostic Score

For each model and comparison pair:

```text
reward_gap = abs(score_a - score_b)
diverging_score = -reward_gap
```

The negative sign is used because a smaller absolute gap is hypothesized to indicate greater model uncertainty and therefore a higher likelihood of human disagreement.

## 9. Primary Metrics

### 9.1 Diverging-ID AUROC

AUROC evaluates whether `diverging_score` separates diverging examples from high-agreement examples.

AUROC is reported:

- overall;
- separately for each task scenario;
- with stratified bootstrap confidence intervals.

### 9.2 Preference Accuracy

For rows with a strict human majority:

```text
model_prediction = A, if score_a > score_b
model_prediction = B, if score_b > score_a
```

Preference accuracy is reported:

- overall;
- by disagreement stratum;
- by task scenario.

Exact model-score ties should be handled explicitly by the analysis script rather than silently assigned to either response.

## 10. Statistical Procedures

### 10.1 Bootstrap confidence intervals

The project uses 2,000 paired stratified bootstrap replicates.

Resampling preserves the comparison between the two reward models on the same examples. The same resampled row indices are used for both models within each replicate.

The reported interval is the percentile 95% bootstrap confidence interval.

### 10.2 Paired model comparisons

For each evaluation scope:

```text
AUROC difference = AUROC_FsfairX - AUROC_Skywork
```

The paired bootstrap distribution is used to estimate the confidence interval for this difference.

A confidence interval excluding zero is treated as evidence of a model difference within that scope. Because multiple task-level comparisons are performed, these findings should remain exploratory unless multiplicity is formally controlled.

### 10.3 Cross-task consistency

Pearson and Spearman correlations compare the six task-level AUROC values across the two models.

With only six task scenarios, these correlations are descriptive and should not be treated as strong inferential evidence.

## 11. Operational High-Gap Disagreement Analysis

For each reward model, the primary threshold is the 75th percentile of the absolute reward-gap distribution among high-agreement examples.

A diverging example is flagged when:

```text
abs_reward_gap >= model_specific_Q75_threshold
```

The high-gap disagreement rate is:

```text
number of flagged diverging examples / number of diverging examples
```

Wilson confidence intervals are reported for the rate.

Sensitivity analyses also use the 50th and 90th percentile thresholds.

This measure is an operational failure diagnostic. It must not be described as a calibrated probability of confidence.

## 12. Manual Subgroup Analysis

Manual labels include task scenarios and potentially overlapping disagreement sources.

Because an example may have more than one disagreement source, subgroup counts are not mutually exclusive. These analyses are exploratory and are not used as the primary evidence for the main claim.

## 13. Freeze and Integrity Policy

Core inputs are frozen before statistical analysis.

Analysis scripts must:

1. read frozen files without modifying them;
2. verify row counts and identifiers;
3. verify SHA-256 checksums when available;
4. write new outputs to separate files;
5. fail explicitly when expected columns or hashes do not match.

The analysis should remain reproducible from the frozen sample IDs, frozen reward-model score tables, and frozen annotation labels.
