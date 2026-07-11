# Results Summary

## 1. Primary Finding

Both scalar reward models contain a modest signal for identifying human preference disagreement from the absolute reward gap. However, the strength of this signal differs substantially across task scenarios.

The current results support a task-dependence hypothesis, but they do not yet establish that the observed task-level patterns are stable across additional models and datasets.

## 2. Overall Disagreement Detection

| Reward model | AUROC | 95% bootstrap CI | Mean absolute gap: diverging | Mean absolute gap: high agreement |
|---|---:|---:|---:|---:|
| Skywork 8B | 0.602 | [0.550, 0.656] | 7.417 | 10.618 |
| FsfairX 8B | 0.630 | [0.578, 0.683] | 2.430 | 3.566 |

For both models, diverging examples have a smaller mean reward gap than high-agreement examples.

The paired overall AUROC difference was:

```text
FsfairX - Skywork = 0.028
95% bootstrap CI = [-0.017, 0.075]
```

The interval includes zero, so the overall model difference is inconclusive.

## 3. Task-Specific Disagreement Detection

| Task scenario | Skywork AUROC | FsfairX AUROC |
|---|---:|---:|
| Open QA | 0.737 | 0.759 |
| Generation | 0.560 | 0.564 |
| Coding | 0.537 | 0.669 |
| Chat | 0.537 | 0.556 |
| Brainstorm | 0.648 | 0.705 |
| Closed QA | 0.580 | 0.423 |

### Bootstrap interpretation

- Open QA: both models show relatively strong positive signal.
- Generation: both confidence intervals include 0.5.
- Coding: FsfairX performs better than Skywork in the paired comparison.
- Chat: both confidence intervals include 0.5.
- Brainstorm: point estimates are positive, but uncertainty remains.
- Closed QA: Skywork performs better than FsfairX in the paired comparison.

Paired AUROC differences:

| Scope | FsfairX - Skywork | 95% bootstrap CI | Interpretation |
|---|---:|---:|---|
| Overall | 0.028 | [-0.017, 0.075] | Inconclusive |
| Open QA | 0.022 | [-0.072, 0.105] | Inconclusive |
| Generation | 0.004 | [-0.117, 0.135] | Inconclusive |
| Coding | 0.131 | [0.016, 0.261] | FsfairX higher |
| Chat | 0.019 | [-0.123, 0.159] | Inconclusive |
| Brainstorm | 0.057 | [-0.053, 0.163] | Inconclusive |
| Closed QA | -0.158 | [-0.303, -0.005] | Skywork higher |

Because multiple task comparisons are performed and task-level sample sizes are limited, these two apparent reversals should be treated as replication targets rather than final conclusions.

## 4. Cross-Task Model Consistency

Across the six task scenarios:

```text
Pearson correlation = 0.629
p = 0.181

Spearman correlation = 0.551
p = 0.257
```

The point estimates suggest moderate similarity, but six task categories are insufficient for strong inference.

## 5. Preference Accuracy

Human majority counts:

| Majority status | Count |
|---|---:|
| A majority | 104 |
| B majority | 240 |
| No strict majority | 56 |
| Total | 400 |

Preference accuracy is computed on the 344 strict-majority examples.

### Overall

| Reward model | Preference accuracy |
|---|---:|
| Skywork 8B | 0.703 |
| FsfairX 8B | 0.782 |

### By disagreement stratum

| Stratum | Skywork 8B | FsfairX 8B |
|---|---:|---:|
| Diverging, strict majority only | 0.590 | 0.660 |
| High agreement | 0.785 | 0.870 |

Both models are less accurate on diverging examples.

### By task scenario

| Task scenario | Skywork 8B | FsfairX 8B |
|---|---:|---:|
| Open QA | 0.803 | 0.859 |
| Generation | 0.689 | 0.770 |
| Coding | 0.702 | 0.789 |
| Chat | 0.672 | 0.787 |
| Brainstorm | 0.717 | 0.817 |
| Closed QA | 0.559 | 0.559 |

## 6. Operational High-Gap Disagreement Rate

### Thresholds from high-agreement examples

| Reward model | Q50 | Q75 | Q90 |
|---|---:|---:|---:|
| Skywork 8B | 8.106 | 14.919 | 24.724 |
| FsfairX 8B | 2.879 | 5.429 | 7.348 |

### Primary Q75 result among diverging examples

| Reward model | Flagged / total | Rate | Wilson 95% CI |
|---|---:|---:|---:|
| Skywork 8B | 25 / 200 | 0.125 | [0.086, 0.178] |
| FsfairX 8B | 20 / 200 | 0.100 | [0.066, 0.149] |

Among diverging examples with a strict human majority:

- Skywork produced 17 high-gap cases, including 3 majority-wrong predictions.
- FsfairX produced 14 high-gap cases, including 4 majority-wrong predictions.

These numbers describe operationally large reward gaps. They do not establish calibrated model confidence.

## 7. Exploratory Manual Subgroups

Point estimates suggest possible differences by disagreement source:

- FsfairX appears stronger on factuality, expertise or complexity, verbosity or concision, technical or coding, and factual QA.
- Skywork appears stronger on safety or sensitive content, reasoning or mathematics, and refusal strategy.

These subgroup labels overlap, confidence intervals were not used for the main interpretation, and no multiplicity adjustment was applied. They should be treated as exploratory observations only.

## 8. Supported Claims

The current results support the following cautious statements:

1. Absolute reward gap contains a modest aggregate signal for human disagreement.
2. The signal is not equally reliable across task scenarios.
3. The relative performance of two scalar reward models may change across tasks.
4. Preference prediction becomes more difficult on diverging examples.
5. Large reward gaps can still occur on human-disagreement cases.

## 9. Claims Not Supported Yet

The current study does not establish that:

1. reward gap is a calibrated uncertainty estimate;
2. one of the two reward models is universally superior;
3. the Coding and Closed QA ranking reversals generalize;
4. the task-level patterns hold on other datasets;
5. the project introduces a new disagreement-detection method;
6. the operational Q75 threshold is a standard metric.

## 10. Required Validation

The next stage should prioritize:

1. additional reward-model checkpoints;
2. a second multi-annotator preference dataset;
3. larger task-level samples;
4. preregistered or frozen task definitions;
5. model-by-task interaction analysis;
6. held-out validation of any task-aware diagnostic.
