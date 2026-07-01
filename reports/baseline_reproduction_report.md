# Baseline Reproduction Report

## 1. Goal

This report summarizes a compute-constrained reproduction of the single-value reward model baseline evaluation pipeline from *Diverging Preferences: When do Annotators Disagree and do Models Know?*

The original paper evaluates reward models on two main types of metrics:

1. Preference Accuracy: whether the reward model prefers the same response as the human majority.
2. Diverging ID AUROC: whether the model's reward gap can identify preference pairs where annotators disagree.

Due to local compute constraints, I use `Skywork-Reward-Llama-3.1-8B-v0.2` instead of the larger Skywork 27B model used in the paper.

## 2. Model and Setup

- Model: `Skywork-Reward-Llama-3.1-8B-v0.2`
- Inference: 4-bit quantized inference
- Training: No model training
- Dataset: MultiPref
- Hardware: local RTX 5070 Laptop GPU
- Max text filter: `total_text_chars <= 4000` for small-scale local feasibility

## 3. Preference Accuracy

For each preference pair, I compute:

- `score_a = RM(prompt, response_a)`
- `score_b = RM(prompt, response_b)`

Prediction rule:

- predict A if `score_a > score_b`
- predict B if `score_b > score_a`

Then I compare this prediction against the human majority preference.

### Result

- Number of examples: 100
- Correct: 75
- Wrong: 25
- Preference Accuracy: 0.7500

### Majority-class baseline

The majority label distribution in this random100 subset is:

majority_label
B    66
A    34

The majority-class baseline is always predicting `B`, which would get:

- Majority-class baseline accuracy: 0.6600

Therefore, Skywork 8B improves over this simple majority-class baseline by:

- Improvement: 0.0900

### Accuracy by majority label

majority_label  count  accuracy
             A     34  0.588235
             B     66  0.833333

## 4. Diverging ID AUROC

For Diverging ID, I compare:

- Positive class: `diverging_paper_like = True`
- Negative class: `high_agreement_pref = True`

For each pair, I compute:

- `score_gap_abs = abs(score_a - score_b)`
- `diverging_score = -score_gap_abs`

The intuition is that smaller reward gaps indicate lower model certainty, so pairs with smaller gaps should be more likely to be diverging.

### Result

- Number of examples: 100
- Diverging examples: 50
- High-agreement examples: 50
- Diverging ID AUROC: 0.6244

### Reward gap by class

 diverging_id_label  count      mean    median       std
                  0     50 14.659980 11.988525 12.129356
                  1     50 10.131574  7.362305  9.346144

Interpretation:

The mean reward gap is smaller for diverging examples than for high-agreement examples:

- Mean gap, diverging: 10.1316
- Mean gap, high-agreement: 14.6600

This suggests that Skywork 8B's reward gap contains a weak-to-moderate signal for identifying human preference disagreement.

## 5. Important Limitations

This is not an exact reproduction of the paper's full Skywork 27B Table 3 result.

Main limitations:

1. I use Skywork 8B instead of Skywork 27B.
2. I use 4-bit quantized inference due to local 8GB GPU memory.
3. The evaluation uses small sampled subsets rather than the full MultiPref benchmark.
4. Long examples are filtered with `total_text_chars <= 4000`.
5. The reported results should be interpreted as a small-scale reproduction of the baseline evaluation pipeline, not as the paper's exact full-scale result.

## 6. Generated Files

Main outputs:

- `results/step5_skywork_8b_random100_results.csv`
- `results/step5_skywork_8b_random100_summary.csv`
- `results/step6_skywork_8b_diverging_id_results.csv`
- `results/step6_skywork_8b_diverging_id_summary.csv`

Step 7 analysis outputs:

- `results/step7_baseline_reproduction_metrics.csv`
- `results/step7_pref_accuracy_by_majority_label.csv`
- `results/step7_pref_accuracy_by_category.csv`
- `results/step7_preference_errors_all.csv`
- `results/step7_preference_errors_high_confidence_top10.csv`
- `results/step7_diverging_gap_by_class.csv`
- `results/step7_diverging_gap_by_category.csv`
- `results/step7_high_agreement_with_small_gap_top10.csv`
- `results/step7_diverging_with_large_gap_top10.csv`

## 7. Short Summary

I successfully reproduced the single-value reward model baseline evaluation pipeline using Skywork 8B under local compute constraints.

The small-scale results are:

- Preference Accuracy: 0.7500
- Diverging ID AUROC: 0.6244

The preference accuracy result shows that Skywork 8B can recover majority human preferences above a simple majority-class baseline on the sampled subset. The Diverging ID result shows that reward gap is smaller for diverging examples than high-agreement examples, suggesting that the model has some signal for recognizing disagreement, but the signal is limited.
