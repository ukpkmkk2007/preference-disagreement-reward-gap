# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:23:37 2026

@author: 23624
"""

import os
import pandas as pd
import numpy as np

PROJECT_DIR = r"C:\Users\23624\Desktop\preference_disagreement_baseline"
os.chdir(PROJECT_DIR)

os.makedirs("results", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("Current working directory:")
print(os.getcwd())


# =========================
# 1. 读取 Step 5: Preference Accuracy 结果
# =========================

pref_result_path = "results/step5_skywork_8b_random100_results.csv"
pref_summary_path = "results/step5_skywork_8b_random100_summary.csv"

if not os.path.exists(pref_result_path):
    raise FileNotFoundError("Cannot find " + pref_result_path)

if not os.path.exists(pref_summary_path):
    raise FileNotFoundError("Cannot find " + pref_summary_path)

pref_df = pd.read_csv(pref_result_path)
pref_summary = pd.read_csv(pref_summary_path)

print("\nLoaded preference accuracy results.")
print("Rows:", len(pref_df))


# =========================
# 2. 读取 Step 6: Diverging ID AUROC 结果
# =========================

div_result_path = "results/step6_skywork_8b_diverging_id_results.csv"
div_summary_path = "results/step6_skywork_8b_diverging_id_summary.csv"

if not os.path.exists(div_result_path):
    raise FileNotFoundError("Cannot find " + div_result_path)

if not os.path.exists(div_summary_path):
    raise FileNotFoundError("Cannot find " + div_summary_path)

div_df = pd.read_csv(div_result_path)
div_summary = pd.read_csv(div_summary_path)

print("\nLoaded Diverging ID results.")
print("Rows:", len(div_df))


# =========================
# 3. Preference Accuracy 细分分析
# =========================

overall_accuracy = pref_df["correct"].mean()
n_pref = len(pref_df)
n_correct = int(pref_df["correct"].sum())
n_wrong = int((~pref_df["correct"]).sum())

label_counts = pref_df["majority_label"].value_counts()
majority_class_label = label_counts.idxmax()
majority_class_acc = label_counts.max() / len(pref_df)

acc_by_majority_label = (
    pref_df.groupby("majority_label")["correct"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(columns={"mean": "accuracy"})
)

acc_by_category = (
    pref_df.groupby("category")["correct"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(columns={"mean": "accuracy"})
    .sort_values("count", ascending=False)
)

pref_error_df = pref_df[pref_df["correct"] == False].copy()
pref_error_df = pref_error_df.sort_values("score_gap_abs", ascending=False)

high_conf_error_df = pref_error_df.head(10).copy()

acc_by_majority_label.to_csv(
    "results/step7_pref_accuracy_by_majority_label.csv",
    index=False,
    encoding="utf-8-sig"
)

acc_by_category.to_csv(
    "results/step7_pref_accuracy_by_category.csv",
    index=False,
    encoding="utf-8-sig"
)

pref_error_df.to_csv(
    "results/step7_preference_errors_all.csv",
    index=False,
    encoding="utf-8-sig"
)

high_conf_error_df.to_csv(
    "results/step7_preference_errors_high_confidence_top10.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================
# 4. Diverging ID 细分分析
# =========================

div_auroc = float(div_summary.loc[0, "diverging_id_auroc"])

gap_by_class = (
    div_df.groupby("diverging_id_label")["score_gap_abs"]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
)

gap_by_category = (
    div_df.groupby(["category", "diverging_id_label"])["score_gap_abs"]
    .agg(["count", "mean", "median"])
    .reset_index()
    .sort_values(["category", "diverging_id_label"])
)

# 最像 diverging 的 high-agreement 样本：label=0 但 gap 很小
false_div_like_high = (
    div_df[div_df["diverging_id_label"] == 0]
    .sort_values("score_gap_abs", ascending=True)
    .head(10)
    .copy()
)

# 最不像 diverging 的 diverging 样本：label=1 但 gap 很大
false_high_like_div = (
    div_df[div_df["diverging_id_label"] == 1]
    .sort_values("score_gap_abs", ascending=False)
    .head(10)
    .copy()
)

gap_by_class.to_csv(
    "results/step7_diverging_gap_by_class.csv",
    index=False,
    encoding="utf-8-sig"
)

gap_by_category.to_csv(
    "results/step7_diverging_gap_by_category.csv",
    index=False,
    encoding="utf-8-sig"
)

false_div_like_high.to_csv(
    "results/step7_high_agreement_with_small_gap_top10.csv",
    index=False,
    encoding="utf-8-sig"
)

false_high_like_div.to_csv(
    "results/step7_diverging_with_large_gap_top10.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================
# 5. 生成总 metrics 表
# =========================

metrics = {
    "model": "Skywork-Reward-Llama-3.1-8B-v0.2",
    "setting": "4-bit inference, no training",
    "preference_accuracy_n": n_pref,
    "preference_accuracy": overall_accuracy,
    "n_correct": n_correct,
    "n_wrong": n_wrong,
    "majority_class_label": majority_class_label,
    "majority_class_baseline_accuracy": majority_class_acc,
    "diverging_id_n": len(div_df),
    "diverging_id_auroc": div_auroc,
    "mean_gap_diverging": float(
        div_df[div_df["diverging_id_label"] == 1]["score_gap_abs"].mean()
    ),
    "mean_gap_high_agreement": float(
        div_df[div_df["diverging_id_label"] == 0]["score_gap_abs"].mean()
    ),
    "median_gap_diverging": float(
        div_df[div_df["diverging_id_label"] == 1]["score_gap_abs"].median()
    ),
    "median_gap_high_agreement": float(
        div_df[div_df["diverging_id_label"] == 0]["score_gap_abs"].median()
    ),
}

metrics_df = pd.DataFrame([metrics])

metrics_path = "results/step7_baseline_reproduction_metrics.csv"
metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")


# =========================
# 6. 生成 Markdown report
# =========================

report_path = "reports/baseline_reproduction_report.md"

report = f"""# Baseline Reproduction Report

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

- Number of examples: {n_pref}
- Correct: {n_correct}
- Wrong: {n_wrong}
- Preference Accuracy: {overall_accuracy:.4f}

### Majority-class baseline

The majority label distribution in this random100 subset is:

{label_counts.to_string()}

The majority-class baseline is always predicting `{majority_class_label}`, which would get:

- Majority-class baseline accuracy: {majority_class_acc:.4f}

Therefore, Skywork 8B improves over this simple majority-class baseline by:

- Improvement: {overall_accuracy - majority_class_acc:.4f}

### Accuracy by majority label

{acc_by_majority_label.to_string(index=False)}

## 4. Diverging ID AUROC

For Diverging ID, I compare:

- Positive class: `diverging_paper_like = True`
- Negative class: `high_agreement_pref = True`

For each pair, I compute:

- `score_gap_abs = abs(score_a - score_b)`
- `diverging_score = -score_gap_abs`

The intuition is that smaller reward gaps indicate lower model certainty, so pairs with smaller gaps should be more likely to be diverging.

### Result

- Number of examples: {len(div_df)}
- Diverging examples: {int((div_df["diverging_id_label"] == 1).sum())}
- High-agreement examples: {int((div_df["diverging_id_label"] == 0).sum())}
- Diverging ID AUROC: {div_auroc:.4f}

### Reward gap by class

{gap_by_class.to_string(index=False)}

Interpretation:

The mean reward gap is smaller for diverging examples than for high-agreement examples:

- Mean gap, diverging: {metrics["mean_gap_diverging"]:.4f}
- Mean gap, high-agreement: {metrics["mean_gap_high_agreement"]:.4f}

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

- Preference Accuracy: {overall_accuracy:.4f}
- Diverging ID AUROC: {div_auroc:.4f}

The preference accuracy result shows that Skywork 8B can recover majority human preferences above a simple majority-class baseline on the sampled subset. The Diverging ID result shows that reward gap is smaller for diverging examples than high-agreement examples, suggesting that the model has some signal for recognizing disagreement, but the signal is limited.
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)


print("\nSaved metrics:")
print(metrics_path)

print("\nSaved report:")
print(report_path)

print("\nSaved diagnostic CSV files in results/")

print("\nMain results:")
print(metrics_df.T)