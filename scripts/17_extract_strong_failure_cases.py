# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:52:52 2026

@author: 23624
"""
import os
import pandas as pd
import numpy as np

os.makedirs("results", exist_ok=True)

df = pd.read_csv("results/step12_scenario_pilot_labeled_manual.csv")

def to_bool(x):
    if x is True or x == "True" or x == 1:
        return True
    return False

df["diverging_paper_like"] = df["diverging_paper_like"].apply(to_bool)
df["high_agreement_pref"] = df["high_agreement_pref"].apply(to_bool)

if "overconfident_diverging" not in df.columns:
    gap_threshold = df["score_gap_abs"].quantile(0.75)
    df["overconfident_diverging"] = (
        (df["diverging_paper_like"] == True) &
        (df["score_gap_abs"] >= gap_threshold)
    )
else:
    df["overconfident_diverging"] = df["overconfident_diverging"].apply(to_bool)

# 模型预测 A/B
df["model_pred"] = np.where(
    df["skywork_score_a"] > df["skywork_score_b"],
    "A",
    "B"
)

# 人类分歧是否接近平衡：2-2, 2-1-1, 1-2-1 这种
df["balanced_disagreement"] = (
    (df["n_A"] > 0) &
    (df["n_B"] > 0) &
    (abs(df["n_A"] - df["n_B"]) <= 1)
)

# 模型是否高置信错判多数偏好
df["high_confidence_wrong"] = (
    df["majority_label"].isin(["A", "B"]) &
    (df["model_pred"] != df["majority_label"]) &
    (df["overconfident_diverging"] == True)
)

# 只看人类有分歧但模型很自信的样本
failure_df = df[df["overconfident_diverging"] == True].copy()

failure_df["failure_strength"] = np.where(
    failure_df["balanced_disagreement"] | failure_df["high_confidence_wrong"],
    "strong_failure",
    "weak_failure"
)

failure_df = failure_df.sort_values(
    ["failure_strength", "score_gap_abs"],
    ascending=[True, False]
)

cols = [
    "comparison_id",
    "scenario_manual",
    "failure_strength",
    "score_gap_abs",
    "skywork_score_a",
    "skywork_score_b",
    "model_pred",
    "majority_label",
    "n_A",
    "n_B",
    "n_Tie",
    "balanced_disagreement",
    "high_confidence_wrong",
    "prompt",
    "response_a",
    "response_b"
]

out_path = "results/step14_strong_failure_cases.csv"
failure_df[cols].to_csv(out_path, index=False, encoding="utf-8-sig")

summary = failure_df.groupby(
    ["scenario_manual", "failure_strength"]
).size().reset_index(name="count")

summary_path = "results/step14_failure_case_summary.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("Failure case summary:")
print(summary)

print("\nStrong failure cases:")
print(
    failure_df[failure_df["failure_strength"] == "strong_failure"][
        ["scenario_manual", "score_gap_abs", "model_pred", "majority_label", "n_A", "n_B", "n_Tie"]
    ]
)

print("\nSaved:")
print(out_path)
print(summary_path)
