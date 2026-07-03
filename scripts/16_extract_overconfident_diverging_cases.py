# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:43:16 2026

@author: 23624
"""
import pandas as pd
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("results/step12_scenario_pilot_labeled_manual.csv")

# 只看人类有分歧，但 reward model 很自信的失败样本
failure_df = df[df["overconfident_diverging"] == True].copy()

# 按 gap 从大到小排
failure_df = failure_df.sort_values("score_gap_abs", ascending=False)

cols = [
    "comparison_id",
    "scenario_manual",
    "scenario_manual_reason",
    "score_gap_abs",
    "skywork_score_a",
    "skywork_score_b",
    "n_A",
    "n_B",
    "n_Tie",
    "prompt",
    "response_a",
    "response_b"
]

failure_cases = failure_df[cols]

out_path = "results/step13_overconfident_diverging_failure_cases.csv"
failure_cases.to_csv(out_path, index=False, encoding="utf-8-sig")

print("Number of failure cases:", len(failure_cases))
print(failure_cases[[
    "scenario_manual",
    "score_gap_abs",
    "n_A",
    "n_B",
    "n_Tie",
    "scenario_manual_reason"
]].head(20))

print("Saved:", out_path)
