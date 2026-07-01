# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:40:07 2026

@author: 23624
"""
import pandas as pd
import os
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_DIR)
print("Current working directory:")
print(os.getcwd())

os.makedirs("data_processed", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv("data_processed/multipref_with_diverging_labels.csv")

print("Loaded data.")
print("Rows:", len(df))


def to_bool_series(s):
    """
    把 True/False 或 'True'/'False' 统一转成 bool。
    """
    if s.dtype == bool:
        return s

    return s.astype(str).str.lower().map({
        "true": True,
        "false": False,
        "1": True,
        "0": False
    })


# 确保这些标签是 bool 类型
bool_cols = [
    "diverging_simple",
    "diverging_paper_like",
    "diverging_substantial",
    "high_agreement_pref",
    "tie_only",
    "only_slight_or_tie"
]

for col in bool_cols:
    if col in df.columns:
        df[col] = to_bool_series(df[col])


def majority_label(row):
    """
    根据 annotator 投票得到人类 majority preference。
    只在 A 票数和 B 票数不相等时给出 A 或 B。
    如果 A 和 B 票数相等，就标为 Ambiguous。
    """
    if row["n_A"] > row["n_B"]:
        return "A"
    elif row["n_B"] > row["n_A"]:
        return "B"
    else:
        return "Ambiguous"


df["majority_label"] = df.apply(majority_label, axis=1)

# 是否是可以用于 preference accuracy 的样本：
# 必须 majority 是 A 或 B，不能是 Ambiguous
df["has_majority_preference"] = df["majority_label"].isin(["A", "B"])

# Skywork baseline 的核心输入列
cols = [
    "comparison_id",
    "prompt_id",
    "category",
    "prompt",
    "model_a",
    "response_a",
    "model_b",
    "response_b",

    "raw_overall_labels",
    "parsed_directions",
    "parsed_strengths",

    "n_A",
    "n_B",
    "n_Tie",
    "n_clear",
    "n_slight",
    "n_valid",

    "majority_label",
    "has_majority_preference",

    "diverging_simple",
    "diverging_paper_like",
    "diverging_substantial",
    "high_agreement_pref",
    "tie_only",
    "only_slight_or_tie"
]

baseline_input = df[cols].copy()

out_path = "data_processed/skywork_baseline_inputs.csv"
baseline_input.to_csv(out_path, index=False, encoding="utf-8-sig")

print("\nSaved Skywork baseline input file:")
print(out_path)


# 统计不同 evaluation subset 的数量
summary = {
    "total_examples": len(baseline_input),

    "has_majority_preference": int(baseline_input["has_majority_preference"].sum()),
    "ambiguous_majority": int((baseline_input["majority_label"] == "Ambiguous").sum()),

    "majority_A": int((baseline_input["majority_label"] == "A").sum()),
    "majority_B": int((baseline_input["majority_label"] == "B").sum()),

    "diverging_paper_like": int(baseline_input["diverging_paper_like"].sum()),
    "diverging_substantial": int(baseline_input["diverging_substantial"].sum()),
    "high_agreement_pref": int(baseline_input["high_agreement_pref"].sum()),

    "diverging_or_high_agreement": int(
        (
            (baseline_input["diverging_paper_like"] == True)
            | (baseline_input["high_agreement_pref"] == True)
        ).sum()
    )
}

summary_df = pd.DataFrame([summary])

summary_path = "results/step_baseline1_skywork_input_summary.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\nSummary:")
print(summary_df.T)

print("\nSaved summary:")
print(summary_path)


# 额外导出一个小样本，方便人工检查
check_cols = [
    "comparison_id",
    "prompt",
    "response_a",
    "response_b",
    "raw_overall_labels",
    "parsed_directions",
    "n_A",
    "n_B",
    "n_Tie",
    "majority_label",
    "diverging_paper_like",
    "high_agreement_pref"
]

check_path = "results/step_baseline1_manual_check.csv"

baseline_input[check_cols].sample(
    n=30,
    random_state=42
).to_csv(
    check_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved manual check file:")
print(check_path)
