# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 18:50:21 2026

@author: 23624
"""
import os
import pandas as pd
import numpy as np

os.makedirs("results", exist_ok=True)

input_path = "results/step6_skywork_8b_diverging_id_results.csv"
df = pd.read_csv(input_path)

# 如果没有 gap，就现算
if "score_gap_abs" not in df.columns:
    df["score_gap_abs"] = (df["skywork_score_a"] - df["skywork_score_b"]).abs()


def word_count(x):
    return len(str(x).split())


def contains_any(text, keywords):
    text = str(text).lower()
    return any(k in text for k in keywords)


def simple_auc(y_true, scores):
    temp = pd.DataFrame({"y": y_true, "s": scores}).dropna()

    pos = temp[temp["y"] == 1]["s"].tolist()
    neg = temp[temp["y"] == 0]["s"].tolist()

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    total = 0
    correct = 0

    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                correct += 1
            elif p == n:
                correct += 0.5

    return correct / total


# 文本长度特征
df["a_words"] = df["response_a"].apply(word_count)
df["b_words"] = df["response_b"].apply(word_count)

min_words = df[["a_words", "b_words"]].min(axis=1).replace(0, 1)
max_words = df[["a_words", "b_words"]].max(axis=1)

df["word_ratio"] = max_words / min_words

aall_text = (
    df["prompt"].astype(str) + " " +
    df["response_a"].astype(str) + " " +
    df["response_b"].astype(str)
)

safety_words = [
    "sorry", "cannot", "can't", "unable", "illegal", "harmful",
    "dangerous", "safe", "safety", "risk", "refuse", "policy"
]

technical_words = [
    "code", "python", "function", "algorithm", "error", "math",
    "equation", "medical", "doctor", "treatment", "symptom",
    "diagnosis", "legal", "law"
]

ambiguity_words = [
    "best", "better", "should i", "which", "recommend", "advice",
    "help me decide", "what should", "pros and cons"
]
all_text = (
    df["prompt"].astype(str) + " " +
    df["response_a"].astype(str) + " " +
    df["response_b"].astype(str)
)

is_safety_refusal = all_text.apply(lambda x: contains_any(x, safety_words))
is_technical_medical = all_text.apply(lambda x: contains_any(x, technical_words))
is_task_ambiguity = all_text.apply(lambda x: contains_any(x, ambiguity_words))
is_verbosity = df["word_ratio"] >= 1.8

df["is_safety_refusal"] = is_safety_refusal
df["is_technical_medical"] = is_technical_medical
df["is_task_ambiguity"] = is_task_ambiguity
df["is_verbosity"] = is_verbosity

df["scenario"] = np.select(
    [
        is_safety_refusal,
        is_technical_medical,
        is_task_ambiguity,
        is_verbosity
    ],
    [
        "safety_refusal",
        "technical_medical",
        "task_ambiguity",
        "verbosity_concision"
    ],
    default="other"
)
print(df[["is_safety_refusal", "is_technical_medical", "is_task_ambiguity", "is_verbosity", "scenario"]].head())

# 高 gap：全样本 gap 最大的 25%
gap_threshold = df["score_gap_abs"].quantile(0.75)

df["overconfident_diverging"] = (
    (df["diverging_paper_like"] == True) &
    (df["score_gap_abs"] >= gap_threshold)
)

rows = []

for scenario, g in df.groupby("scenario"):
    n = len(g)

    auc = simple_auc(
        y_true=g["diverging_id_label"],
        scores=-g["score_gap_abs"]
    )

    diverging_part = g[g["diverging_paper_like"] == True]
    high_part = g[g["high_agreement_pref"] == True]

    rows.append({
        "scenario": scenario,
        "n": n,
        "n_diverging": int(g["diverging_paper_like"].sum()),
        "n_high_agreement": int(g["high_agreement_pref"].sum()),
        "mean_gap": g["score_gap_abs"].mean(),
        "mean_gap_diverging": diverging_part["score_gap_abs"].mean(),
        "mean_gap_high_agreement": high_part["score_gap_abs"].mean(),
        "diverging_id_auc": auc,
        "overconfident_diverging_count": int(g["overconfident_diverging"].sum()),
        "overconfident_rate_among_diverging": diverging_part["overconfident_diverging"].mean()
    })

summary = pd.DataFrame(rows).sort_values("n", ascending=False)

df.to_csv(
    "results/step12_scenario_pilot_labeled.csv",
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    "results/step12_scenario_pilot_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print(summary)
