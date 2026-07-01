# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:18:03 2026

@author: 23624
"""
import pandas as pd
import numpy as np
import os
import re

os.makedirs("data_processed", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv("data_processed/multipref_with_diverging_labels.csv")

print("Loaded data.")
print("Rows:", len(df))


def clean_text(x):
    """
    把缺失值变成空字符串，其他内容变成字符串。
    """
    if pd.isna(x):
        return ""
    return str(x)


def char_count(text):
    text = clean_text(text)
    return len(text)


def word_count(text):
    text = clean_text(text)
    words = text.split()
    return len(words)


def line_count(text):
    text = clean_text(text)
    if text == "":
        return 0
    return len(text.splitlines())


def sentence_count(text):
    text = clean_text(text)
    pieces = re.split(r"[.!?]+", text)
    pieces = [p.strip() for p in pieces if p.strip() != ""]
    return len(pieces)


def bullet_count(text):
    """
    粗略统计 bullet points 数量。
    包括：
    - xxx
    * xxx
    • xxx
    1. xxx
    1) xxx
    """
    text = clean_text(text)
    count = 0

    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^[-*•]\s+", line):
            count += 1
        elif re.match(r"^\d+[\.\)]\s+", line):
            count += 1

    return count


def code_block_count(text):
    text = clean_text(text)
    return text.count("```")


def question_mark_count(text):
    text = clean_text(text)
    return text.count("?")


def exclamation_mark_count(text):
    text = clean_text(text)
    return text.count("!")


refusal_phrases = [
    "i can't",
    "i cannot",
    "i can’t",
    "i'm sorry",
    "i am sorry",
    "sorry, but",
    "i cannot help",
    "i can't help",
    "i’m unable",
    "i am unable",
    "as an ai",
    "not able to",
    "unable to assist",
    "cannot assist",
]


def has_refusal(text):
    """
    判断回答里是否出现常见拒答表达。
    """
    text = clean_text(text).lower()
    for phrase in refusal_phrases:
        if phrase in text:
            return 1
    return 0


caution_phrases = [
    "consult a doctor",
    "consult your doctor",
    "healthcare professional",
    "medical professional",
    "seek medical advice",
    "talk to your doctor",
    "it depends",
    "may vary",
    "individual circumstances",
    "not medical advice",
]


def has_caution(text):
    """
    判断回答里是否出现谨慎/免责声明表达。
    """
    text = clean_text(text).lower()
    for phrase in caution_phrases:
        if phrase in text:
            return 1
    return 0


# prompt features
df["prompt_chars"] = df["prompt"].apply(char_count)
df["prompt_words"] = df["prompt"].apply(word_count)
df["prompt_questions"] = df["prompt"].apply(question_mark_count)

# response A features
df["response_a_chars"] = df["response_a"].apply(char_count)
df["response_a_words"] = df["response_a"].apply(word_count)
df["response_a_lines"] = df["response_a"].apply(line_count)
df["response_a_sentences"] = df["response_a"].apply(sentence_count)
df["response_a_bullets"] = df["response_a"].apply(bullet_count)
df["response_a_code_blocks"] = df["response_a"].apply(code_block_count)
df["response_a_questions"] = df["response_a"].apply(question_mark_count)
df["response_a_exclamations"] = df["response_a"].apply(exclamation_mark_count)
df["response_a_refusal"] = df["response_a"].apply(has_refusal)
df["response_a_caution"] = df["response_a"].apply(has_caution)

# response B features
df["response_b_chars"] = df["response_b"].apply(char_count)
df["response_b_words"] = df["response_b"].apply(word_count)
df["response_b_lines"] = df["response_b"].apply(line_count)
df["response_b_sentences"] = df["response_b"].apply(sentence_count)
df["response_b_bullets"] = df["response_b"].apply(bullet_count)
df["response_b_code_blocks"] = df["response_b"].apply(code_block_count)
df["response_b_questions"] = df["response_b"].apply(question_mark_count)
df["response_b_exclamations"] = df["response_b"].apply(exclamation_mark_count)
df["response_b_refusal"] = df["response_b"].apply(has_refusal)
df["response_b_caution"] = df["response_b"].apply(has_caution)

# pair-level difference features
df["char_diff_abs"] = (df["response_a_chars"] - df["response_b_chars"]).abs()
df["word_diff_abs"] = (df["response_a_words"] - df["response_b_words"]).abs()
df["line_diff_abs"] = (df["response_a_lines"] - df["response_b_lines"]).abs()
df["sentence_diff_abs"] = (df["response_a_sentences"] - df["response_b_sentences"]).abs()
df["bullet_diff_abs"] = (df["response_a_bullets"] - df["response_b_bullets"]).abs()
df["code_block_diff_abs"] = (df["response_a_code_blocks"] - df["response_b_code_blocks"]).abs()

# ratio features
df["word_ratio_max_min"] = (
    np.maximum(df["response_a_words"], df["response_b_words"])
    / (np.minimum(df["response_a_words"], df["response_b_words"]) + 1)
)

df["char_ratio_max_min"] = (
    np.maximum(df["response_a_chars"], df["response_b_chars"])
    / (np.minimum(df["response_a_chars"], df["response_b_chars"]) + 1)
)

# mismatch features
df["refusal_mismatch"] = (
    df["response_a_refusal"] != df["response_b_refusal"]
).astype(int)

df["caution_mismatch"] = (
    df["response_a_caution"] != df["response_b_caution"]
).astype(int)

df["bullet_mismatch"] = (
    (df["response_a_bullets"] > 0) != (df["response_b_bullets"] > 0)
).astype(int)

df["code_block_mismatch"] = (
    (df["response_a_code_blocks"] > 0) != (df["response_b_code_blocks"] > 0)
).astype(int)

df["format_mismatch_simple"] = (
    (df["bullet_mismatch"] == 1)
    | (df["code_block_mismatch"] == 1)
    | (df["line_diff_abs"] >= 3)
).astype(int)

feature_cols = [
    "prompt_chars",
    "prompt_words",
    "prompt_questions",

    "response_a_chars",
    "response_a_words",
    "response_a_lines",
    "response_a_sentences",
    "response_a_bullets",
    "response_a_code_blocks",
    "response_a_questions",
    "response_a_exclamations",
    "response_a_refusal",
    "response_a_caution",

    "response_b_chars",
    "response_b_words",
    "response_b_lines",
    "response_b_sentences",
    "response_b_bullets",
    "response_b_code_blocks",
    "response_b_questions",
    "response_b_exclamations",
    "response_b_refusal",
    "response_b_caution",

    "char_diff_abs",
    "word_diff_abs",
    "line_diff_abs",
    "sentence_diff_abs",
    "bullet_diff_abs",
    "code_block_diff_abs",
    "word_ratio_max_min",
    "char_ratio_max_min",

    "refusal_mismatch",
    "caution_mismatch",
    "bullet_mismatch",
    "code_block_mismatch",
    "format_mismatch_simple",
]

out_path = "data_processed/multipref_with_features.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("Saved feature file:")
print(out_path)

print("\nFeature preview:")
print(df[feature_cols].head())

summary = df[feature_cols].describe().T
summary_path = "results/stepH_feature_summary.csv"
summary.to_csv(summary_path, encoding="utf-8-sig")

print("\nSaved feature summary:")
print(summary_path)

if "diverging_paper_like" in df.columns:
    by_diverging = df.groupby("diverging_paper_like")[feature_cols].mean().T
    by_diverging_path = "results/stepH_features_by_diverging.csv"
    by_diverging.to_csv(by_diverging_path, encoding="utf-8-sig")

    print("\nMean features by diverging_paper_like:")
    print(by_diverging.head(30))
    print("\nSaved group comparison:")
    print(by_diverging_path)
