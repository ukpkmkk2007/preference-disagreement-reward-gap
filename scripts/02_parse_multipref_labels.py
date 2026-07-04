# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 19:19:29 2026

@author: 23624
"""
from datasets import load_dataset
import pandas as pd
import os
from collections import Counter

os.makedirs("data_processed", exist_ok=True)
os.makedirs("results", exist_ok=True)
def parse_overall_pref(label):
    """
    把 MultiPref 的 overall_pref 解析成两个信息：
    1. direction: A / B / Tie / Invalid
    2. strength: clear / slight / tie / invalid
    """

    if label is None:
        return "Invalid", "invalid"

    label = str(label)

    if label == "A-is-clearly-better":
        return "A", "clear"

    if label == "A-is-slightly-better":
        return "A", "slight"

    if label == "B-is-clearly-better":
        return "B", "clear"

    if label == "B-is-slightly-better":
        return "B", "slight"

    if label == "Tie":
        return "Tie", "tie"

    return "Invalid", "invalid"
def collect_annotations(example):
    """
    合并 normal_worker_annotations 和 expert_worker_annotations。
    """

    annotations = []

    normal = example.get("normal_worker_annotations", None)
    expert = example.get("expert_worker_annotations", None)

    if normal is not None:
        annotations.extend(normal)

    if expert is not None:
        annotations.extend(expert)

    return annotations
print("Loading MultiPref...")

try:
    ds = load_dataset("allenai/multipref", "default")
except Exception:
    ds = load_dataset("allenai/multipref")

split_name = list(ds.keys())[0]
data = ds[split_name]

print("Split:", split_name)
print("Number of examples:", len(data))
rows = []

for ex in data:
    annotations = collect_annotations(ex)

    directions = []
    strengths = []
    raw_labels = []
    evaluator_names = []

    for ann in annotations:
        raw = ann.get("overall_pref", None)
        direction, strength = parse_overall_pref(raw)

        raw_labels.append(raw)
        directions.append(direction)
        strengths.append(strength)
        evaluator_names.append(ann.get("evaluator", ""))

    direction_counter = Counter(directions)
    strength_counter = Counter(strengths)
    row = {
        "comparison_id": ex.get("comparison_id"),
        "prompt_id": ex.get("prompt_id"),
        "prompt": ex.get("text"),
        "model_a": ex.get("model_a"),
        "model_b": ex.get("model_b"),
        "response_a": ex.get("completion_a"),
        "response_b": ex.get("completion_b"),
        "source": ex.get("source"),
        "category": ex.get("category"),
        "subject_study": ex.get("subject_study"),

        "raw_overall_labels": str(raw_labels),
        "parsed_directions": str(directions),
        "parsed_strengths": str(strengths),
        "evaluator_names": str(evaluator_names),

        "n_A": direction_counter["A"],
        "n_B": direction_counter["B"],
        "n_Tie": direction_counter["Tie"],
        "n_Invalid": direction_counter["Invalid"],

        "n_clear": strength_counter["clear"],
        "n_slight": strength_counter["slight"],
        "n_tie_strength": strength_counter["tie"],
        "n_invalid_strength": strength_counter["invalid"],

        "n_annotations": len(annotations),
        "n_valid": direction_counter["A"] + direction_counter["B"] + direction_counter["Tie"],
    }
    rows.append(row)
df = pd.DataFrame(rows)

out_path = "data_processed/multipref_parsed_labels.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("Saved parsed labels to:", out_path)
print("Rows:", len(df))

print("\nFirst 5 rows:")
print(df[
    [
        "comparison_id",
        "n_A",
        "n_B",
        "n_Tie",
        "n_Invalid",
        "n_clear",
        "n_slight",
        "n_valid",
        "raw_overall_labels",
        "parsed_directions",
        "parsed_strengths",
    ]
].head())
summary = {
    "total_examples": len(df),
    "total_annotations": int(df["n_annotations"].sum()),
    "total_A_votes": int(df["n_A"].sum()),
    "total_B_votes": int(df["n_B"].sum()),
    "total_Tie_votes": int(df["n_Tie"].sum()),
    "total_Invalid_votes": int(df["n_Invalid"].sum()),
    "total_clear_votes": int(df["n_clear"].sum()),
    "total_slight_votes": int(df["n_slight"].sum()),
    "examples_with_invalid": int((df["n_Invalid"] > 0).sum()),
    "avg_annotations_per_example": float(df["n_annotations"].mean()),
}

summary_df = pd.DataFrame([summary])

summary_path = "results/step4_label_parsing_summary.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\nSummary:")
print(summary_df.T)
print("Saved summary to:", summary_path)