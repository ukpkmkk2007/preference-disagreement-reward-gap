# -*- coding: utf-8 -*-
"""
Step 21: Audit the full eligible MultiPref pool by original category and class.

Purpose:
Before redesigning the formal sample, check how many eligible diverging and
high-agreement pairs actually exist in each original MultiPref category.

This script does NOT create a new formal sample and does NOT modify existing files.

Output:
1. results/full_eligible_pool_by_category.csv
2. results/full_eligible_pool_audit.csv
"""

import os
from pathlib import Path

import pandas as pd


# =========================
# 0. Fixed configuration
# =========================

PROJECT_DIR = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

INPUT_CANDIDATES = [
    PROJECT_DIR / "data_processed" / "multipref_with_diverging_labels.csv",
    PROJECT_DIR / "scripts" / "data_processed" / "multipref_with_diverging_labels.csv",
]

CATEGORY_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "full_eligible_pool_by_category.csv"
)

AUDIT_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "full_eligible_pool_audit.csv"
)

MAX_TOTAL_CHARS = 4000
MIN_PER_CLASS = 20


# =========================
# 1. Setup and locate input
# =========================

if not PROJECT_DIR.exists():
    raise FileNotFoundError(
        f"Project directory does not exist:\n{PROJECT_DIR}"
    )

os.chdir(PROJECT_DIR)
(PROJECT_DIR / "results").mkdir(exist_ok=True)

input_path = None

for candidate in INPUT_CANDIDATES:
    if candidate.exists():
        input_path = candidate
        break

if input_path is None:
    searched = "\n- ".join(str(p) for p in INPUT_CANDIDATES)
    raise FileNotFoundError(
        "Cannot find multipref_with_diverging_labels.csv.\n"
        "Searched:\n- "
        + searched
    )

print("Current working directory:")
print(Path.cwd())

print("\nUsing input file:")
print(input_path)


# =========================
# 2. Load and validate data
# =========================

df = pd.read_csv(input_path)

required_columns = [
    "comparison_id",
    "category",
    "prompt",
    "response_a",
    "response_b",
    "n_valid",
    "diverging_paper_like",
    "high_agreement_pref",
]

missing = [
    col for col in required_columns if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n- "
        + "\n- ".join(missing)
    )

print("\nLoaded source data.")
print("Rows:", len(df))


# =========================
# 3. Normalize booleans
# =========================

def to_bool_series(series):
    if series.dtype == bool:
        return series

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    if normalized.isna().any():
        bad_values = sorted(
            series[normalized.isna()]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Cannot parse boolean values in column "
            f"'{series.name}': {bad_values}"
        )

    return normalized.astype(bool)


for col in [
    "diverging_paper_like",
    "high_agreement_pref",
]:
    df[col] = to_bool_series(df[col])


# =========================
# 4. Reproduce the shared eligibility filter
# =========================

df["total_text_chars"] = (
    df["prompt"].fillna("").astype(str).str.len()
    + df["response_a"].fillna("").astype(str).str.len()
    + df["response_b"].fillna("").astype(str).str.len()
)

eligible_df = df[
    (df["n_valid"] > 0)
    & (df["total_text_chars"] <= MAX_TOTAL_CHARS)
].copy()

primary_df = eligible_df[
    eligible_df["diverging_paper_like"]
    | eligible_df["high_agreement_pref"]
].copy()

primary_df["diverging_id_label"] = (
    primary_df["diverging_paper_like"]
    .astype(int)
)


# =========================
# 5. Count category x class
# =========================

category_table = pd.crosstab(
    primary_df["category"],
    primary_df["diverging_id_label"],
).rename(
    columns={
        0: "n_high_agreement",
        1: "n_diverging",
    }
)

for col in [
    "n_high_agreement",
    "n_diverging",
]:
    if col not in category_table.columns:
        category_table[col] = 0

category_table = category_table[
    [
        "n_diverging",
        "n_high_agreement",
    ]
].copy()

category_table["n_total_primary"] = (
    category_table["n_diverging"]
    + category_table["n_high_agreement"]
)

category_table["max_balanced_pairs_per_class"] = category_table[
    [
        "n_diverging",
        "n_high_agreement",
    ]
].min(axis=1)

category_table["can_support_20_per_class"] = (
    category_table["max_balanced_pairs_per_class"]
    >= MIN_PER_CLASS
)

category_table["can_support_40_per_class"] = (
    category_table["max_balanced_pairs_per_class"]
    >= 40
)

category_table["can_support_50_per_class"] = (
    category_table["max_balanced_pairs_per_class"]
    >= 50
)

category_table = (
    category_table
    .reset_index()
    .sort_values(
        [
            "can_support_20_per_class",
            "max_balanced_pairs_per_class",
            "n_total_primary",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
)

category_table.to_csv(
    CATEGORY_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 6. Save audit summary
# =========================

audit = {
    "source_rows": len(df),
    "eligible_rows_after_length_filter": len(eligible_df),
    "primary_binary_rows": len(primary_df),
    "n_diverging": int(
        primary_df["diverging_id_label"].sum()
    ),
    "n_high_agreement": int(
        (primary_df["diverging_id_label"] == 0).sum()
    ),
    "max_total_chars": MAX_TOTAL_CHARS,
    "min_per_class_reference": MIN_PER_CLASS,
    "n_categories_supporting_20_per_class": int(
        category_table["can_support_20_per_class"].sum()
    ),
    "n_categories_supporting_40_per_class": int(
        category_table["can_support_40_per_class"].sum()
    ),
    "n_categories_supporting_50_per_class": int(
        category_table["can_support_50_per_class"].sum()
    ),
}

pd.DataFrame([audit]).to_csv(
    AUDIT_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 7. Print results
# =========================

print("\nFull eligible pool by category:")
print(category_table.to_string(index=False))

print("\nSummary:")
for key, value in audit.items():
    print(f"{key}: {value}")

print("\nSaved:")
print(CATEGORY_OUTPUT_PATH)
print(AUDIT_OUTPUT_PATH)

print("\nIMPORTANT:")
print(
    "This script only audits the source pool. "
    "It does not replace the existing 400-pair sample."
)
