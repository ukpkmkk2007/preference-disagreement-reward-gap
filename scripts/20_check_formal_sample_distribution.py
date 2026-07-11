# -*- coding: utf-8 -*-
"""
Step 20: Check whether the fixed formal sample supports per-scenario analysis.

This script does NOT change the formal sample.
It only audits the distribution of the existing 400 pairs.

Outputs:
1. results/formal_sample_v1_category_by_class.csv
2. results/formal_sample_v1_category_audit.csv
"""

import os
from pathlib import Path

import pandas as pd


# =========================
# 0. Project paths
# =========================

PROJECT_DIR = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

INPUT_PATH = (
    PROJECT_DIR
    / "data_processed"
    / "formal_sample_v1_master.csv"
)

CROSSTAB_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_category_by_class.csv"
)

AUDIT_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_category_audit.csv"
)

MIN_PER_CLASS = 20


# =========================
# 1. Setup
# =========================

if not PROJECT_DIR.exists():
    raise FileNotFoundError(
        f"Project directory does not exist:\n{PROJECT_DIR}"
    )

os.chdir(PROJECT_DIR)

(PROJECT_DIR / "results").mkdir(exist_ok=True)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find:\n{INPUT_PATH}\n\n"
        "Run scripts/19_prepare_formal_sample_v1.py first."
    )

print("Current working directory:")
print(Path.cwd())


# =========================
# 2. Load fixed sample
# =========================

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "comparison_id",
    "category",
    "diverging_id_label",
]

missing = [
    col for col in required_columns if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n- "
        + "\n- ".join(missing)
    )

print("\nLoaded formal sample.")
print("Rows:", len(df))

if len(df) != 400:
    print(
        "\nWARNING: Expected 400 rows, "
        f"but found {len(df)} rows."
    )


# =========================
# 3. Category x class table
# =========================

category_by_class = pd.crosstab(
    df["category"],
    df["diverging_id_label"],
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
    if col not in category_by_class.columns:
        category_by_class[col] = 0

category_by_class = category_by_class[
    [
        "n_diverging",
        "n_high_agreement",
    ]
].copy()

category_by_class["n_total"] = (
    category_by_class["n_diverging"]
    + category_by_class["n_high_agreement"]
)

category_by_class["eligible_for_primary_auroc"] = (
    (category_by_class["n_diverging"] >= MIN_PER_CLASS)
    & (
        category_by_class["n_high_agreement"]
        >= MIN_PER_CLASS
    )
)

category_by_class = (
    category_by_class
    .reset_index()
    .sort_values(
        [
            "eligible_for_primary_auroc",
            "n_total",
        ],
        ascending=[
            False,
            False,
        ],
    )
)

category_by_class.to_csv(
    CROSSTAB_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 4. Audit summary
# =========================

eligible_categories = category_by_class[
    category_by_class[
        "eligible_for_primary_auroc"
    ]
]

audit = {
    "n_total_rows": len(df),
    "n_diverging": int(
        (df["diverging_id_label"] == 1).sum()
    ),
    "n_high_agreement": int(
        (df["diverging_id_label"] == 0).sum()
    ),
    "n_categories_total": int(
        df["category"].nunique(dropna=False)
    ),
    "min_per_class_required": MIN_PER_CLASS,
    "n_categories_eligible_for_primary_auroc": (
        len(eligible_categories)
    ),
    "eligible_categories": "; ".join(
        eligible_categories["category"]
        .astype(str)
        .tolist()
    ),
}

pd.DataFrame([audit]).to_csv(
    AUDIT_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 5. Print results
# =========================

print("\nCategory distribution by class:")
print(category_by_class.to_string(index=False))

print("\nPrimary AUROC threshold:")
print(
    f"At least {MIN_PER_CLASS} diverging "
    f"and {MIN_PER_CLASS} high-agreement pairs "
    "within a category."
)

print("\nEligible categories:")
if len(eligible_categories) == 0:
    print("None")
else:
    print(
        eligible_categories[
            [
                "category",
                "n_diverging",
                "n_high_agreement",
            ]
        ].to_string(index=False)
    )

print("\nSaved:")
print(CROSSTAB_OUTPUT_PATH)
print(AUDIT_OUTPUT_PATH)

print("\nIMPORTANT:")
print(
    "Do not recreate or resample the formal sample yet. "
    "First inspect this distribution."
)
