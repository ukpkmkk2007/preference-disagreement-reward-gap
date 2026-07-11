# -*- coding: utf-8 -*-
"""
Step 22: Create a stratified formal sample for scenario analysis.

Why this script exists:
The first random 400-pair sample was dominated by Open QA.
This script creates a new, separate 400-pair sample with fixed quotas
across six original MultiPref categories, so that each category has
enough diverging and high-agreement examples for subgroup AUROC.

This script does NOT overwrite the earlier random sample.

Outputs:
1. data_processed/formal_sample_v1_stratified_master.csv
2. data_annotation/formal_sample_v1_stratified_annotation.csv
3. data_processed/formal_sample_v1_stratified_ids.csv
4. results/formal_sample_v1_stratified_manifest.csv
5. results/formal_sample_v1_stratified_category_counts.csv
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

MASTER_OUTPUT_PATH = (
    PROJECT_DIR
    / "data_processed"
    / "formal_sample_v1_stratified_master.csv"
)

ANNOTATION_OUTPUT_PATH = (
    PROJECT_DIR
    / "data_annotation"
    / "formal_sample_v1_stratified_annotation.csv"
)

IDS_OUTPUT_PATH = (
    PROJECT_DIR
    / "data_processed"
    / "formal_sample_v1_stratified_ids.csv"
)

MANIFEST_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_stratified_manifest.csv"
)

COUNTS_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_stratified_category_counts.csv"
)

MAX_TOTAL_CHARS = 4000
RANDOM_STATE = 42

# Number sampled PER CLASS within each original MultiPref category.
# Total per class = 40 + 35 + 35 + 35 + 35 + 20 = 200.
# Total sample = 400.
CATEGORY_QUOTAS_PER_CLASS = {
    "Open QA": 40,
    "Generation": 35,
    "Coding": 35,
    "Chat": 35,
    "Brainstorm": 35,
    "Closed QA": 20,
}

OVERWRITE = False


# =========================
# 1. Setup
# =========================

if not PROJECT_DIR.exists():
    raise FileNotFoundError(
        f"Project directory does not exist:\n{PROJECT_DIR}"
    )

os.chdir(PROJECT_DIR)

(PROJECT_DIR / "data_processed").mkdir(exist_ok=True)
(PROJECT_DIR / "data_annotation").mkdir(exist_ok=True)
(PROJECT_DIR / "results").mkdir(exist_ok=True)

print("Current working directory:")
print(Path.cwd())


# =========================
# 2. Locate input
# =========================

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

print("\nUsing input file:")
print(input_path)


# =========================
# 3. Protect existing outputs
# =========================

output_paths = [
    MASTER_OUTPUT_PATH,
    ANNOTATION_OUTPUT_PATH,
    IDS_OUTPUT_PATH,
    MANIFEST_OUTPUT_PATH,
    COUNTS_OUTPUT_PATH,
]

existing_outputs = [
    str(path) for path in output_paths if path.exists()
]

if existing_outputs and not OVERWRITE:
    raise FileExistsError(
        "Stratified formal-sample output already exists.\n"
        "Existing files:\n- "
        + "\n- ".join(existing_outputs)
        + "\n\nDo not silently replace a fixed formal sample. "
          "Archive the old files first if recreation is intentional."
    )


# =========================
# 4. Load and validate data
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
# 5. Normalize booleans
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
# 6. Apply the shared eligibility filter
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

overlap_mask = (
    eligible_df["diverging_paper_like"]
    & eligible_df["high_agreement_pref"]
)

if overlap_mask.any():
    raise ValueError(
        "Some rows are simultaneously diverging and high-agreement."
    )


# =========================
# 7. Draw fixed strata
# =========================

sample_parts = []
availability_records = []

for category_index, (category, quota) in enumerate(
    CATEGORY_QUOTAS_PER_CLASS.items()
):
    category_df = eligible_df[
        eligible_df["category"] == category
    ].copy()

    diverging_pool = category_df[
        category_df["diverging_paper_like"]
    ].copy()

    high_agreement_pool = category_df[
        category_df["high_agreement_pref"]
    ].copy()

    availability_records.append({
        "category": category,
        "quota_per_class": quota,
        "available_diverging": len(diverging_pool),
        "available_high_agreement": len(high_agreement_pool),
    })

    if len(diverging_pool) < quota:
        raise ValueError(
            f"{category}: need {quota} diverging pairs, "
            f"but only {len(diverging_pool)} are eligible."
        )

    if len(high_agreement_pool) < quota:
        raise ValueError(
            f"{category}: need {quota} high-agreement pairs, "
            f"but only {len(high_agreement_pool)} are eligible."
        )

    # Distinct deterministic seeds for each stratum and class.
    diverging_seed = RANDOM_STATE + category_index * 10 + 1
    high_agreement_seed = RANDOM_STATE + category_index * 10 + 2

    sample_diverging = diverging_pool.sample(
        n=quota,
        random_state=diverging_seed,
        replace=False,
    ).copy()

    sample_high_agreement = high_agreement_pool.sample(
        n=quota,
        random_state=high_agreement_seed,
        replace=False,
    ).copy()

    sample_diverging["diverging_id_label"] = 1
    sample_high_agreement["diverging_id_label"] = 0

    sample_diverging["sampling_stratum"] = category
    sample_high_agreement["sampling_stratum"] = category

    sample_parts.extend([
        sample_diverging,
        sample_high_agreement,
    ])


# =========================
# 8. Combine and shuffle
# =========================

sample_df = pd.concat(
    sample_parts,
    axis=0,
    ignore_index=True,
)

sample_df = sample_df.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

sample_df.insert(
    0,
    "annotation_id",
    [
        f"SV1_{i:04d}"
        for i in range(1, len(sample_df) + 1)
    ],
)

expected_per_class = sum(
    CATEGORY_QUOTAS_PER_CLASS.values()
)
expected_total = expected_per_class * 2

if len(sample_df) != expected_total:
    raise AssertionError(
        f"Expected {expected_total} rows, got {len(sample_df)}."
    )

if sample_df["comparison_id"].duplicated().any():
    duplicate_ids = (
        sample_df.loc[
            sample_df["comparison_id"].duplicated(keep=False),
            "comparison_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    raise ValueError(
        "Duplicate comparison IDs found:\n- "
        + "\n- ".join(duplicate_ids[:10])
    )


# =========================
# 9. Verify stratum counts
# =========================

category_counts = (
    sample_df
    .groupby(
        [
            "sampling_stratum",
            "diverging_id_label",
        ]
    )
    .size()
    .unstack(fill_value=0)
    .rename(
        columns={
            0: "n_high_agreement",
            1: "n_diverging",
        }
    )
    .reset_index()
    .rename(
        columns={
            "sampling_stratum": "category",
        }
    )
)

for category, quota in CATEGORY_QUOTAS_PER_CLASS.items():
    row = category_counts[
        category_counts["category"] == category
    ]

    if len(row) != 1:
        raise AssertionError(
            f"Missing count row for category: {category}"
        )

    n_div = int(row.iloc[0]["n_diverging"])
    n_high = int(row.iloc[0]["n_high_agreement"])

    if n_div != quota or n_high != quota:
        raise AssertionError(
            f"{category}: expected {quota}/{quota}, "
            f"got {n_div}/{n_high}."
        )

category_counts["n_total"] = (
    category_counts["n_diverging"]
    + category_counts["n_high_agreement"]
)

category_counts.to_csv(
    COUNTS_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 10. Save internal master
# =========================

sample_df.to_csv(
    MASTER_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 11. Save blinded annotation table
# =========================

annotation_df = sample_df[
    [
        "annotation_id",
        "comparison_id",
        "prompt",
        "response_a",
        "response_b",
    ]
].copy()

annotation_df["task_scenario"] = ""
annotation_df["disagreement_source_1"] = ""
annotation_df["disagreement_source_2"] = ""
annotation_df["disagreement_source_3"] = ""
annotation_df["annotation_notes"] = ""
annotation_df["annotation_status"] = "unlabeled"

annotation_df.to_csv(
    ANNOTATION_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 12. Save fixed IDs
# =========================

ids_df = sample_df[
    [
        "annotation_id",
        "comparison_id",
        "sampling_stratum",
        "diverging_id_label",
    ]
].copy()

ids_df.to_csv(
    IDS_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 13. Save manifest
# =========================

manifest_rows = []

for record in availability_records:
    category = record["category"]
    quota = record["quota_per_class"]

    manifest_rows.append({
        "sample_name": "formal_sample_v1_stratified",
        "input_file": str(input_path),
        "category": category,
        "quota_diverging": quota,
        "quota_high_agreement": quota,
        "available_diverging": record["available_diverging"],
        "available_high_agreement": record["available_high_agreement"],
        "max_total_chars": MAX_TOTAL_CHARS,
        "base_random_state": RANDOM_STATE,
        "annotation_is_blinded_to_human_labels": True,
    })

pd.DataFrame(manifest_rows).to_csv(
    MANIFEST_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 14. Final report
# =========================

print("\nStratified category counts:")
print(category_counts.to_string(index=False))

print("\n==============================")
print("Stratified formal sample v1 created.")
print("Total rows:", len(sample_df))
print(
    "Diverging:",
    int((sample_df["diverging_id_label"] == 1).sum()),
)
print(
    "High-agreement:",
    int((sample_df["diverging_id_label"] == 0).sum()),
)
print(
    "Maximum total characters:",
    int(sample_df["total_text_chars"].max()),
)
print(
    "Annotation table contains human labels:",
    any(
        col in annotation_df.columns
        for col in [
            "diverging_id_label",
            "diverging_paper_like",
            "high_agreement_pref",
            "n_A",
            "n_B",
            "n_Tie",
            "majority_label",
            "category",
            "sampling_stratum",
        ]
    ),
)
print("==============================")

print("\nSaved:")
print(MASTER_OUTPUT_PATH)
print(ANNOTATION_OUTPUT_PATH)
print(IDS_OUTPUT_PATH)
print(MANIFEST_OUTPUT_PATH)
print(COUNTS_OUTPUT_PATH)

print("\nIMPORTANT:")
print(
    "Use the files containing 'stratified' for subsequent "
    "scenario annotation and reward scoring."
)
