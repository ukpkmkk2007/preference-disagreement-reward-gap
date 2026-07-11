# -*- coding: utf-8 -*-
"""
Step 13: Prepare the fixed formal sample and a blinded annotation table for v1.

Outputs:
1. data_processed/formal_sample_v1_master.csv
   - Internal master table.
   - Contains human-disagreement labels and metadata.
   - Do not use this file while manually annotating task scenarios.

2. data_annotation/formal_sample_v1_annotation.csv
   - Blinded table for manual annotation.
   - Does not contain diverging/high-agreement labels, vote counts, reward scores, or reward gaps.

3. data_processed/formal_sample_v1_ids.csv
   - Fixed sample IDs for later reward-model scoring.

4. results/formal_sample_v1_manifest.csv
   - Sampling configuration and class counts.
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

INPUT_PATH = (
    PROJECT_DIR
    / "scripts"
    / "data_processed"
    / "multipref_with_diverging_labels.csv"
)
MASTER_OUTPUT_PATH = Path(
    "data_processed/formal_sample_v1_master.csv"
)

ANNOTATION_OUTPUT_PATH = Path(
    "data_annotation/formal_sample_v1_annotation.csv"
)

IDS_OUTPUT_PATH = Path(
    "data_processed/formal_sample_v1_ids.csv"
)

MANIFEST_OUTPUT_PATH = Path(
    "results/formal_sample_v1_manifest.csv"
)

N_PER_CLASS = 200
MAX_TOTAL_CHARS = 4000
RANDOM_STATE = 42

# Once the formal sample is created, do not silently overwrite it.
OVERWRITE = False


# =========================
# 1. Project setup
# =========================

if not PROJECT_DIR.exists():
    raise FileNotFoundError(
        f"Project directory does not exist:\n{PROJECT_DIR}"
    )

os.chdir(PROJECT_DIR)

Path("data_processed").mkdir(exist_ok=True)
Path("data_annotation").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

print("Current working directory:")
print(Path.cwd())


# =========================
# 2. Safety checks
# =========================

output_paths = [
    MASTER_OUTPUT_PATH,
    ANNOTATION_OUTPUT_PATH,
    IDS_OUTPUT_PATH,
    MANIFEST_OUTPUT_PATH,
]

existing_outputs = [
    str(path) for path in output_paths if path.exists()
]

if existing_outputs and not OVERWRITE:
    raise FileExistsError(
        "Formal-sample output already exists. "
        "The fixed sample should not be silently replaced.\n"
        "Existing files:\n- "
        + "\n- ".join(existing_outputs)
        + "\n\nIf you intentionally need to recreate the sample, "
          "first archive the old outputs and then set OVERWRITE = True."
    )

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find:\n{INPUT_PATH}\n\n"
        "Run scripts/03_define_diverging.py first."
    )


# =========================
# 3. Load data
# =========================

df = pd.read_csv(INPUT_PATH)

print("\nLoaded input data.")
print("Rows:", len(df))

required_columns = [
    "comparison_id",
    "prompt",
    "response_a",
    "response_b",
    "diverging_paper_like",
    "high_agreement_pref",
    "n_valid",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n- "
        + "\n- ".join(missing_columns)
    )


# =========================
# 4. Normalize boolean columns
# =========================

def to_bool_series(series):
    """
    Convert booleans or common string/integer representations to bool.
    """
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
# 5. Compute shared length filter
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

# The two primary classes should be mutually exclusive.
overlap_mask = (
    eligible_df["diverging_paper_like"]
    & eligible_df["high_agreement_pref"]
)

if overlap_mask.any():
    overlap_ids = (
        eligible_df.loc[overlap_mask, "comparison_id"]
        .astype(str)
        .head(10)
        .tolist()
    )
    raise ValueError(
        "Some rows are simultaneously labeled diverging and "
        "high-agreement. Example comparison IDs:\n- "
        + "\n- ".join(overlap_ids)
    )

diverging_pool = eligible_df[
    eligible_df["diverging_paper_like"]
].copy()

high_agreement_pool = eligible_df[
    eligible_df["high_agreement_pref"]
].copy()

print("\nEligible pools after filtering:")
print("All eligible:", len(eligible_df))
print("Diverging pool:", len(diverging_pool))
print("High-agreement pool:", len(high_agreement_pool))

if len(diverging_pool) < N_PER_CLASS:
    raise ValueError(
        f"Need {N_PER_CLASS} diverging pairs, "
        f"but only {len(diverging_pool)} are eligible."
    )

if len(high_agreement_pool) < N_PER_CLASS:
    raise ValueError(
        f"Need {N_PER_CLASS} high-agreement pairs, "
        f"but only {len(high_agreement_pool)} are eligible."
    )


# =========================
# 6. Draw the fixed balanced sample
# =========================

sample_diverging = diverging_pool.sample(
    n=N_PER_CLASS,
    random_state=RANDOM_STATE,
    replace=False,
).copy()

sample_high_agreement = high_agreement_pool.sample(
    n=N_PER_CLASS,
    random_state=RANDOM_STATE,
    replace=False,
).copy()

sample_diverging["diverging_id_label"] = 1
sample_high_agreement["diverging_id_label"] = 0

sample_df = pd.concat(
    [sample_diverging, sample_high_agreement],
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
        f"V1_{i:04d}"
        for i in range(1, len(sample_df) + 1)
    ],
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
        "Duplicate comparison IDs found in formal sample:\n- "
        + "\n- ".join(duplicate_ids[:10])
    )

expected_total = N_PER_CLASS * 2

if len(sample_df) != expected_total:
    raise AssertionError(
        f"Expected {expected_total} rows, got {len(sample_df)}."
    )

class_counts = (
    sample_df["diverging_id_label"]
    .value_counts()
    .sort_index()
    .to_dict()
)

if class_counts.get(0, 0) != N_PER_CLASS:
    raise AssertionError("High-agreement class count is incorrect.")

if class_counts.get(1, 0) != N_PER_CLASS:
    raise AssertionError("Diverging class count is incorrect.")


# =========================
# 7. Save internal master table
# =========================

sample_df.to_csv(
    MASTER_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved internal master table:")
print(MASTER_OUTPUT_PATH)


# =========================
# 8. Create blinded annotation table
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

# One primary task scenario.
annotation_df["task_scenario"] = ""

# Up to three disagreement-source labels.
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

print("\nSaved blinded annotation table:")
print(ANNOTATION_OUTPUT_PATH)


# =========================
# 9. Save fixed ID list
# =========================

ids_df = sample_df[
    [
        "annotation_id",
        "comparison_id",
        "diverging_id_label",
    ]
].copy()

ids_df.to_csv(
    IDS_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved fixed sample IDs:")
print(IDS_OUTPUT_PATH)


# =========================
# 10. Save sampling manifest
# =========================

manifest = {
    "input_file": str(INPUT_PATH),
    "random_state": RANDOM_STATE,
    "max_total_chars": MAX_TOTAL_CHARS,
    "n_per_class_target": N_PER_CLASS,
    "n_total_sampled": len(sample_df),
    "n_diverging_sampled": int(
        (sample_df["diverging_id_label"] == 1).sum()
    ),
    "n_high_agreement_sampled": int(
        (sample_df["diverging_id_label"] == 0).sum()
    ),
    "eligible_total": len(eligible_df),
    "eligible_diverging": len(diverging_pool),
    "eligible_high_agreement": len(high_agreement_pool),
    "annotation_is_blinded_to_human_labels": True,
}

pd.DataFrame([manifest]).to_csv(
    MANIFEST_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved manifest:")
print(MANIFEST_OUTPUT_PATH)


# =========================
# 11. Final verification
# =========================

print("\n==============================")
print("Formal sample v1 created.")
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
        ]
    ),
)
print("==============================")
