# -*- coding: utf-8 -*-
"""
Step 23: Validate and freeze the completed annotation table.

Inputs
------
1. data_annotation/formal_sample_v1_stratified_annotation.csv
2. data_processed/formal_sample_v1_stratified_master.csv

Outputs
-------
1. data_processed/formal_sample_v1_stratified_annotation_frozen.csv
2. results/formal_sample_v1_annotation_freeze_manifest.csv
3. results/formal_sample_v1_annotation_label_counts.csv
"""

import hashlib
import os
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

ANNOTATION_PATH = (
    PROJECT_DIR
    / "data_annotation"
    / "formal_sample_v1_stratified_annotation.csv"
)

MASTER_PATH = (
    PROJECT_DIR
    / "data_processed"
    / "formal_sample_v1_stratified_master.csv"
)

FROZEN_OUTPUT_PATH = (
    PROJECT_DIR
    / "data_processed"
    / "formal_sample_v1_stratified_annotation_frozen.csv"
)

MANIFEST_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_annotation_freeze_manifest.csv"
)

LABEL_COUNTS_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_annotation_label_counts.csv"
)

EXPECTED_ROWS = 400
OVERWRITE = False

ALLOWED_TASK_SCENARIOS = {
    "factual_qa",
    "technical_or_coding",
    "medical_or_health",
    "safety_or_sensitive",
    "general_advice_or_chat",
    "writing_or_creative",
    "reasoning_or_math",
    "summarization_or_transformation",
    "brainstorming_or_planning",
    "other",
}

ALLOWED_DISAGREEMENT_SOURCES = {
    "",
    "verbosity_or_concision",
    "format_or_style",
    "task_ambiguity",
    "expertise_or_complexity",
    "refusal_strategy",
    "factuality",
    "instruction_following",
    "value_or_subjective_preference",
    "clear_quality_difference",
    "other",
}

FORBIDDEN_ANNOTATION_COLUMNS = {
    "category",
    "sampling_stratum",
    "diverging_id_label",
    "diverging_paper_like",
    "high_agreement_pref",
    "n_A",
    "n_B",
    "n_Tie",
    "n_valid",
    "majority_label",
    "score_a",
    "score_b",
    "reward_a",
    "reward_b",
    "reward_gap",
}

if not PROJECT_DIR.exists():
    raise FileNotFoundError(
        f"Project directory does not exist:\n{PROJECT_DIR}"
    )

os.chdir(PROJECT_DIR)

(PROJECT_DIR / "data_processed").mkdir(exist_ok=True)
(PROJECT_DIR / "results").mkdir(exist_ok=True)

print("Current working directory:")
print(Path.cwd())

output_paths = [
    FROZEN_OUTPUT_PATH,
    MANIFEST_OUTPUT_PATH,
    LABEL_COUNTS_OUTPUT_PATH,
]

existing_outputs = [
    str(path) for path in output_paths if path.exists()
]

if existing_outputs and not OVERWRITE:
    raise FileExistsError(
        "Frozen annotation outputs already exist.\n"
        "Existing files:\n- "
        + "\n- ".join(existing_outputs)
        + "\n\nArchive the old files first if recreation is intentional."
    )

if not ANNOTATION_PATH.exists():
    raise FileNotFoundError(
        "Cannot find the completed annotation file:\n"
        f"{ANNOTATION_PATH}"
    )

if not MASTER_PATH.exists():
    raise FileNotFoundError(
        "Cannot find the fixed master sample:\n"
        f"{MASTER_PATH}"
    )

annotation_df = pd.read_csv(ANNOTATION_PATH)
master_df = pd.read_csv(MASTER_PATH)

print("\nLoaded annotation table.")
print("Rows:", len(annotation_df))

print("\nLoaded master table.")
print("Rows:", len(master_df))

required_annotation_columns = [
    "annotation_id",
    "comparison_id",
    "prompt",
    "response_a",
    "response_b",
    "task_scenario",
    "disagreement_source_1",
    "disagreement_source_2",
    "disagreement_source_3",
    "annotation_notes",
    "annotation_status",
]

missing_annotation_columns = [
    col
    for col in required_annotation_columns
    if col not in annotation_df.columns
]

if missing_annotation_columns:
    raise ValueError(
        "Annotation file is missing required columns:\n- "
        + "\n- ".join(missing_annotation_columns)
    )

forbidden_present = sorted(
    FORBIDDEN_ANNOTATION_COLUMNS.intersection(
        set(annotation_df.columns)
    )
)

if forbidden_present:
    raise ValueError(
        "Annotation file contains forbidden outcome or sampling columns:\n- "
        + "\n- ".join(forbidden_present)
    )

required_master_columns = [
    "annotation_id",
    "comparison_id",
    "prompt",
    "response_a",
    "response_b",
]

missing_master_columns = [
    col
    for col in required_master_columns
    if col not in master_df.columns
]

if missing_master_columns:
    raise ValueError(
        "Master file is missing required columns:\n- "
        + "\n- ".join(missing_master_columns)
    )

text_columns = [
    "task_scenario",
    "disagreement_source_1",
    "disagreement_source_2",
    "disagreement_source_3",
    "annotation_notes",
    "annotation_status",
]

for col in text_columns:
    annotation_df[col] = (
        annotation_df[col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

if len(annotation_df) != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} annotation rows, "
        f"but found {len(annotation_df)}."
    )

if len(master_df) != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} master rows, "
        f"but found {len(master_df)}."
    )

if annotation_df["annotation_id"].duplicated().any():
    raise ValueError("Duplicate annotation_id values found.")

if annotation_df["comparison_id"].duplicated().any():
    raise ValueError("Duplicate comparison_id values found.")

annotation_pairs = set(
    zip(
        annotation_df["annotation_id"].astype(str),
        annotation_df["comparison_id"].astype(str),
    )
)

master_pairs = set(
    zip(
        master_df["annotation_id"].astype(str),
        master_df["comparison_id"].astype(str),
    )
)

if annotation_pairs != master_pairs:
    raise ValueError(
        "Annotation IDs do not exactly match the fixed master sample."
    )

not_completed = annotation_df[
    annotation_df["annotation_status"] != "completed"
]

if len(not_completed) > 0:
    raise ValueError(
        "Some rows are not marked completed:\n"
        + not_completed[
            ["annotation_id", "annotation_status"]
        ]
        .head(20)
        .to_string(index=False)
    )

blank_task = annotation_df[
    annotation_df["task_scenario"] == ""
]

if len(blank_task) > 0:
    raise ValueError(
        "Some rows have a blank task_scenario:\n"
        + blank_task["annotation_id"]
        .head(20)
        .to_string(index=False)
    )

invalid_tasks = sorted(
    set(annotation_df["task_scenario"])
    - ALLOWED_TASK_SCENARIOS
)

if invalid_tasks:
    raise ValueError(
        "Invalid task_scenario labels:\n- "
        + "\n- ".join(invalid_tasks)
    )

source_columns = [
    "disagreement_source_1",
    "disagreement_source_2",
    "disagreement_source_3",
]

all_source_values = set()

for col in source_columns:
    all_source_values.update(
        annotation_df[col].unique().tolist()
    )

invalid_sources = sorted(
    all_source_values
    - ALLOWED_DISAGREEMENT_SOURCES
)

if invalid_sources:
    raise ValueError(
        "Invalid disagreement-source labels:\n- "
        + "\n- ".join(invalid_sources)
    )

ordering_errors = []
duplicate_source_errors = []

for _, row in annotation_df.iterrows():
    sources = [
        row["disagreement_source_1"],
        row["disagreement_source_2"],
        row["disagreement_source_3"],
    ]

    if sources[1] and not sources[0]:
        ordering_errors.append(row["annotation_id"])

    if sources[2] and not sources[1]:
        ordering_errors.append(row["annotation_id"])

    nonempty_sources = [
        value for value in sources if value
    ]

    if len(nonempty_sources) != len(set(nonempty_sources)):
        duplicate_source_errors.append(
            row["annotation_id"]
        )

if ordering_errors:
    raise ValueError(
        "Disagreement-source fields are not left-packed for:\n- "
        + "\n- ".join(sorted(set(ordering_errors))[:20])
    )

if duplicate_source_errors:
    raise ValueError(
        "Duplicate disagreement-source labels found within rows:\n- "
        + "\n- ".join(sorted(set(duplicate_source_errors))[:20])
    )

master_text = master_df[
    [
        "annotation_id",
        "prompt",
        "response_a",
        "response_b",
    ]
].copy()

annotation_text = annotation_df[
    [
        "annotation_id",
        "prompt",
        "response_a",
        "response_b",
    ]
].copy()

merged_text = annotation_text.merge(
    master_text,
    on="annotation_id",
    how="outer",
    suffixes=("_annotation", "_master"),
    validate="one_to_one",
)

text_mismatch_mask = (
    merged_text["prompt_annotation"].fillna("").astype(str)
    != merged_text["prompt_master"].fillna("").astype(str)
) | (
    merged_text["response_a_annotation"].fillna("").astype(str)
    != merged_text["response_a_master"].fillna("").astype(str)
) | (
    merged_text["response_b_annotation"].fillna("").astype(str)
    != merged_text["response_b_master"].fillna("").astype(str)
)

if text_mismatch_mask.any():
    mismatch_ids = (
        merged_text.loc[
            text_mismatch_mask,
            "annotation_id",
        ]
        .astype(str)
        .tolist()
    )
    raise ValueError(
        "Prompt or response text differs from the fixed master sample for:\n- "
        + "\n- ".join(mismatch_ids[:20])
    )

annotation_df = annotation_df[
    required_annotation_columns
].copy()

annotation_df.to_csv(
    FROZEN_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()

frozen_sha256 = sha256_file(
    FROZEN_OUTPUT_PATH
)

task_counts = (
    annotation_df["task_scenario"]
    .value_counts(dropna=False)
    .rename_axis("label")
    .reset_index(name="count")
)

task_counts.insert(
    0,
    "label_type",
    "task_scenario",
)

all_sources = pd.concat(
    [
        annotation_df[col]
        for col in source_columns
    ],
    ignore_index=True,
)

all_sources = all_sources[
    all_sources != ""
]

source_counts = (
    all_sources
    .value_counts(dropna=False)
    .rename_axis("label")
    .reset_index(name="count")
)

source_counts.insert(
    0,
    "label_type",
    "disagreement_source",
)

label_counts = pd.concat(
    [
        task_counts,
        source_counts,
    ],
    ignore_index=True,
)

label_counts.to_csv(
    LABEL_COUNTS_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

source_count_per_row = (
    annotation_df[source_columns]
    .ne("")
    .sum(axis=1)
)

manifest = pd.DataFrame(
    [
        {
            "annotation_version": "formal_sample_v1_stratified_annotation_frozen",
            "source_annotation_file": str(ANNOTATION_PATH),
            "master_sample_file": str(MASTER_PATH),
            "frozen_output_file": str(FROZEN_OUTPUT_PATH),
            "frozen_sha256": frozen_sha256,
            "n_rows": len(annotation_df),
            "n_completed": int(
                (
                    annotation_df["annotation_status"]
                    == "completed"
                ).sum()
            ),
            "n_unique_annotation_ids": int(
                annotation_df["annotation_id"].nunique()
            ),
            "n_unique_comparison_ids": int(
                annotation_df["comparison_id"].nunique()
            ),
            "n_task_scenarios": int(
                annotation_df["task_scenario"].nunique()
            ),
            "n_rows_zero_sources": int(
                (source_count_per_row == 0).sum()
            ),
            "n_rows_one_source": int(
                (source_count_per_row == 1).sum()
            ),
            "n_rows_two_sources": int(
                (source_count_per_row == 2).sum()
            ),
            "n_rows_three_sources": int(
                (source_count_per_row == 3).sum()
            ),
            "outcome_columns_present": False,
            "ids_match_master": True,
            "texts_match_master": True,
            "annotation_frozen_before_reward_scoring": True,
        }
    ]
)

manifest.to_csv(
    MANIFEST_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\n==============================")
print("Annotation validation passed.")
print("Rows:", len(annotation_df))
print(
    "Completed:",
    int(
        (
            annotation_df["annotation_status"]
            == "completed"
        ).sum()
    ),
)
print(
    "Unique annotation IDs:",
    annotation_df["annotation_id"].nunique(),
)
print(
    "Unique comparison IDs:",
    annotation_df["comparison_id"].nunique(),
)
print(
    "Task scenarios:",
    annotation_df["task_scenario"].nunique(),
)
print(
    "Rows with zero disagreement sources:",
    int((source_count_per_row == 0).sum()),
)
print(
    "Rows with one disagreement source:",
    int((source_count_per_row == 1).sum()),
)
print(
    "Rows with two disagreement sources:",
    int((source_count_per_row == 2).sum()),
)
print(
    "Rows with three disagreement sources:",
    int((source_count_per_row == 3).sum()),
)
print("IDs match master: True")
print("Texts match master: True")
print("Forbidden outcome columns present: False")
print("==============================")

print("\nFrozen SHA-256:")
print(frozen_sha256)

print("\nSaved:")
print(FROZEN_OUTPUT_PATH)
print(MANIFEST_OUTPUT_PATH)
print(LABEL_COUNTS_OUTPUT_PATH)

print("\nIMPORTANT:")
print(
    "Do not edit the frozen annotation file after reward scoring begins."
)
