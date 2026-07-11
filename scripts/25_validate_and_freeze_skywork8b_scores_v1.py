# -*- coding: utf-8 -*-
"""
Step 25: Validate and freeze the completed Skywork 8B score table.

Inputs
------
1. results/formal_sample_v1_skywork8b_scores.csv
2. results/formal_sample_v1_skywork8b_scoring_manifest.csv
3. data_processed/formal_sample_v1_stratified_annotation_frozen.csv

Outputs
-------
1. data_processed/formal_sample_v1_skywork8b_scores_frozen.csv
2. results/formal_sample_v1_skywork8b_score_validation_manifest.csv
"""

import hashlib
import math
import os
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

SCORE_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_skywork8b_scores.csv"
)

SCORING_MANIFEST_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_skywork8b_scoring_manifest.csv"
)

FROZEN_ANNOTATION_PATH = (
    PROJECT_DIR
    / "data_processed"
    / "formal_sample_v1_stratified_annotation_frozen.csv"
)

FROZEN_SCORE_OUTPUT_PATH = (
    PROJECT_DIR
    / "data_processed"
    / "formal_sample_v1_skywork8b_scores_frozen.csv"
)

VALIDATION_MANIFEST_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_skywork8b_score_validation_manifest.csv"
)

EXPECTED_ROWS = 400
ABS_TOL = 1e-12
OVERWRITE = False

REQUIRED_SCORE_COLUMNS = [
    "annotation_id",
    "comparison_id",
    "skywork_score_a",
    "skywork_score_b",
    "score_gap_abs",
    "diverging_score",
]

FORBIDDEN_SCORE_COLUMNS = {
    "category",
    "sampling_stratum",
    "diverging_id_label",
    "diverging_paper_like",
    "high_agreement_pref",
    "n_A",
    "n_B",
    "n_Tie",
    "majority_label",
    "task_scenario",
    "disagreement_source_1",
    "disagreement_source_2",
    "disagreement_source_3",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# =========================
# 1. Setup
# =========================

if not PROJECT_DIR.exists():
    raise FileNotFoundError(
        f"Project directory does not exist:\n{PROJECT_DIR}"
    )

os.chdir(PROJECT_DIR)

(PROJECT_DIR / "data_processed").mkdir(exist_ok=True)
(PROJECT_DIR / "results").mkdir(exist_ok=True)

print("Current working directory:")
print(Path.cwd())


# =========================
# 2. Input/output checks
# =========================

for path in [
    SCORE_PATH,
    SCORING_MANIFEST_PATH,
    FROZEN_ANNOTATION_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file does not exist:\n{path}"
        )

existing_outputs = [
    path
    for path in [
        FROZEN_SCORE_OUTPUT_PATH,
        VALIDATION_MANIFEST_OUTPUT_PATH,
    ]
    if path.exists()
]

if existing_outputs and not OVERWRITE:
    raise FileExistsError(
        "Frozen score outputs already exist:\n- "
        + "\n- ".join(str(path) for path in existing_outputs)
        + "\n\nArchive them first if recreation is intentional."
    )


# =========================
# 3. Load files
# =========================

score_df = pd.read_csv(SCORE_PATH)
scoring_manifest = pd.read_csv(SCORING_MANIFEST_PATH)
annotation_df = pd.read_csv(FROZEN_ANNOTATION_PATH)

print("\nLoaded score table.")
print("Rows:", len(score_df))

print("\nLoaded frozen annotation table.")
print("Rows:", len(annotation_df))


# =========================
# 4. Validate schema
# =========================

missing_columns = [
    col for col in REQUIRED_SCORE_COLUMNS
    if col not in score_df.columns
]

if missing_columns:
    raise ValueError(
        "Score file is missing required columns:\n- "
        + "\n- ".join(missing_columns)
    )

forbidden_present = sorted(
    FORBIDDEN_SCORE_COLUMNS.intersection(
        set(score_df.columns)
    )
)

if forbidden_present:
    raise ValueError(
        "Score file contains forbidden outcome or annotation columns:\n- "
        + "\n- ".join(forbidden_present)
    )

if len(score_df) != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} score rows, "
        f"found {len(score_df)}."
    )

if len(annotation_df) != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} annotation rows, "
        f"found {len(annotation_df)}."
    )

if score_df["annotation_id"].duplicated().any():
    raise ValueError("Duplicate annotation_id values found.")

if score_df["comparison_id"].duplicated().any():
    raise ValueError("Duplicate comparison_id values found.")


# =========================
# 5. Verify scoring manifest hash
# =========================

if len(scoring_manifest) != 1:
    raise ValueError(
        "Scoring manifest must contain exactly one row."
    )

if "output_sha256" not in scoring_manifest.columns:
    raise ValueError(
        "Scoring manifest has no output_sha256 column."
    )

expected_score_hash = str(
    scoring_manifest.loc[0, "output_sha256"]
).strip()

actual_score_hash = sha256_file(SCORE_PATH)

if actual_score_hash != expected_score_hash:
    raise ValueError(
        "Score-file SHA-256 mismatch.\n"
        f"Expected: {expected_score_hash}\n"
        f"Actual:   {actual_score_hash}"
    )

print("\nScore SHA-256 verified:")
print(actual_score_hash)


# =========================
# 6. Check IDs against frozen annotation
# =========================

score_pairs = set(
    zip(
        score_df["annotation_id"].astype(str),
        score_df["comparison_id"].astype(str),
    )
)

annotation_pairs = set(
    zip(
        annotation_df["annotation_id"].astype(str),
        annotation_df["comparison_id"].astype(str),
    )
)

if score_pairs != annotation_pairs:
    raise ValueError(
        "Score IDs do not exactly match the frozen annotation sample."
    )


# =========================
# 7. Validate numeric values
# =========================

numeric_columns = [
    "skywork_score_a",
    "skywork_score_b",
    "score_gap_abs",
    "diverging_score",
]

for col in numeric_columns:
    score_df[col] = pd.to_numeric(
        score_df[col],
        errors="coerce",
    )

    if score_df[col].isna().any():
        bad_ids = (
            score_df.loc[
                score_df[col].isna(),
                "annotation_id",
            ]
            .astype(str)
            .head(20)
            .tolist()
        )
        raise ValueError(
            f"Non-numeric or missing values in {col}:\n- "
            + "\n- ".join(bad_ids)
        )

    nonfinite_mask = ~score_df[col].map(math.isfinite)

    if nonfinite_mask.any():
        bad_ids = (
            score_df.loc[
                nonfinite_mask,
                "annotation_id",
            ]
            .astype(str)
            .head(20)
            .tolist()
        )
        raise ValueError(
            f"Non-finite values in {col}:\n- "
            + "\n- ".join(bad_ids)
        )


# =========================
# 8. Recompute derived values
# =========================

recomputed_gap = (
    score_df["skywork_score_a"]
    - score_df["skywork_score_b"]
).abs()

gap_error = (
    recomputed_gap
    - score_df["score_gap_abs"]
).abs()

if (gap_error > ABS_TOL).any():
    bad_ids = (
        score_df.loc[
            gap_error > ABS_TOL,
            "annotation_id",
        ]
        .astype(str)
        .head(20)
        .tolist()
    )
    raise ValueError(
        "Stored score_gap_abs does not equal abs(score_a-score_b):\n- "
        + "\n- ".join(bad_ids)
    )

diverging_error = (
    score_df["diverging_score"]
    + score_df["score_gap_abs"]
).abs()

if (diverging_error > ABS_TOL).any():
    bad_ids = (
        score_df.loc[
            diverging_error > ABS_TOL,
            "annotation_id",
        ]
        .astype(str)
        .head(20)
        .tolist()
    )
    raise ValueError(
        "Stored diverging_score does not equal -score_gap_abs:\n- "
        + "\n- ".join(bad_ids)
    )

if (score_df["score_gap_abs"] < 0).any():
    raise ValueError(
        "Negative values found in score_gap_abs."
    )


# =========================
# 9. Restore canonical sample order
# =========================

order_df = annotation_df[
    [
        "annotation_id",
        "comparison_id",
    ]
].copy()

order_df["_canonical_order"] = range(len(order_df))

validated_df = score_df.merge(
    order_df,
    on=[
        "annotation_id",
        "comparison_id",
    ],
    how="left",
    validate="one_to_one",
)

if validated_df["_canonical_order"].isna().any():
    raise ValueError(
        "Some score rows could not be aligned to the frozen sample."
    )

validated_df = (
    validated_df
    .sort_values("_canonical_order")
    .drop(columns=["_canonical_order"])
    .reset_index(drop=True)
)

validated_df = validated_df[
    REQUIRED_SCORE_COLUMNS
].copy()


# =========================
# 10. Save frozen score table
# =========================

validated_df.to_csv(
    FROZEN_SCORE_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

frozen_score_hash = sha256_file(
    FROZEN_SCORE_OUTPUT_PATH
)


# =========================
# 11. Save validation manifest
# =========================

validation_manifest = pd.DataFrame(
    [
        {
            "score_version": "formal_sample_v1_skywork8b_scores_frozen",
            "source_score_file": str(SCORE_PATH),
            "source_score_sha256": actual_score_hash,
            "frozen_score_file": str(FROZEN_SCORE_OUTPUT_PATH),
            "frozen_score_sha256": frozen_score_hash,
            "frozen_annotation_file": str(FROZEN_ANNOTATION_PATH),
            "n_rows": len(validated_df),
            "n_unique_annotation_ids": int(
                validated_df["annotation_id"].nunique()
            ),
            "n_unique_comparison_ids": int(
                validated_df["comparison_id"].nunique()
            ),
            "ids_match_frozen_annotation": True,
            "all_scores_finite": True,
            "all_gaps_recomputed_correctly": True,
            "all_diverging_scores_recomputed_correctly": True,
            "forbidden_outcome_columns_present": False,
            "mean_score_gap_abs": float(
                validated_df["score_gap_abs"].mean()
            ),
            "median_score_gap_abs": float(
                validated_df["score_gap_abs"].median()
            ),
            "min_score_gap_abs": float(
                validated_df["score_gap_abs"].min()
            ),
            "max_score_gap_abs": float(
                validated_df["score_gap_abs"].max()
            ),
        }
    ]
)

validation_manifest.to_csv(
    VALIDATION_MANIFEST_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 12. Final report
# =========================

print("\n==============================")
print("Skywork 8B score validation passed.")
print("Rows:", len(validated_df))
print(
    "Unique annotation IDs:",
    validated_df["annotation_id"].nunique(),
)
print(
    "Unique comparison IDs:",
    validated_df["comparison_id"].nunique(),
)
print("IDs match frozen annotation: True")
print("All scores finite: True")
print("All gaps correct: True")
print("All diverging scores correct: True")
print("Forbidden outcome columns present: False")
print(
    "Mean absolute reward gap:",
    float(validated_df["score_gap_abs"].mean()),
)
print(
    "Median absolute reward gap:",
    float(validated_df["score_gap_abs"].median()),
)
print(
    "Minimum absolute reward gap:",
    float(validated_df["score_gap_abs"].min()),
)
print(
    "Maximum absolute reward gap:",
    float(validated_df["score_gap_abs"].max()),
)
print("==============================")

print("\nFrozen score SHA-256:")
print(frozen_score_hash)

print("\nSaved:")
print(FROZEN_SCORE_OUTPUT_PATH)
print(VALIDATION_MANIFEST_OUTPUT_PATH)

print("\nIMPORTANT:")
print(
    "Use the frozen Skywork score file for all later analyses."
)
