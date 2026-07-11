from __future__ import annotations

import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# 30_validate_and_freeze_stratified_ids_v1.py
#
# Purpose:
# 1. Validate the fixed outcome/stratum key created by Step 22.
# 2. Cross-check it against the fixed master and frozen annotation.
# 3. Verify the exact 200/200 class balance and six category quotas.
# 4. Freeze an exact byte-for-byte copy for formal analysis.
#
# IMPORTANT:
# - This script does not compute AUROC.
# - It does not modify any existing source or frozen file.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

IDS_SOURCE = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_stratified_ids.csv"
)

MASTER_SOURCE = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_stratified_master.csv"
)

FROZEN_ANNOTATION = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_stratified_annotation_frozen.csv"
)

FROZEN_OUTPUT = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_stratified_ids_frozen.csv"
)

MANIFEST_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_stratified_ids_freeze_manifest.csv"
)

EXPECTED_ANNOTATION_SHA256 = (
    "7e8c614e9f9dec12d48e74a31142a2b25fc9d01ec7070570745ef08925a7e10c"
)

EXPECTED_ROWS = 400
EXPECTED_PER_CLASS = 200

EXPECTED_QUOTAS_PER_CLASS = {
    "Open QA": 40,
    "Generation": 35,
    "Coding": 35,
    "Chat": 35,
    "Brainstorm": 35,
    "Closed QA": 20,
}

REQUIRED_COLUMNS = [
    "annotation_id",
    "comparison_id",
    "sampling_stratum",
    "diverging_id_label",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_binary_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.isna().any():
        bad_values = sorted(
            series[numeric.isna()]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "diverging_id_label contains non-numeric values: "
            f"{bad_values}"
        )

    if not numeric.isin([0, 1]).all():
        bad_values = sorted(numeric[~numeric.isin([0, 1])].unique())
        raise ValueError(
            "diverging_id_label must contain only 0 and 1. "
            f"Bad values: {bad_values}"
        )

    return numeric.astype("int64")


def validate_required_columns(
    frame: pd.DataFrame,
    table_name: str,
) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns:\n- "
            + "\n- ".join(missing)
        )


def validate_unique_ids(
    frame: pd.DataFrame,
    table_name: str,
) -> None:
    for column in ["annotation_id", "comparison_id"]:
        values = normalize_text(frame[column])

        if values.isna().any() or values.eq("").any():
            raise ValueError(
                f"{table_name}.{column} contains missing or empty IDs."
            )

        duplicated = values[values.duplicated(keep=False)]

        if not duplicated.empty:
            examples = (
                duplicated.drop_duplicates().head(10).tolist()
            )
            raise ValueError(
                f"{table_name}.{column} contains duplicates. "
                f"Examples: {examples}"
            )


def verify_same_id_pairs_and_order(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    other_name: str,
) -> None:
    reference_pairs = list(
        zip(
            normalize_text(reference["annotation_id"]).tolist(),
            normalize_text(reference["comparison_id"]).tolist(),
        )
    )

    other_pairs = list(
        zip(
            normalize_text(other["annotation_id"]).tolist(),
            normalize_text(other["comparison_id"]).tolist(),
        )
    )

    if set(reference_pairs) != set(other_pairs):
        missing = sorted(set(reference_pairs) - set(other_pairs))
        extra = sorted(set(other_pairs) - set(reference_pairs))

        raise ValueError(
            f"{other_name} contains different ID pairs.\n"
            f"Missing examples: {missing[:10]}\n"
            f"Extra examples:   {extra[:10]}"
        )

    if reference_pairs != other_pairs:
        first_mismatch = next(
            index
            for index, (left, right) in enumerate(
                zip(reference_pairs, other_pairs)
            )
            if left != right
        )

        raise ValueError(
            f"{other_name} has a different row order.\n"
            f"First mismatch at zero-based row {first_mismatch}:\n"
            f"Reference: {reference_pairs[first_mismatch]}\n"
            f"Other:     {other_pairs[first_mismatch]}"
        )


def main() -> None:
    print("Current working directory:")
    print(Path.cwd())
    print()

    require_file(IDS_SOURCE, "Stratified ID key")
    require_file(MASTER_SOURCE, "Stratified master table")
    require_file(FROZEN_ANNOTATION, "Frozen annotation table")

    annotation_sha = sha256_file(FROZEN_ANNOTATION)

    print("Frozen annotation SHA-256:")
    print(annotation_sha)
    print()

    if annotation_sha != EXPECTED_ANNOTATION_SHA256:
        raise ValueError(
            "Frozen annotation SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_ANNOTATION_SHA256}\n"
            f"Actual:   {annotation_sha}"
        )

    ids_source_sha = sha256_file(IDS_SOURCE)
    master_source_sha = sha256_file(MASTER_SOURCE)

    print("Source file SHA-256 values:")
    print(f"IDs key: {ids_source_sha}")
    print(f"Master:  {master_source_sha}")
    print()

    ids_df = pd.read_csv(IDS_SOURCE)
    master_df = pd.read_csv(MASTER_SOURCE)
    annotation_df = pd.read_csv(FROZEN_ANNOTATION)

    print("Loaded tables:")
    print(
        f"IDs key:    {len(ids_df)} rows, "
        f"{len(ids_df.columns)} columns"
    )
    print(
        f"Master:     {len(master_df)} rows, "
        f"{len(master_df.columns)} columns"
    )
    print(
        f"Annotation: {len(annotation_df)} rows, "
        f"{len(annotation_df.columns)} columns"
    )
    print()

    validate_required_columns(ids_df, "Stratified ID key")
    validate_required_columns(master_df, "Stratified master table")

    annotation_required = ["annotation_id", "comparison_id"]
    missing_annotation = [
        column
        for column in annotation_required
        if column not in annotation_df.columns
    ]

    if missing_annotation:
        raise ValueError(
            "Frozen annotation is missing ID columns:\n- "
            + "\n- ".join(missing_annotation)
        )

    if len(ids_df) != EXPECTED_ROWS:
        raise ValueError(
            f"IDs key must contain {EXPECTED_ROWS} rows, "
            f"but contains {len(ids_df)}."
        )

    if len(master_df) != EXPECTED_ROWS:
        raise ValueError(
            f"Master table must contain {EXPECTED_ROWS} rows, "
            f"but contains {len(master_df)}."
        )

    if len(annotation_df) != EXPECTED_ROWS:
        raise ValueError(
            f"Frozen annotation must contain {EXPECTED_ROWS} rows, "
            f"but contains {len(annotation_df)}."
        )

    validate_unique_ids(ids_df, "Stratified ID key")
    validate_unique_ids(master_df, "Stratified master table")
    validate_unique_ids(annotation_df, "Frozen annotation table")

    verify_same_id_pairs_and_order(
        reference=ids_df,
        other=master_df,
        other_name="Stratified master table",
    )

    verify_same_id_pairs_and_order(
        reference=ids_df,
        other=annotation_df,
        other_name="Frozen annotation table",
    )

    print("ID pairs and row order verified across all three tables.")
    print()

    ids_df = ids_df[REQUIRED_COLUMNS].copy()
    master_key = master_df[REQUIRED_COLUMNS].copy()

    for frame in (ids_df, master_key):
        frame["annotation_id"] = normalize_text(
            frame["annotation_id"]
        )
        frame["comparison_id"] = normalize_text(
            frame["comparison_id"]
        )
        frame["sampling_stratum"] = normalize_text(
            frame["sampling_stratum"]
        )
        frame["diverging_id_label"] = normalize_binary_label(
            frame["diverging_id_label"]
        )

    if not ids_df.equals(master_key):
        mismatch_columns = []

        for column in REQUIRED_COLUMNS:
            if not ids_df[column].equals(master_key[column]):
                mismatch_columns.append(column)

        raise ValueError(
            "IDs key does not exactly match the corresponding master "
            "columns.\n"
            f"Mismatching columns: {mismatch_columns}"
        )

    print("Outcome labels and sampling strata match the master table.")
    print()

    labels = ids_df["diverging_id_label"]

    n_high = int((labels == 0).sum())
    n_diverging = int((labels == 1).sum())

    if n_high != EXPECTED_PER_CLASS:
        raise ValueError(
            f"Expected {EXPECTED_PER_CLASS} high-agreement rows, "
            f"found {n_high}."
        )

    if n_diverging != EXPECTED_PER_CLASS:
        raise ValueError(
            f"Expected {EXPECTED_PER_CLASS} diverging rows, "
            f"found {n_diverging}."
        )

    actual_strata = set(ids_df["sampling_stratum"].unique())
    expected_strata = set(EXPECTED_QUOTAS_PER_CLASS)

    if actual_strata != expected_strata:
        raise ValueError(
            "Unexpected sampling strata.\n"
            f"Expected: {sorted(expected_strata)}\n"
            f"Actual:   {sorted(actual_strata)}"
        )

    counts = (
        ids_df
        .groupby(
            ["sampling_stratum", "diverging_id_label"],
            observed=False,
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
    )

    for stratum, expected_per_class in (
        EXPECTED_QUOTAS_PER_CLASS.items()
    ):
        row = counts[counts["sampling_stratum"] == stratum]

        if len(row) != 1:
            raise ValueError(
                f"Missing count row for stratum: {stratum}"
            )

        actual_high = int(row.iloc[0]["n_high_agreement"])
        actual_diverging = int(row.iloc[0]["n_diverging"])

        if (
            actual_high != expected_per_class
            or actual_diverging != expected_per_class
        ):
            raise ValueError(
                f"{stratum}: expected "
                f"{expected_per_class}/{expected_per_class}, "
                f"found {actual_diverging}/{actual_high} "
                "(diverging/high-agreement)."
            )

    counts["n_total"] = (
        counts["n_high_agreement"]
        + counts["n_diverging"]
    )

    print("Class balance verified:")
    print(f"Diverging:      {n_diverging}")
    print(f"High-agreement: {n_high}")
    print()

    print("Per-stratum counts:")
    print(counts.to_string(index=False))
    print()

    FROZEN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if FROZEN_OUTPUT.exists():
        existing_sha = sha256_file(FROZEN_OUTPUT)

        if existing_sha != ids_source_sha:
            raise FileExistsError(
                "A different frozen IDs key already exists.\n"
                f"Path: {FROZEN_OUTPUT}\n"
                f"Existing SHA-256: {existing_sha}\n"
                f"Source SHA-256:   {ids_source_sha}\n"
                "The existing file was not overwritten."
            )

        frozen_sha = existing_sha
        print(
            "Frozen IDs key already exists and matches the source."
        )
    else:
        shutil.copy2(IDS_SOURCE, FROZEN_OUTPUT)
        frozen_sha = sha256_file(FROZEN_OUTPUT)

        if frozen_sha != ids_source_sha:
            FROZEN_OUTPUT.unlink(missing_ok=True)
            raise IOError(
                "Frozen copy SHA-256 does not match the source. "
                "The invalid copy was removed."
            )

        print("Frozen exact byte-for-byte IDs key created.")

    print()
    print("Frozen IDs key SHA-256:")
    print(frozen_sha)
    print()

    manifest = pd.DataFrame(
        [
            {
                "source_ids_path": str(IDS_SOURCE),
                "source_ids_sha256": ids_source_sha,
                "frozen_ids_path": str(FROZEN_OUTPUT),
                "frozen_ids_sha256": frozen_sha,
                "master_path": str(MASTER_SOURCE),
                "master_sha256": master_source_sha,
                "frozen_annotation_path": str(FROZEN_ANNOTATION),
                "frozen_annotation_sha256": annotation_sha,
                "rows": len(ids_df),
                "n_diverging": n_diverging,
                "n_high_agreement": n_high,
                "n_sampling_strata": ids_df[
                    "sampling_stratum"
                ].nunique(),
                "ids_match_master": True,
                "ids_match_frozen_annotation": True,
                "labels_and_strata_match_master": True,
                "validated_at_local": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
        ]
    )

    manifest.to_csv(
        MANIFEST_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved frozen IDs key:")
    print(FROZEN_OUTPUT)
    print()
    print("Saved freeze manifest:")
    print(MANIFEST_OUTPUT)
    print()
    print("=" * 72)
    print("Stratified outcome/stratum key validation completed.")
    print("No AUROC or scenario analysis was performed.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("STRATIFIED IDS VALIDATION / FREEZE FAILED")
        print("=" * 72)
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)
