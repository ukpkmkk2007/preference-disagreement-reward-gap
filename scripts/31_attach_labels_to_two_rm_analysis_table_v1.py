from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# 31_attach_labels_to_two_rm_analysis_table_v1.py
#
# Purpose:
# 1. Read the validated two-RM analysis table from Step 29.
# 2. Read the frozen outcome/stratum key from Step 30.
# 3. Verify hashes, row counts, IDs, row order, labels, and quotas.
# 4. Append sampling_stratum and diverging_id_label.
# 5. Save one final labeled analysis table for AUROC work.
#
# IMPORTANT:
# - This script does NOT compute AUROC.
# - This script does NOT modify any frozen input.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

ANALYSIS_SOURCE = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_two_rm_analysis_table.csv"
)

FROZEN_IDS = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_stratified_ids_frozen.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_two_rm_labeled_analysis_table.csv"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_labeled_analysis_table_manifest.csv"
)

EXPECTED_ANALYSIS_SHA256 = (
    "804fc4119b07b663a2ce1ae6ca88a1cdb3f280f119f856a2cee29fe136727bbf"
)

EXPECTED_IDS_SHA256 = (
    "981edcd63732d41e7bfd3e5ef7a18f2076ab95dd0e85b5ff4bd73debef344bbb"
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

REQUIRED_ANALYSIS_COLUMNS = [
    "annotation_id",
    "comparison_id",
    "skywork_score_a",
    "skywork_score_b",
    "skywork_score_gap_abs",
    "skywork_diverging_score",
    "fsfairx_score_a",
    "fsfairx_score_b",
    "fsfairx_score_gap_abs",
    "fsfairx_diverging_score",
]

REQUIRED_IDS_COLUMNS = [
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


def require_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns:\n- "
            + "\n- ".join(missing)
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
        bad_values = sorted(
            numeric[~numeric.isin([0, 1])]
            .unique()
            .tolist()
        )
        raise ValueError(
            "diverging_id_label must contain only 0 and 1. "
            f"Bad values: {bad_values}"
        )

    return numeric.astype("int64")


def validate_unique_id_pairs(
    frame: pd.DataFrame,
    table_name: str,
) -> None:
    annotation_ids = normalize_text(frame["annotation_id"])
    comparison_ids = normalize_text(frame["comparison_id"])

    for column_name, values in (
        ("annotation_id", annotation_ids),
        ("comparison_id", comparison_ids),
    ):
        if values.isna().any() or values.eq("").any():
            raise ValueError(
                f"{table_name}.{column_name} contains "
                "missing or empty values."
            )

        duplicated = values[values.duplicated(keep=False)]

        if not duplicated.empty:
            examples = (
                duplicated.drop_duplicates().head(10).tolist()
            )
            raise ValueError(
                f"{table_name}.{column_name} contains duplicates. "
                f"Examples: {examples}"
            )

    pairs = pd.DataFrame(
        {
            "annotation_id": annotation_ids,
            "comparison_id": comparison_ids,
        }
    )

    duplicated_pairs = pairs.duplicated(keep=False)

    if duplicated_pairs.any():
        examples = (
            pairs.loc[duplicated_pairs]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{table_name} contains duplicate ID pairs. "
            f"Examples: {examples}"
        )


def verify_same_id_pairs_and_order(
    analysis: pd.DataFrame,
    ids_key: pd.DataFrame,
) -> None:
    analysis_pairs = list(
        zip(
            normalize_text(
                analysis["annotation_id"]
            ).tolist(),
            normalize_text(
                analysis["comparison_id"]
            ).tolist(),
        )
    )

    key_pairs = list(
        zip(
            normalize_text(
                ids_key["annotation_id"]
            ).tolist(),
            normalize_text(
                ids_key["comparison_id"]
            ).tolist(),
        )
    )

    if set(analysis_pairs) != set(key_pairs):
        missing = sorted(set(analysis_pairs) - set(key_pairs))
        extra = sorted(set(key_pairs) - set(analysis_pairs))

        raise ValueError(
            "Analysis table and frozen IDs key contain "
            "different ID pairs.\n"
            f"Missing from IDs key: {missing[:10]}\n"
            f"Extra in IDs key:     {extra[:10]}"
        )

    if analysis_pairs != key_pairs:
        first_mismatch = next(
            index
            for index, (left, right) in enumerate(
                zip(analysis_pairs, key_pairs)
            )
            if left != right
        )

        raise ValueError(
            "Analysis table and frozen IDs key contain the "
            "same ID pairs but in different row order.\n"
            f"First mismatch at zero-based row {first_mismatch}:\n"
            f"Analysis: {analysis_pairs[first_mismatch]}\n"
            f"IDs key:  {key_pairs[first_mismatch]}"
        )


def verify_class_balance_and_quotas(
    ids_key: pd.DataFrame,
) -> pd.DataFrame:
    labels = ids_key["diverging_id_label"]

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

    actual_strata = set(
        ids_key["sampling_stratum"].unique()
    )
    expected_strata = set(EXPECTED_QUOTAS_PER_CLASS)

    if actual_strata != expected_strata:
        raise ValueError(
            "Unexpected sampling strata.\n"
            f"Expected: {sorted(expected_strata)}\n"
            f"Actual:   {sorted(actual_strata)}"
        )

    counts = (
        ids_key
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
        row = counts[
            counts["sampling_stratum"] == stratum
        ]

        if len(row) != 1:
            raise ValueError(
                f"Missing quota row for stratum: {stratum}"
            )

        actual_high = int(
            row.iloc[0]["n_high_agreement"]
        )
        actual_diverging = int(
            row.iloc[0]["n_diverging"]
        )

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

    return counts


def main() -> None:
    print("Current working directory:")
    print(Path.cwd())
    print()

    require_file(
        ANALYSIS_SOURCE,
        "Two-RM analysis table",
    )
    require_file(
        FROZEN_IDS,
        "Frozen stratified IDs key",
    )

    analysis_sha = sha256_file(ANALYSIS_SOURCE)
    ids_sha = sha256_file(FROZEN_IDS)

    print("Input SHA-256 values:")
    print(f"Two-RM analysis table: {analysis_sha}")
    print(f"Frozen IDs key:        {ids_sha}")
    print()

    if analysis_sha != EXPECTED_ANALYSIS_SHA256:
        raise ValueError(
            "Two-RM analysis table SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_ANALYSIS_SHA256}\n"
            f"Actual:   {analysis_sha}"
        )

    if ids_sha != EXPECTED_IDS_SHA256:
        raise ValueError(
            "Frozen IDs key SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_IDS_SHA256}\n"
            f"Actual:   {ids_sha}"
        )

    print("Both input hashes verified.")
    print()

    analysis = pd.read_csv(ANALYSIS_SOURCE)
    ids_key = pd.read_csv(FROZEN_IDS)

    print("Loaded tables:")
    print(
        f"Two-RM analysis table: "
        f"{len(analysis)} rows, "
        f"{len(analysis.columns)} columns"
    )
    print(
        f"Frozen IDs key:        "
        f"{len(ids_key)} rows, "
        f"{len(ids_key.columns)} columns"
    )
    print()

    require_columns(
        analysis,
        REQUIRED_ANALYSIS_COLUMNS,
        "Two-RM analysis table",
    )
    require_columns(
        ids_key,
        REQUIRED_IDS_COLUMNS,
        "Frozen IDs key",
    )

    if len(analysis) != EXPECTED_ROWS:
        raise ValueError(
            f"Two-RM analysis table must contain "
            f"{EXPECTED_ROWS} rows, "
            f"but contains {len(analysis)}."
        )

    if len(ids_key) != EXPECTED_ROWS:
        raise ValueError(
            f"Frozen IDs key must contain "
            f"{EXPECTED_ROWS} rows, "
            f"but contains {len(ids_key)}."
        )

    validate_unique_id_pairs(
        analysis,
        "Two-RM analysis table",
    )
    validate_unique_id_pairs(
        ids_key,
        "Frozen IDs key",
    )

    verify_same_id_pairs_and_order(
        analysis=analysis,
        ids_key=ids_key,
    )

    print("ID pairs and row order verified.")
    print()

    ids_key = ids_key[
        REQUIRED_IDS_COLUMNS
    ].copy()

    ids_key["annotation_id"] = normalize_text(
        ids_key["annotation_id"]
    )
    ids_key["comparison_id"] = normalize_text(
        ids_key["comparison_id"]
    )
    ids_key["sampling_stratum"] = normalize_text(
        ids_key["sampling_stratum"]
    )
    ids_key["diverging_id_label"] = (
        normalize_binary_label(
            ids_key["diverging_id_label"]
        )
    )

    counts = verify_class_balance_and_quotas(
        ids_key
    )

    print("Class balance and six stratum quotas verified.")
    print()
    print(counts.to_string(index=False))
    print()

    collision_columns = [
        column
        for column in (
            "sampling_stratum",
            "diverging_id_label",
        )
        if column in analysis.columns
    ]

    if collision_columns:
        raise ValueError(
            "The analysis table already contains columns that "
            "this script is supposed to attach:\n- "
            + "\n- ".join(collision_columns)
        )

    output = analysis.copy()

    output["sampling_stratum"] = (
        ids_key["sampling_stratum"].to_numpy()
    )
    output["diverging_id_label"] = (
        ids_key["diverging_id_label"].to_numpy()
    )

    if len(output) != EXPECTED_ROWS:
        raise RuntimeError(
            "Unexpected output row count."
        )

    if output[
        ["sampling_stratum", "diverging_id_label"]
    ].isna().any().any():
        raise ValueError(
            "Missing labels or strata after attachment."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        float_format="%.17g",
    )

    output_sha = sha256_file(OUTPUT_PATH)

    manifest = pd.DataFrame(
        [
            {
                "analysis_source_path": str(
                    ANALYSIS_SOURCE
                ),
                "analysis_source_sha256": analysis_sha,
                "frozen_ids_path": str(FROZEN_IDS),
                "frozen_ids_sha256": ids_sha,
                "output_path": str(OUTPUT_PATH),
                "output_sha256": output_sha,
                "rows": len(output),
                "columns": len(output.columns),
                "n_diverging": int(
                    (
                        output["diverging_id_label"]
                        == 1
                    ).sum()
                ),
                "n_high_agreement": int(
                    (
                        output["diverging_id_label"]
                        == 0
                    ).sum()
                ),
                "n_sampling_strata": int(
                    output["sampling_stratum"].nunique()
                ),
                "id_pairs_and_order_verified": True,
                "class_balance_verified": True,
                "stratum_quotas_verified": True,
                "auroc_computed": False,
                "created_at_local": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
            }
        ]
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved labeled analysis table:")
    print(OUTPUT_PATH)
    print()
    print("Output summary:")
    print(f"Rows:    {len(output)}")
    print(f"Columns: {len(output.columns)}")
    print()
    print("Output SHA-256:")
    print(output_sha)
    print()
    print("Saved manifest:")
    print(MANIFEST_PATH)
    print()
    print("=" * 72)
    print(
        "Final labeled two-RM analysis table "
        "prepared successfully."
    )
    print(
        "No AUROC or scenario-difference analysis "
        "was performed."
    )
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("LABEL ATTACHMENT FAILED")
        print("=" * 72)
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)
