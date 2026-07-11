from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 29_prepare_two_rm_analysis_table_v1.py
#
# Purpose:
# 1. Read the three frozen inputs:
#    - frozen annotation table
#    - frozen Skywork 8B score table
#    - frozen FsfairX score table
# 2. Verify hashes, row counts, IDs, row order, and numerical fields.
# 3. Recompute reward-gap variables and verify stored derived columns.
# 4. Create one deterministic analysis-ready table.
#
# IMPORTANT:
# - This script does NOT compute AUROC.
# - This script does NOT modify any frozen input.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

ANNOTATION_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_stratified_annotation_frozen.csv"
)

SKYWORK_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_skywork8b_scores_frozen.csv"
)

FSFAIRX_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_fsfairx_rm_scores_frozen.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_two_rm_analysis_table.csv"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_analysis_table_manifest.csv"
)

EXPECTED_ROWS = 400

EXPECTED_HASHES = {
    "annotation": (
        "7e8c614e9f9dec12d48e74a31142a2b25fc9d01ec7070570745ef08925a7e10c"
    ),
    "skywork": (
        "b65688cc2039a99b26ad0a6acb349f2fa5bfb5368d63b84260e3ff49e5f73fe7"
    ),
    "fsfairx": (
        "87896f1aabf778f78c8aa1a082fa6ec86014d749e2a014fe90eadc279e6cab68"
    ),
}

ID_PRIORITY = (
    "annotation_id",
    "comparison_id",
    "example_id",
    "sample_id",
    "pair_id",
    "row_id",
    "id",
)

MODEL_SCORE_PAIRS = {
    "skywork": (
        ("skywork_score_a", "skywork_score_b"),
        ("score_a", "score_b"),
        ("reward_a", "reward_b"),
        ("response_a_score", "response_b_score"),
        ("rm_score_a", "rm_score_b"),
    ),
    "fsfairx": (
        ("fsfairx_score_a", "fsfairx_score_b"),
        ("score_a", "score_b"),
        ("reward_a", "reward_b"),
        ("response_a_score", "response_b_score"),
        ("rm_score_a", "rm_score_b"),
    ),
}


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


def normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def validate_basic_table(
    frame: pd.DataFrame,
    table_name: str,
    id_column: str,
) -> None:
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"{table_name} must contain exactly {EXPECTED_ROWS} rows, "
            f"but contains {len(frame)}."
        )

    ids = normalize_ids(frame[id_column])

    if ids.isna().any():
        raise ValueError(
            f"{table_name} contains missing values in {id_column}."
        )

    if ids.eq("").any():
        raise ValueError(
            f"{table_name} contains empty values in {id_column}."
        )

    duplicated = ids[ids.duplicated(keep=False)]
    if not duplicated.empty:
        examples = duplicated.drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{table_name} contains duplicated IDs in {id_column}. "
            f"Examples: {examples}"
        )


def detect_shared_id_column(
    annotation: pd.DataFrame,
    skywork: pd.DataFrame,
    fsfairx: pd.DataFrame,
) -> str:
    for column in ID_PRIORITY:
        if (
            column in annotation.columns
            and column in skywork.columns
            and column in fsfairx.columns
        ):
            return column

    raise ValueError(
        "Could not find one ID column shared by all three frozen tables.\n"
        f"Candidates: {list(ID_PRIORITY)}\n"
        f"Annotation columns: {list(annotation.columns)}\n"
        f"Skywork columns: {list(skywork.columns)}\n"
        f"FsfairX columns: {list(fsfairx.columns)}"
    )


def detect_score_columns(
    frame: pd.DataFrame,
    model_name: str,
) -> tuple[str, str]:
    for score_a, score_b in MODEL_SCORE_PAIRS[model_name]:
        if score_a in frame.columns and score_b in frame.columns:
            return score_a, score_b

    lower_to_original = {
        str(column).lower(): str(column)
        for column in frame.columns
    }

    a_matches = [
        original
        for lower, original in lower_to_original.items()
        if lower.endswith("score_a")
    ]
    b_matches = [
        original
        for lower, original in lower_to_original.items()
        if lower.endswith("score_b")
    ]

    if len(a_matches) == 1 and len(b_matches) == 1:
        return a_matches[0], b_matches[0]

    raise ValueError(
        f"Could not detect score columns for {model_name}.\n"
        f"Actual columns: {list(frame.columns)}"
    )


def verify_same_ids_and_order(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    id_column: str,
    other_name: str,
) -> None:
    reference_ids = normalize_ids(reference[id_column])
    other_ids = normalize_ids(other[id_column])

    reference_set = set(reference_ids.tolist())
    other_set = set(other_ids.tolist())

    missing = sorted(reference_set - other_set)
    extra = sorted(other_set - reference_set)

    if missing or extra:
        raise ValueError(
            f"{other_name} IDs differ from frozen annotation IDs.\n"
            f"Missing from {other_name}: {missing[:10]}\n"
            f"Extra in {other_name}: {extra[:10]}"
        )

    if not reference_ids.reset_index(drop=True).equals(
        other_ids.reset_index(drop=True)
    ):
        first_mismatch = next(
            index
            for index, (reference_id, other_id) in enumerate(
                zip(reference_ids.tolist(), other_ids.tolist())
            )
            if reference_id != other_id
        )
        raise ValueError(
            f"{other_name} has the correct ID set but a different row order.\n"
            f"First mismatch at zero-based row {first_mismatch}:\n"
            f"Annotation ID: {reference_ids.iloc[first_mismatch]}\n"
            f"{other_name} ID: {other_ids.iloc[first_mismatch]}"
        )


def numeric_series(
    frame: pd.DataFrame,
    column: str,
    table_name: str,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    finite_mask = np.isfinite(values.to_numpy(dtype=float))

    if not finite_mask.all():
        bad_rows = np.flatnonzero(~finite_mask).tolist()
        raise ValueError(
            f"{table_name}.{column} contains missing, non-numeric, "
            f"or infinite values. Bad zero-based rows: {bad_rows[:20]}"
        )

    return values.astype(float)


def verify_optional_derived_columns(
    frame: pd.DataFrame,
    model_name: str,
    score_a: pd.Series,
    score_b: pd.Series,
) -> None:
    expected_gap = np.abs(
        score_a.to_numpy(dtype=float)
        - score_b.to_numpy(dtype=float)
    )
    expected_diverging = -expected_gap

    gap_candidates = (
        f"{model_name}_score_gap_abs",
        f"{model_name}_reward_gap_abs",
        "score_gap_abs",
        "reward_gap_abs",
    )

    diverging_candidates = (
        f"{model_name}_diverging_score",
        "diverging_score",
    )

    gap_column = next(
        (column for column in gap_candidates if column in frame.columns),
        None,
    )
    diverging_column = next(
        (
            column
            for column in diverging_candidates
            if column in frame.columns
        ),
        None,
    )

    if gap_column is not None:
        stored_gap = numeric_series(
            frame,
            gap_column,
            f"{model_name} score table",
        ).to_numpy(dtype=float)

        if not np.allclose(
            stored_gap,
            expected_gap,
            rtol=0.0,
            atol=1e-12,
        ):
            max_error = float(
                np.max(np.abs(stored_gap - expected_gap))
            )
            raise ValueError(
                f"{model_name}.{gap_column} does not equal "
                f"abs(score_a - score_b). Maximum error: {max_error}"
            )

    if diverging_column is not None:
        stored_diverging = numeric_series(
            frame,
            diverging_column,
            f"{model_name} score table",
        ).to_numpy(dtype=float)

        if not np.allclose(
            stored_diverging,
            expected_diverging,
            rtol=0.0,
            atol=1e-12,
        ):
            max_error = float(
                np.max(
                    np.abs(
                        stored_diverging - expected_diverging
                    )
                )
            )
            raise ValueError(
                f"{model_name}.{diverging_column} does not equal "
                f"-abs(score_a - score_b). Maximum error: {max_error}"
            )


def build_model_block(
    frame: pd.DataFrame,
    model_name: str,
    id_column: str,
    score_a_column: str,
    score_b_column: str,
) -> pd.DataFrame:
    score_a = numeric_series(
        frame,
        score_a_column,
        f"{model_name} score table",
    )
    score_b = numeric_series(
        frame,
        score_b_column,
        f"{model_name} score table",
    )

    verify_optional_derived_columns(
        frame=frame,
        model_name=model_name,
        score_a=score_a,
        score_b=score_b,
    )

    gap_abs = np.abs(
        score_a.to_numpy(dtype=float)
        - score_b.to_numpy(dtype=float)
    )

    return pd.DataFrame(
        {
            id_column: normalize_ids(frame[id_column]),
            f"{model_name}_score_a": score_a.to_numpy(dtype=float),
            f"{model_name}_score_b": score_b.to_numpy(dtype=float),
            f"{model_name}_score_gap_abs": gap_abs,
            f"{model_name}_diverging_score": -gap_abs,
        }
    )


def main() -> None:
    print("Current working directory:")
    print(Path.cwd())
    print()

    inputs = {
        "annotation": ANNOTATION_PATH,
        "skywork": SKYWORK_PATH,
        "fsfairx": FSFAIRX_PATH,
    }

    for name, path in inputs.items():
        require_file(path, f"Frozen {name} table")

    actual_hashes = {
        name: sha256_file(path)
        for name, path in inputs.items()
    }

    print("Frozen input SHA-256 values:")
    for name in ("annotation", "skywork", "fsfairx"):
        print(f"{name:10s}: {actual_hashes[name]}")
    print()

    for name, expected_hash in EXPECTED_HASHES.items():
        actual_hash = actual_hashes[name]
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen {name} SHA-256 mismatch.\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hash}\n"
                "A frozen input may have changed."
            )

    print("All three frozen input hashes verified.")
    print()

    annotation = pd.read_csv(ANNOTATION_PATH)
    skywork = pd.read_csv(SKYWORK_PATH)
    fsfairx = pd.read_csv(FSFAIRX_PATH)

    print("Loaded tables:")
    print(
        f"Annotation: {len(annotation)} rows, "
        f"{len(annotation.columns)} columns"
    )
    print(
        f"Skywork:   {len(skywork)} rows, "
        f"{len(skywork.columns)} columns"
    )
    print(
        f"FsfairX:   {len(fsfairx)} rows, "
        f"{len(fsfairx.columns)} columns"
    )
    print()

    id_column = detect_shared_id_column(
        annotation=annotation,
        skywork=skywork,
        fsfairx=fsfairx,
    )

    print(f"Shared ID column: {id_column}")
    print()

    validate_basic_table(
        annotation,
        "Frozen annotation table",
        id_column,
    )
    validate_basic_table(
        skywork,
        "Frozen Skywork score table",
        id_column,
    )
    validate_basic_table(
        fsfairx,
        "Frozen FsfairX score table",
        id_column,
    )

    verify_same_ids_and_order(
        reference=annotation,
        other=skywork,
        id_column=id_column,
        other_name="Skywork score table",
    )
    verify_same_ids_and_order(
        reference=annotation,
        other=fsfairx,
        id_column=id_column,
        other_name="FsfairX score table",
    )

    print("ID coverage and row order verified for all three tables.")
    print()

    skywork_score_a, skywork_score_b = detect_score_columns(
        skywork,
        "skywork",
    )
    fsfairx_score_a, fsfairx_score_b = detect_score_columns(
        fsfairx,
        "fsfairx",
    )

    print("Detected score columns:")
    print(
        f"Skywork: {skywork_score_a}, {skywork_score_b}"
    )
    print(
        f"FsfairX: {fsfairx_score_a}, {fsfairx_score_b}"
    )
    print()

    skywork_block = build_model_block(
        frame=skywork,
        model_name="skywork",
        id_column=id_column,
        score_a_column=skywork_score_a,
        score_b_column=skywork_score_b,
    )

    fsfairx_block = build_model_block(
        frame=fsfairx,
        model_name="fsfairx",
        id_column=id_column,
        score_a_column=fsfairx_score_a,
        score_b_column=fsfairx_score_b,
    )

    analysis_table = annotation.copy()
    analysis_table[id_column] = normalize_ids(
        analysis_table[id_column]
    )

    for model_block in (skywork_block, fsfairx_block):
        model_columns = [
            column
            for column in model_block.columns
            if column != id_column
        ]
        for column in model_columns:
            if column in analysis_table.columns:
                raise ValueError(
                    f"Output column collision: {column}"
                )
            analysis_table[column] = model_block[column].to_numpy()

    if len(analysis_table) != EXPECTED_ROWS:
        raise RuntimeError(
            "Unexpected output row count after assembling analysis table."
        )

    new_score_columns = [
        "skywork_score_a",
        "skywork_score_b",
        "skywork_score_gap_abs",
        "skywork_diverging_score",
        "fsfairx_score_a",
        "fsfairx_score_b",
        "fsfairx_score_gap_abs",
        "fsfairx_diverging_score",
    ]

    if analysis_table[new_score_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in newly assembled score columns."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    analysis_table.to_csv(
        OUTPUT_PATH,
        index=False,
        float_format="%.17g",
    )

    output_sha = sha256_file(OUTPUT_PATH)

    manifest = pd.DataFrame(
        [
            {
                "annotation_path": str(ANNOTATION_PATH),
                "annotation_sha256": actual_hashes["annotation"],
                "skywork_path": str(SKYWORK_PATH),
                "skywork_sha256": actual_hashes["skywork"],
                "fsfairx_path": str(FSFAIRX_PATH),
                "fsfairx_sha256": actual_hashes["fsfairx"],
                "output_path": str(OUTPUT_PATH),
                "output_sha256": output_sha,
                "rows": len(analysis_table),
                "columns": len(analysis_table.columns),
                "shared_id_column": id_column,
                "created_at_local": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "auroc_computed": False,
            }
        ]
    )
    manifest.to_csv(MANIFEST_PATH, index=False)

    print("Reward-gap cross-checks passed.")
    print()
    print("Output summary:")
    print(f"Rows:    {len(analysis_table)}")
    print(f"Columns: {len(analysis_table.columns)}")
    print()
    print("Mean absolute reward gaps:")
    print(
        "Skywork: "
        f"{analysis_table['skywork_score_gap_abs'].mean()}"
    )
    print(
        "FsfairX: "
        f"{analysis_table['fsfairx_score_gap_abs'].mean()}"
    )
    print()
    print("Saved analysis-ready table:")
    print(OUTPUT_PATH)
    print()
    print("Output SHA-256:")
    print(output_sha)
    print()
    print("Saved manifest:")
    print(MANIFEST_PATH)
    print()
    print("=" * 72)
    print("Two-RM analysis table preparation completed successfully.")
    print("No AUROC or scenario-difference analysis was performed.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("ANALYSIS TABLE PREPARATION FAILED")
        print("=" * 72)
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)
