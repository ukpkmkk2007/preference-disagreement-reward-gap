from __future__ import annotations

import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 28_validate_and_freeze_fsfairx_scores_v1.py
#
# Purpose:
# 1. Validate the final 400-row FsfairX reward-score table.
# 2. Verify that it matches the frozen annotation sample exactly.
# 3. Verify the expected SHA-256 printed by the scoring script.
# 4. Freeze an exact byte-for-byte copy under data_processed/.
#
# IMPORTANT:
# - This script never reads the *_partial.csv file.
# - This script does not compute AUROC or scenario-level results.
# - An existing frozen file is never overwritten silently.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

SCORE_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_fsfairx_rm_scores.csv"
)

FROZEN_ANNOTATION = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_stratified_annotation_frozen.csv"
)

FROZEN_OUTPUT = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_fsfairx_rm_scores_frozen.csv"
)

FREEZE_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_fsfairx_rm_scores_freeze_manifest.csv"
)

EXPECTED_SCORE_SHA256 = (
    "87896f1aabf778f78c8aa1a082fa6ec86014d749e2a014fe90eadc279e6cab68"
)

EXPECTED_ANNOTATION_SHA256 = (
    "7e8c614e9f9dec12d48e74a31142a2b25fc9d01ec7070570745ef08925a7e10c"
)

EXPECTED_ROWS = 400

ID_CANDIDATES = (
    "annotation_id",
    "comparison_id",
    "example_id",
    "sample_id",
    "pair_id",
    "row_id",
    "id",
)

SCORE_COLUMN_PAIRS = (
    ("fsfairx_score_a", "fsfairx_score_b"),
    ("score_a", "score_b"),
    ("reward_a", "reward_b"),
    ("response_a_score", "response_b_score"),
    ("rm_score_a", "rm_score_b"),
    ("a_score", "b_score"),
)


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
            f"{description} was not found:\n{path}"
        )


def detect_id_column(frame: pd.DataFrame, table_name: str) -> str:
    for column in ID_CANDIDATES:
        if column in frame.columns:
            return column

    raise ValueError(
        f"Could not detect an ID column in {table_name}.\n"
        f"Expected one of: {list(ID_CANDIDATES)}\n"
        f"Actual columns: {list(frame.columns)}"
    )


def detect_score_columns(frame: pd.DataFrame) -> tuple[str, str]:
    for score_a, score_b in SCORE_COLUMN_PAIRS:
        if score_a in frame.columns and score_b in frame.columns:
            return score_a, score_b

    # Fallback: detect one column ending in score_a and one ending in score_b.
    lower_to_original = {str(col).lower(): str(col) for col in frame.columns}
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
        "Could not detect the two FsfairX score columns.\n"
        f"Tried these pairs: {list(SCORE_COLUMN_PAIRS)}\n"
        f"Actual columns: {list(frame.columns)}"
    )


def normalize_ids(series: pd.Series) -> pd.Series:
    # String normalization avoids false mismatches caused by pandas dtype inference.
    return series.astype("string").str.strip()


def validate_table_shape(
    frame: pd.DataFrame,
    table_name: str,
    id_column: str,
) -> None:
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"{table_name} must contain exactly {EXPECTED_ROWS} rows, "
            f"but contains {len(frame)}."
        )

    normalized_ids = normalize_ids(frame[id_column])

    if normalized_ids.isna().any():
        count = int(normalized_ids.isna().sum())
        raise ValueError(
            f"{table_name} contains {count} missing ID value(s)."
        )

    empty_mask = normalized_ids.eq("")
    if empty_mask.any():
        count = int(empty_mask.sum())
        raise ValueError(
            f"{table_name} contains {count} empty ID value(s)."
        )

    duplicated = normalized_ids[normalized_ids.duplicated(keep=False)]
    if not duplicated.empty:
        examples = duplicated.drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{table_name} contains duplicated IDs. "
            f"Examples: {examples}"
        )


def main() -> None:
    print("Current working directory:")
    print(Path.cwd())
    print()

    require_file(SCORE_SOURCE, "Final FsfairX score table")
    require_file(FROZEN_ANNOTATION, "Frozen annotation table")

    if "_partial" in SCORE_SOURCE.name.lower():
        raise RuntimeError(
            "Refusing to freeze a partial score file."
        )

    # --------------------------------------------------------
    # 1. Verify source file identities before reading contents.
    # --------------------------------------------------------
    score_source_sha = sha256_file(SCORE_SOURCE)
    annotation_sha = sha256_file(FROZEN_ANNOTATION)

    print("Source score SHA-256:")
    print(score_source_sha)
    print()

    if score_source_sha != EXPECTED_SCORE_SHA256:
        raise ValueError(
            "Final FsfairX score SHA-256 does not match the expected value.\n"
            f"Expected: {EXPECTED_SCORE_SHA256}\n"
            f"Actual:   {score_source_sha}\n"
            "Do not freeze this file until the mismatch is explained."
        )

    print("FsfairX source score SHA-256 verified.")
    print()

    if annotation_sha != EXPECTED_ANNOTATION_SHA256:
        raise ValueError(
            "Frozen annotation SHA-256 does not match the project record.\n"
            f"Expected: {EXPECTED_ANNOTATION_SHA256}\n"
            f"Actual:   {annotation_sha}\n"
            "The frozen annotation file may have changed."
        )

    print("Frozen annotation SHA-256 verified.")
    print()

    # --------------------------------------------------------
    # 2. Load and validate table structure.
    # --------------------------------------------------------
    scores = pd.read_csv(SCORE_SOURCE)
    annotation = pd.read_csv(FROZEN_ANNOTATION)

    print("Loaded final FsfairX score table.")
    print(f"Rows: {len(scores)}")
    print(f"Columns: {len(scores.columns)}")
    print()

    print("Loaded frozen annotation table.")
    print(f"Rows: {len(annotation)}")
    print(f"Columns: {len(annotation.columns)}")
    print()

    score_id_column = detect_id_column(scores, "FsfairX score table")
    annotation_id_column = detect_id_column(
        annotation,
        "frozen annotation table",
    )
    score_a_column, score_b_column = detect_score_columns(scores)

    print("Detected columns:")
    print(f"Score ID column:      {score_id_column}")
    print(f"Annotation ID column: {annotation_id_column}")
    print(f"Score A column:       {score_a_column}")
    print(f"Score B column:       {score_b_column}")
    print()

    validate_table_shape(scores, "FsfairX score table", score_id_column)
    validate_table_shape(
        annotation,
        "Frozen annotation table",
        annotation_id_column,
    )

    score_ids = normalize_ids(scores[score_id_column])
    annotation_ids = normalize_ids(annotation[annotation_id_column])

    score_id_set = set(score_ids.tolist())
    annotation_id_set = set(annotation_ids.tolist())

    missing_from_scores = sorted(annotation_id_set - score_id_set)
    extra_in_scores = sorted(score_id_set - annotation_id_set)

    if missing_from_scores or extra_in_scores:
        raise ValueError(
            "The score table and frozen annotation contain different IDs.\n"
            f"Missing from scores: {missing_from_scores[:10]}\n"
            f"Extra in scores:     {extra_in_scores[:10]}"
        )

    if not score_ids.reset_index(drop=True).equals(
        annotation_ids.reset_index(drop=True)
    ):
        first_mismatch = next(
            index
            for index, (score_id, annotation_id) in enumerate(
                zip(score_ids.tolist(), annotation_ids.tolist())
            )
            if score_id != annotation_id
        )
        raise ValueError(
            "The two tables contain the same ID set but not in the same order.\n"
            f"First mismatch at zero-based row {first_mismatch}:\n"
            f"Score ID:      {score_ids.iloc[first_mismatch]}\n"
            f"Annotation ID: {annotation_ids.iloc[first_mismatch]}\n"
            "Refusing to reorder automatically because the frozen score file "
            "must remain an exact copy of the validated source."
        )

    print("ID coverage and row order verified.")
    print()

    # --------------------------------------------------------
    # 3. Validate numerical score values.
    # --------------------------------------------------------
    score_a = pd.to_numeric(scores[score_a_column], errors="coerce")
    score_b = pd.to_numeric(scores[score_b_column], errors="coerce")

    invalid_a = ~np.isfinite(score_a.to_numpy(dtype=float))
    invalid_b = ~np.isfinite(score_b.to_numpy(dtype=float))

    if invalid_a.any() or invalid_b.any():
        bad_rows = sorted(
            set(np.flatnonzero(invalid_a).tolist())
            | set(np.flatnonzero(invalid_b).tolist())
        )
        raise ValueError(
            "Non-numeric, missing, or infinite score values were found.\n"
            f"Zero-based bad row indices: {bad_rows[:20]}"
        )

    absolute_gap = np.abs(
        score_a.to_numpy(dtype=float) - score_b.to_numpy(dtype=float)
    )

    print("Numerical score validation passed.")
    print(f"Score A minimum: {score_a.min()}")
    print(f"Score A maximum: {score_a.max()}")
    print(f"Score B minimum: {score_b.min()}")
    print(f"Score B maximum: {score_b.max()}")
    print(f"Mean absolute reward gap:   {absolute_gap.mean()}")
    print(f"Median absolute reward gap: {np.median(absolute_gap)}")
    print(f"Minimum absolute gap:       {absolute_gap.min()}")
    print(f"Maximum absolute gap:       {absolute_gap.max()}")
    print()

    # --------------------------------------------------------
    # 4. Freeze exact bytes; never silently overwrite.
    # --------------------------------------------------------
    FROZEN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    if FROZEN_OUTPUT.exists():
        existing_frozen_sha = sha256_file(FROZEN_OUTPUT)

        if existing_frozen_sha == score_source_sha:
            print("Frozen output already exists and matches the source.")
            print(f"Frozen SHA-256: {existing_frozen_sha}")
        else:
            raise FileExistsError(
                "A different frozen FsfairX score file already exists.\n"
                f"Path: {FROZEN_OUTPUT}\n"
                f"Existing SHA-256: {existing_frozen_sha}\n"
                f"Source SHA-256:   {score_source_sha}\n"
                "The existing frozen file was not overwritten."
            )
    else:
        shutil.copy2(SCORE_SOURCE, FROZEN_OUTPUT)
        existing_frozen_sha = sha256_file(FROZEN_OUTPUT)

        if existing_frozen_sha != score_source_sha:
            FROZEN_OUTPUT.unlink(missing_ok=True)
            raise IOError(
                "Frozen copy SHA-256 does not match the source. "
                "The invalid copy was removed."
            )

        print("Frozen exact byte-for-byte copy created.")
        print(f"Frozen SHA-256: {existing_frozen_sha}")

    print()
    print("Saved frozen score table:")
    print(FROZEN_OUTPUT)
    print()

    # --------------------------------------------------------
    # 5. Save an auditable freeze manifest.
    # --------------------------------------------------------
    manifest = pd.DataFrame(
        [
            {
                "model": "sfairXC/FsfairX-LLaMA3-RM-v0.1",
                "model_revision": (
                    "94fad49f1b3227aa8b566f415a335adb68ec544c"
                ),
                "source_path": str(SCORE_SOURCE),
                "source_sha256": score_source_sha,
                "frozen_path": str(FROZEN_OUTPUT),
                "frozen_sha256": existing_frozen_sha,
                "annotation_path": str(FROZEN_ANNOTATION),
                "annotation_sha256": annotation_sha,
                "rows": len(scores),
                "score_id_column": score_id_column,
                "annotation_id_column": annotation_id_column,
                "score_a_column": score_a_column,
                "score_b_column": score_b_column,
                "validated_at_local": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
        ]
    )
    manifest.to_csv(FREEZE_MANIFEST, index=False)

    print("Saved freeze manifest:")
    print(FREEZE_MANIFEST)
    print()
    print("=" * 70)
    print("FsfairX score validation and freezing completed successfully.")
    print("All 400 score rows are now eligible for the next analysis stage.")
    print("Do not modify the frozen score file.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 70)
        print("VALIDATION / FREEZE FAILED")
        print("=" * 70)
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)
