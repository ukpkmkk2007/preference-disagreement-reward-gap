from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# 37_audit_preference_accuracy_inputs_v1.py
#
# Purpose:
# 1. Inspect the fixed stratified master table for columns that
#    can define human majority preference / chosen response.
# 2. Verify row identity against the frozen labeled analysis table.
# 3. Print candidate columns, dtypes, missingness, and value counts.
#
# IMPORTANT:
# - This script does not compute preference accuracy.
# - This script does not compute overconfident-disagreement rate.
# - It does not modify any file.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

MASTER_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_stratified_master.csv"
)

LABELED_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_two_rm_labeled_analysis_table.csv"
)

EXPECTED_ANALYSIS_SHA256 = (
    "d9447c490d613ee65b2618ed9025ca5e82227b925c87df2a5020d5d9b8da7f56"
)

EXPECTED_ROWS = 400

ID_COLUMNS = [
    "annotation_id",
    "comparison_id",
]

PREFERENCE_KEYWORDS = [
    "preference",
    "preferred",
    "winner",
    "chosen",
    "choice",
    "label",
    "majority",
    "vote",
    "response_a",
    "response_b",
    "tie",
    "agreement",
    "rank",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def detect_candidate_columns(frame: pd.DataFrame) -> list[str]:
    candidates = []

    for column in frame.columns:
        lower = str(column).lower()

        if any(keyword in lower for keyword in PREFERENCE_KEYWORDS):
            candidates.append(str(column))

    return candidates


def compact_value_counts(
    series: pd.Series,
    max_values: int = 30,
) -> str:
    counts = (
        series
        .astype("string")
        .fillna("<NA>")
        .value_counts(
            dropna=False,
        )
    )

    lines = []

    for value, count in counts.head(max_values).items():
        lines.append(
            f"{repr(str(value))}: {int(count)}"
        )

    if len(counts) > max_values:
        lines.append(
            f"... {len(counts) - max_values} more unique values"
        )

    return "\n".join(lines)


def main() -> None:
    print("Current working directory:")
    print(Path.cwd())
    print()

    require_file(
        MASTER_PATH,
        "Stratified master table",
    )
    require_file(
        LABELED_ANALYSIS_PATH,
        "Final labeled analysis table",
    )

    analysis_sha = sha256_file(
        LABELED_ANALYSIS_PATH
    )

    print("Labeled analysis SHA-256:")
    print(analysis_sha)
    print()

    if analysis_sha != EXPECTED_ANALYSIS_SHA256:
        raise ValueError(
            "Labeled analysis SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_ANALYSIS_SHA256}\n"
            f"Actual:   {analysis_sha}"
        )

    master = pd.read_csv(MASTER_PATH)
    analysis = pd.read_csv(
        LABELED_ANALYSIS_PATH
    )

    print("Loaded tables:")
    print(
        f"Master:   {len(master)} rows, "
        f"{len(master.columns)} columns"
    )
    print(
        f"Analysis: {len(analysis)} rows, "
        f"{len(analysis.columns)} columns"
    )
    print()

    if len(master) != EXPECTED_ROWS:
        raise ValueError(
            f"Master must contain {EXPECTED_ROWS} rows, "
            f"found {len(master)}."
        )

    if len(analysis) != EXPECTED_ROWS:
        raise ValueError(
            f"Analysis must contain {EXPECTED_ROWS} rows, "
            f"found {len(analysis)}."
        )

    for column in ID_COLUMNS:
        if column not in master.columns:
            raise ValueError(
                f"Master is missing ID column: {column}"
            )

        if column not in analysis.columns:
            raise ValueError(
                f"Analysis is missing ID column: {column}"
            )

    master_pairs = list(
        zip(
            normalize_id(
                master["annotation_id"]
            ).tolist(),
            normalize_id(
                master["comparison_id"]
            ).tolist(),
        )
    )

    analysis_pairs = list(
        zip(
            normalize_id(
                analysis["annotation_id"]
            ).tolist(),
            normalize_id(
                analysis["comparison_id"]
            ).tolist(),
        )
    )

    if set(master_pairs) != set(analysis_pairs):
        missing = sorted(
            set(analysis_pairs)
            - set(master_pairs)
        )
        extra = sorted(
            set(master_pairs)
            - set(analysis_pairs)
        )

        raise ValueError(
            "Master and analysis contain different ID pairs.\n"
            f"Missing from master: {missing[:10]}\n"
            f"Extra in master:     {extra[:10]}"
        )

    if master_pairs != analysis_pairs:
        first = next(
            index
            for index, (left, right) in enumerate(
                zip(master_pairs, analysis_pairs)
            )
            if left != right
        )

        raise ValueError(
            "Master and analysis have the same ID pairs "
            "but different row order.\n"
            f"First mismatch at zero-based row {first}:\n"
            f"Master:   {master_pairs[first]}\n"
            f"Analysis: {analysis_pairs[first]}"
        )

    print("ID pairs and row order verified.")
    print()

    candidates = detect_candidate_columns(
        master
    )

    print("All master columns:")
    print()

    for index, column in enumerate(
        master.columns,
        start=1,
    ):
        print(f"{index:02d}. {column}")

    print()
    print("=" * 72)
    print("Preference-related candidate columns")
    print("=" * 72)
    print()

    if not candidates:
        print(
            "No candidate columns matched the keyword list."
        )
    else:
        for column in candidates:
            series = master[column]

            print(f"Column: {column}")
            print(f"dtype: {series.dtype}")
            print(
                f"missing: {int(series.isna().sum())}"
            )
            print(
                f"unique_nonmissing: "
                f"{int(series.nunique(dropna=True))}"
            )
            print("value counts:")
            print(
                compact_value_counts(series)
            )
            print("-" * 72)

    print()
    print("=" * 72)
    print("Audit completed successfully.")
    print(
        "No preference accuracy or overconfident-disagreement "
        "metric was computed."
    )
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("PREFERENCE INPUT AUDIT FAILED")
        print("=" * 72)
        print(
            f"{type(error).__name__}: {error}"
        )
        sys.exit(1)
