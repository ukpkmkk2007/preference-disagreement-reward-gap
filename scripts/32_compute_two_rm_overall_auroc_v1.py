from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# 32_compute_two_rm_overall_auroc_v1.py
#
# Purpose:
# 1. Verify the final labeled two-RM analysis table.
# 2. Compute overall AUROC for Skywork and FsfairX.
# 3. Use diverging_id_label == 1 as the positive class.
# 4. Use -abs(score_a - score_b) as the primary predictor.
#
# IMPORTANT:
# - This script computes only overall AUROC.
# - It does not compute category-specific AUROC.
# - It does not compute bootstrap confidence intervals.
# - It does not modify any frozen or analysis input file.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_two_rm_labeled_analysis_table.csv"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_overall_auroc.csv"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_overall_auroc_manifest.csv"
)

EXPECTED_INPUT_SHA256 = (
    "d9447c490d613ee65b2618ed9025ca5e82227b925c87df2a5020d5d9b8da7f56"
)

EXPECTED_ROWS = 400
EXPECTED_DIVERGING = 200
EXPECTED_HIGH_AGREEMENT = 200

LABEL_COLUMN = "diverging_id_label"

MODEL_COLUMNS = {
    "Skywork-Reward-Llama-3.1-8B-v0.2": {
        "score_a": "skywork_score_a",
        "score_b": "skywork_score_b",
        "gap_abs": "skywork_score_gap_abs",
        "diverging_score": "skywork_diverging_score",
    },
    "FsfairX-LLaMA3-RM-v0.1": {
        "score_a": "fsfairx_score_a",
        "score_b": "fsfairx_score_b",
        "gap_abs": "fsfairx_score_gap_abs",
        "diverging_score": "fsfairx_diverging_score",
    },
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


def require_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Input table is missing required columns:\n- "
            + "\n- ".join(missing)
        )


def finite_numeric_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    finite_mask = np.isfinite(values.to_numpy(dtype=float))

    if not finite_mask.all():
        bad_rows = np.flatnonzero(~finite_mask).tolist()

        raise ValueError(
            f"{column} contains missing, non-numeric, or infinite "
            f"values. Bad zero-based rows: {bad_rows[:20]}"
        )

    return values.astype(float)


def validate_labels(frame: pd.DataFrame) -> pd.Series:
    labels = pd.to_numeric(
        frame[LABEL_COLUMN],
        errors="coerce",
    )

    if labels.isna().any():
        bad_values = sorted(
            frame.loc[
                labels.isna(),
                LABEL_COLUMN,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{LABEL_COLUMN} contains non-numeric values: "
            f"{bad_values}"
        )

    if not labels.isin([0, 1]).all():
        bad_values = sorted(
            labels[~labels.isin([0, 1])]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{LABEL_COLUMN} must contain only 0 and 1. "
            f"Bad values: {bad_values}"
        )

    labels = labels.astype("int64")

    n_diverging = int((labels == 1).sum())
    n_high_agreement = int((labels == 0).sum())

    if n_diverging != EXPECTED_DIVERGING:
        raise ValueError(
            f"Expected {EXPECTED_DIVERGING} diverging rows, "
            f"found {n_diverging}."
        )

    if n_high_agreement != EXPECTED_HIGH_AGREEMENT:
        raise ValueError(
            f"Expected {EXPECTED_HIGH_AGREEMENT} high-agreement "
            f"rows, found {n_high_agreement}."
        )

    return labels


def verify_model_columns(
    frame: pd.DataFrame,
    model_name: str,
    columns: dict[str, str],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    score_a = finite_numeric_series(
        frame,
        columns["score_a"],
    )
    score_b = finite_numeric_series(
        frame,
        columns["score_b"],
    )
    stored_gap = finite_numeric_series(
        frame,
        columns["gap_abs"],
    )
    stored_diverging = finite_numeric_series(
        frame,
        columns["diverging_score"],
    )

    recomputed_gap = np.abs(
        score_a.to_numpy(dtype=float)
        - score_b.to_numpy(dtype=float)
    )
    recomputed_diverging = -recomputed_gap

    if not np.allclose(
        stored_gap.to_numpy(dtype=float),
        recomputed_gap,
        rtol=0.0,
        atol=1e-12,
    ):
        max_error = float(
            np.max(
                np.abs(
                    stored_gap.to_numpy(dtype=float)
                    - recomputed_gap
                )
            )
        )

        raise ValueError(
            f"{model_name}: stored absolute reward gap does not "
            f"equal abs(score_a - score_b). "
            f"Maximum error: {max_error}"
        )

    if not np.allclose(
        stored_diverging.to_numpy(dtype=float),
        recomputed_diverging,
        rtol=0.0,
        atol=1e-12,
    ):
        max_error = float(
            np.max(
                np.abs(
                    stored_diverging.to_numpy(dtype=float)
                    - recomputed_diverging
                )
            )
        )

        raise ValueError(
            f"{model_name}: stored diverging score does not equal "
            f"-abs(score_a - score_b). "
            f"Maximum error: {max_error}"
        )

    return score_a, score_b, stored_gap, stored_diverging


def safe_float(value: float) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"Encountered non-finite result: {value}"
        )

    return value


def compute_model_result(
    labels: pd.Series,
    gap_abs: pd.Series,
    diverging_score: pd.Series,
    model_name: str,
) -> dict[str, object]:
    y_true = labels.to_numpy(dtype=int)
    y_diverging_score = diverging_score.to_numpy(dtype=float)
    y_gap_abs = gap_abs.to_numpy(dtype=float)

    auroc_primary = safe_float(
        roc_auc_score(
            y_true,
            y_diverging_score,
        )
    )

    auroc_raw_gap = safe_float(
        roc_auc_score(
            y_true,
            y_gap_abs,
        )
    )

    complement_error = abs(
        auroc_primary + auroc_raw_gap - 1.0
    )

    if complement_error > 1e-12:
        raise ValueError(
            f"{model_name}: AUROC orientation sanity check failed. "
            f"AUC(-gap) + AUC(gap) = "
            f"{auroc_primary + auroc_raw_gap}, expected 1."
        )

    diverging_mask = y_true == 1
    high_mask = y_true == 0

    gap_diverging = y_gap_abs[diverging_mask]
    gap_high = y_gap_abs[high_mask]

    return {
        "model": model_name,
        "positive_class": "diverging_id_label == 1",
        "primary_predictor": "-abs(score_a - score_b)",
        "n_total": int(len(y_true)),
        "n_diverging": int(diverging_mask.sum()),
        "n_high_agreement": int(high_mask.sum()),
        "overall_auroc": auroc_primary,
        "raw_gap_auroc_sanity_check": auroc_raw_gap,
        "auroc_complement_error": complement_error,
        "mean_abs_gap_diverging": safe_float(
            np.mean(gap_diverging)
        ),
        "median_abs_gap_diverging": safe_float(
            np.median(gap_diverging)
        ),
        "mean_abs_gap_high_agreement": safe_float(
            np.mean(gap_high)
        ),
        "median_abs_gap_high_agreement": safe_float(
            np.median(gap_high)
        ),
        "mean_gap_difference_diverging_minus_high": safe_float(
            np.mean(gap_diverging) - np.mean(gap_high)
        ),
        "interpretation_direction": (
            "smaller reward gaps rank diverging examples higher"
            if auroc_primary > 0.5
            else (
                "larger reward gaps rank diverging examples higher"
                if auroc_primary < 0.5
                else "no ranking discrimination"
            )
        ),
    }


def main() -> None:
    print("Current working directory:")
    print(Path.cwd())
    print()

    require_file(
        INPUT_PATH,
        "Final labeled two-RM analysis table",
    )

    input_sha = sha256_file(INPUT_PATH)

    print("Input SHA-256:")
    print(input_sha)
    print()

    if input_sha != EXPECTED_INPUT_SHA256:
        raise ValueError(
            "Input SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_INPUT_SHA256}\n"
            f"Actual:   {input_sha}\n"
            "The labeled analysis table may have changed."
        )

    print("Input SHA-256 verified.")
    print()

    frame = pd.read_csv(INPUT_PATH)

    print("Loaded labeled analysis table.")
    print(f"Rows: {len(frame)}")
    print(f"Columns: {len(frame.columns)}")
    print()

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(frame)}."
        )

    required_columns = [LABEL_COLUMN]

    for columns in MODEL_COLUMNS.values():
        required_columns.extend(columns.values())

    require_columns(
        frame,
        required_columns,
    )

    labels = validate_labels(frame)

    print("Outcome labels verified.")
    print(f"Diverging:      {int((labels == 1).sum())}")
    print(f"High-agreement: {int((labels == 0).sum())}")
    print()

    results: list[dict[str, object]] = []

    for model_name, columns in MODEL_COLUMNS.items():
        (
            _score_a,
            _score_b,
            gap_abs,
            diverging_score,
        ) = verify_model_columns(
            frame=frame,
            model_name=model_name,
            columns=columns,
        )

        result = compute_model_result(
            labels=labels,
            gap_abs=gap_abs,
            diverging_score=diverging_score,
            model_name=model_name,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    results_sha = sha256_file(RESULTS_PATH)

    manifest = pd.DataFrame(
        [
            {
                "input_path": str(INPUT_PATH),
                "input_sha256": input_sha,
                "results_path": str(RESULTS_PATH),
                "results_sha256": results_sha,
                "rows_in_input": len(frame),
                "n_diverging": int((labels == 1).sum()),
                "n_high_agreement": int((labels == 0).sum()),
                "models_evaluated": json.dumps(
                    list(MODEL_COLUMNS.keys()),
                    ensure_ascii=False,
                ),
                "positive_class": "diverging_id_label == 1",
                "primary_predictor": "-abs(score_a - score_b)",
                "category_specific_auroc_computed": False,
                "bootstrap_ci_computed": False,
                "created_at_local": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
        ]
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("Derived score columns revalidated.")
    print()
    print("Overall AUROC results:")
    print()

    display_columns = [
        "model",
        "overall_auroc",
        "mean_abs_gap_diverging",
        "mean_abs_gap_high_agreement",
        "mean_gap_difference_diverging_minus_high",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
        )
    )
    print()

    print("Orientation sanity checks:")
    for row in results:
        print(
            f"{row['model']}: "
            f"AUC(-gap)={row['overall_auroc']:.12f}, "
            f"AUC(gap)={row['raw_gap_auroc_sanity_check']:.12f}, "
            f"sum={row['overall_auroc'] + row['raw_gap_auroc_sanity_check']:.12f}"
        )
    print()

    print("Saved overall AUROC results:")
    print(RESULTS_PATH)
    print()

    print("Results SHA-256:")
    print(results_sha)
    print()

    print("Saved manifest:")
    print(MANIFEST_PATH)
    print()

    print("=" * 72)
    print("Overall two-model AUROC analysis completed successfully.")
    print("No category-specific AUROC or bootstrap CI was computed.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("OVERALL AUROC ANALYSIS FAILED")
        print("=" * 72)
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)
