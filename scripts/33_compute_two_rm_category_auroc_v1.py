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
# 33_compute_two_rm_category_auroc_v1.py
#
# Purpose:
# 1. Verify the final labeled two-RM analysis table.
# 2. Compute AUROC separately within each of the six original
#    sampling strata.
# 3. Evaluate both Skywork and FsfairX using:
#       diverging_score = -abs(score_a - score_b)
# 4. Save a long-format category-by-model result table.
#
# IMPORTANT:
# - This script does NOT compute bootstrap confidence intervals.
# - It does NOT test whether the two models differ significantly.
# - It does NOT analyze manual task_scenario or
#   disagreement_source variables.
# - It does NOT modify any input file.
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
    / "formal_sample_v1_two_rm_category_auroc.csv"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_category_auroc_manifest.csv"
)

EXPECTED_INPUT_SHA256 = (
    "d9447c490d613ee65b2618ed9025ca5e82227b925c87df2a5020d5d9b8da7f56"
)

EXPECTED_ROWS = 400

LABEL_COLUMN = "diverging_id_label"
STRATUM_COLUMN = "sampling_stratum"

EXPECTED_QUOTAS_PER_CLASS = {
    "Open QA": 40,
    "Generation": 35,
    "Coding": 35,
    "Chat": 35,
    "Brainstorm": 35,
    "Closed QA": 20,
}

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


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def require_file(
    path: Path,
    description: str,
) -> None:
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
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    finite_mask = np.isfinite(
        values.to_numpy(dtype=float)
    )

    if not finite_mask.all():
        bad_rows = np.flatnonzero(
            ~finite_mask
        ).tolist()

        raise ValueError(
            f"{column} contains missing, non-numeric, "
            f"or infinite values. "
            f"Bad zero-based rows: {bad_rows[:20]}"
        )

    return values.astype(float)


def normalize_labels(
    frame: pd.DataFrame,
) -> pd.Series:
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
            f"{LABEL_COLUMN} contains non-numeric "
            f"values: {bad_values}"
        )

    if not labels.isin([0, 1]).all():
        bad_values = sorted(
            labels[
                ~labels.isin([0, 1])
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{LABEL_COLUMN} must contain only "
            f"0 and 1. Bad values: {bad_values}"
        )

    return labels.astype("int64")


def normalize_strata(
    frame: pd.DataFrame,
) -> pd.Series:
    strata = (
        frame[STRATUM_COLUMN]
        .astype("string")
        .str.strip()
    )

    if strata.isna().any() or strata.eq("").any():
        raise ValueError(
            f"{STRATUM_COLUMN} contains missing "
            "or empty values."
        )

    actual = set(strata.unique())
    expected = set(EXPECTED_QUOTAS_PER_CLASS)

    if actual != expected:
        raise ValueError(
            "Unexpected sampling strata.\n"
            f"Expected: {sorted(expected)}\n"
            f"Actual:   {sorted(actual)}"
        )

    return strata


def verify_model_columns(
    frame: pd.DataFrame,
    model_name: str,
    columns: dict[str, str],
) -> tuple[pd.Series, pd.Series]:
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
            f"{model_name}: stored gap does not "
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
            f"{model_name}: stored diverging score "
            f"does not equal -abs(score_a - score_b). "
            f"Maximum error: {max_error}"
        )

    return stored_gap, stored_diverging


def validate_category_quotas(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    counts = (
        frame
        .groupby(
            [STRATUM_COLUMN, LABEL_COLUMN],
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
            counts[STRATUM_COLUMN] == stratum
        ]

        if len(row) != 1:
            raise ValueError(
                f"Missing count row for stratum: "
                f"{stratum}"
            )

        actual_high = int(
            row.iloc[0]["n_high_agreement"]
        )
        actual_diverging = int(
            row.iloc[0]["n_diverging"]
        )

        if (
            actual_high != expected_per_class
            or actual_diverging
            != expected_per_class
        ):
            raise ValueError(
                f"{stratum}: expected "
                f"{expected_per_class}/"
                f"{expected_per_class}, found "
                f"{actual_diverging}/"
                f"{actual_high} "
                "(diverging/high-agreement)."
            )

    counts["n_total"] = (
        counts["n_high_agreement"]
        + counts["n_diverging"]
    )

    return counts


def safe_float(value: float) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"Encountered non-finite result: {value}"
        )

    return value


def compute_category_result(
    subset: pd.DataFrame,
    model_name: str,
    gap_column: str,
    diverging_score_column: str,
    stratum: str,
) -> dict[str, object]:
    labels = subset[
        LABEL_COLUMN
    ].to_numpy(dtype=int)

    gap = subset[
        gap_column
    ].to_numpy(dtype=float)

    diverging_score = subset[
        diverging_score_column
    ].to_numpy(dtype=float)

    unique_labels = np.unique(labels)

    if set(unique_labels.tolist()) != {0, 1}:
        raise ValueError(
            f"{stratum} / {model_name}: both label "
            "classes are required for AUROC."
        )

    auc_primary = safe_float(
        roc_auc_score(
            labels,
            diverging_score,
        )
    )

    auc_raw_gap = safe_float(
        roc_auc_score(
            labels,
            gap,
        )
    )

    complement_error = abs(
        auc_primary + auc_raw_gap - 1.0
    )

    if complement_error > 1e-12:
        raise ValueError(
            f"{stratum} / {model_name}: AUROC "
            "orientation sanity check failed."
        )

    diverging_gap = gap[labels == 1]
    high_gap = gap[labels == 0]

    return {
        "sampling_stratum": stratum,
        "model": model_name,
        "n_total": int(len(subset)),
        "n_diverging": int((labels == 1).sum()),
        "n_high_agreement": int((labels == 0).sum()),
        "category_auroc": auc_primary,
        "raw_gap_auroc_sanity_check": auc_raw_gap,
        "auroc_complement_error": complement_error,
        "mean_abs_gap_diverging": safe_float(
            np.mean(diverging_gap)
        ),
        "median_abs_gap_diverging": safe_float(
            np.median(diverging_gap)
        ),
        "mean_abs_gap_high_agreement": safe_float(
            np.mean(high_gap)
        ),
        "median_abs_gap_high_agreement": safe_float(
            np.median(high_gap)
        ),
        "mean_gap_difference_diverging_minus_high": (
            safe_float(
                np.mean(diverging_gap)
                - np.mean(high_gap)
            )
        ),
        "positive_class": (
            "diverging_id_label == 1"
        ),
        "predictor": (
            "-abs(score_a - score_b)"
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
            f"Actual:   {input_sha}"
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

    required_columns = [
        "annotation_id",
        "comparison_id",
        LABEL_COLUMN,
        STRATUM_COLUMN,
    ]

    for columns in MODEL_COLUMNS.values():
        required_columns.extend(
            columns.values()
        )

    require_columns(
        frame,
        required_columns,
    )

    frame[LABEL_COLUMN] = normalize_labels(
        frame
    )
    frame[STRATUM_COLUMN] = normalize_strata(
        frame
    )

    quota_table = validate_category_quotas(
        frame
    )

    print(
        "Six sampling-stratum quotas verified."
    )
    print()
    print(quota_table.to_string(index=False))
    print()

    validated_model_columns = {}

    for model_name, columns in (
        MODEL_COLUMNS.items()
    ):
        gap, diverging_score = (
            verify_model_columns(
                frame=frame,
                model_name=model_name,
                columns=columns,
            )
        )

        frame[columns["gap_abs"]] = (
            gap.to_numpy(dtype=float)
        )
        frame[columns["diverging_score"]] = (
            diverging_score.to_numpy(dtype=float)
        )

        validated_model_columns[
            model_name
        ] = columns

    print(
        "Both models' derived reward-gap "
        "columns revalidated."
    )
    print()

    results: list[dict[str, object]] = []

    for stratum in (
        EXPECTED_QUOTAS_PER_CLASS.keys()
    ):
        subset = (
            frame.loc[
                frame[STRATUM_COLUMN] == stratum
            ]
            .copy()
        )

        for model_name, columns in (
            validated_model_columns.items()
        ):
            result = compute_category_result(
                subset=subset,
                model_name=model_name,
                gap_column=columns["gap_abs"],
                diverging_score_column=(
                    columns["diverging_score"]
                ),
                stratum=stratum,
            )

            results.append(result)

    results_df = pd.DataFrame(results)

    results_df["category_auroc_rank_within_model"] = (
        results_df
        .groupby("model")["category_auroc"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    stratum_order = {
        stratum: index
        for index, stratum in enumerate(
            EXPECTED_QUOTAS_PER_CLASS.keys()
        )
    }

    model_order = {
        model: index
        for index, model in enumerate(
            MODEL_COLUMNS.keys()
        )
    }

    results_df["_stratum_order"] = (
        results_df["sampling_stratum"]
        .map(stratum_order)
    )
    results_df["_model_order"] = (
        results_df["model"]
        .map(model_order)
    )

    results_df = (
        results_df
        .sort_values(
            [
                "_stratum_order",
                "_model_order",
            ]
        )
        .drop(
            columns=[
                "_stratum_order",
                "_model_order",
            ]
        )
        .reset_index(drop=True)
    )

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

    results_sha = sha256_file(
        RESULTS_PATH
    )

    manifest = pd.DataFrame(
        [
            {
                "input_path": str(INPUT_PATH),
                "input_sha256": input_sha,
                "results_path": str(RESULTS_PATH),
                "results_sha256": results_sha,
                "rows_in_input": len(frame),
                "sampling_strata": json.dumps(
                    list(
                        EXPECTED_QUOTAS_PER_CLASS.keys()
                    ),
                    ensure_ascii=False,
                ),
                "models_evaluated": json.dumps(
                    list(MODEL_COLUMNS.keys()),
                    ensure_ascii=False,
                ),
                "result_rows": len(results_df),
                "positive_class": (
                    "diverging_id_label == 1"
                ),
                "predictor": (
                    "-abs(score_a - score_b)"
                ),
                "bootstrap_ci_computed": False,
                "model_difference_tested": False,
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

    print("Category-specific AUROC results:")
    print()

    pivot = (
        results_df
        .pivot(
            index="sampling_stratum",
            columns="model",
            values="category_auroc",
        )
        .reindex(
            list(
                EXPECTED_QUOTAS_PER_CLASS.keys()
            )
        )
    )

    print(
        pivot.to_string(
            float_format=lambda value: (
                f"{value:.6f}"
            )
        )
    )
    print()

    print("Saved category AUROC results:")
    print(RESULTS_PATH)
    print()

    print("Results SHA-256:")
    print(results_sha)
    print()

    print("Saved manifest:")
    print(MANIFEST_PATH)
    print()

    print("=" * 72)
    print(
        "Category-specific two-model AUROC "
        "analysis completed successfully."
    )
    print(
        "No bootstrap confidence intervals "
        "or model-difference tests were computed."
    )
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print(
            "CATEGORY AUROC ANALYSIS FAILED"
        )
        print("=" * 72)
        print(
            f"{type(error).__name__}: {error}"
        )
        sys.exit(1)
