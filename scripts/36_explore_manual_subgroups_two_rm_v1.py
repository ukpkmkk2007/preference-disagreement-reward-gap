from __future__ import annotations

import hashlib
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# 36_explore_manual_subgroups_two_rm_v1.py
#
# Purpose:
# 1. Verify the final labeled two-RM analysis table.
# 2. Summarize the frozen manual annotation variables:
#       - task_scenario
#       - disagreement_source
# 3. Compute exploratory subgroup AUROCs for both reward models
#    where both outcome classes have enough observations.
#
# IMPORTANT:
# - This is exploratory analysis.
# - No bootstrap confidence intervals are computed here.
# - No multiple-comparison adjustment is applied.
# - Small or one-class subgroups are reported but not assigned AUROC.
# - No input file is modified.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_two_rm_labeled_analysis_table.csv"
)

COUNTS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_manual_subgroup_counts.csv"
)

AUROC_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_manual_subgroup_auroc.csv"
)

MANIFEST_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_manual_subgroup_manifest.csv"
)

EXPECTED_INPUT_SHA256 = (
    "d9447c490d613ee65b2618ed9025ca5e82227b925c87df2a5020d5d9b8da7f56"
)

EXPECTED_ROWS = 400
LABEL_COLUMN = "diverging_id_label"

ID_COLUMN = "annotation_id"
TASK_SCENARIO_COLUMN = "task_scenario"

DISAGREEMENT_SOURCE_COLUMNS = [
    "disagreement_source_1",
    "disagreement_source_2",
    "disagreement_source_3",
]
MODEL_SCORE_COLUMNS = {
    "Skywork-Reward-Llama-3.1-8B-v0.2":
        "skywork_diverging_score",
    "FsfairX-LLaMA3-RM-v0.1":
        "fsfairx_diverging_score",
}

MIN_PER_CLASS_FOR_AUROC = 10


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


def safe_float(value: float) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"Encountered non-finite value: {value}"
        )

    return value





def normalize_labels(frame: pd.DataFrame) -> pd.Series:
    labels = pd.to_numeric(
        frame[LABEL_COLUMN],
        errors="coerce",
    )

    if labels.isna().any():
        raise ValueError(
            f"{LABEL_COLUMN} contains missing or non-numeric values."
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
            f"{LABEL_COLUMN} must contain only 0 and 1. "
            f"Bad values: {bad_values}"
        )

    return labels.astype("int64")


def normalize_group_values(
    series: pd.Series,
    logical_name: str,
) -> pd.Series:
    values = (
        series
        .astype("string")
        .str.strip()
    )

    missing = values.isna() | values.eq("")

    if missing.any():
        count = int(missing.sum())
        raise ValueError(
            f"{logical_name} contains {count} missing or empty values."
        )

    return values


def normalize_scores(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    for model_name, score_column in (
        MODEL_SCORE_COLUMNS.items()
    ):
        if score_column not in result.columns:
            raise ValueError(
                f"Missing score column for {model_name}: "
                f"{score_column}"
            )

        values = pd.to_numeric(
            result[score_column],
            errors="coerce",
        )

        finite = np.isfinite(
            values.to_numpy(dtype=float)
        )

        if not finite.all():
            bad_rows = np.flatnonzero(
                ~finite
            ).tolist()

            raise ValueError(
                f"{score_column} contains invalid values "
                f"at rows {bad_rows[:20]}."
            )

        result[score_column] = values.astype(float)

    return result


def build_subgroup_long(frame: pd.DataFrame) -> pd.DataFrame:
    score_columns = list(MODEL_SCORE_COLUMNS.values())

    # task_scenario：单标签
    task_long = frame[
        [
            ID_COLUMN,
            LABEL_COLUMN,
            TASK_SCENARIO_COLUMN,
            *score_columns,
        ]
    ].copy()

    task_long = task_long.rename(
        columns={TASK_SCENARIO_COLUMN: "group_value"}
    )
    task_long["grouping_variable"] = "task_scenario"
    task_long["source_column"] = TASK_SCENARIO_COLUMN
    task_long["multi_label_grouping"] = False

    # disagreement_source：三个多标签列展开
    source_long = frame[
        [
            ID_COLUMN,
            LABEL_COLUMN,
            *score_columns,
            *DISAGREEMENT_SOURCE_COLUMNS,
        ]
    ].melt(
        id_vars=[
            ID_COLUMN,
            LABEL_COLUMN,
            *score_columns,
        ],
        value_vars=DISAGREEMENT_SOURCE_COLUMNS,
        var_name="original_source_column",
        value_name="group_value",
    )

    source_long["group_value"] = (
        source_long["group_value"]
        .astype("string")
        .str.strip()
    )

    invalid = (
        source_long["group_value"].isna()
        | source_long["group_value"].eq("")
        | source_long["group_value"].str.lower().isin(
            ["nan", "none", "null", "n/a", "na"]
        )
    )

    source_long = source_long.loc[~invalid].copy()

    # 同一样本重复出现同一来源时只计算一次
    source_long = source_long.drop_duplicates(
        subset=[ID_COLUMN, "group_value"]
    )

    source_long["grouping_variable"] = "disagreement_source"
    source_long["source_column"] = (
        "disagreement_source_1|"
        "disagreement_source_2|"
        "disagreement_source_3"
    )
    source_long["multi_label_grouping"] = True

    common_columns = [
        ID_COLUMN,
        LABEL_COLUMN,
        *score_columns,
        "grouping_variable",
        "source_column",
        "group_value",
        "multi_label_grouping",
    ]

    return pd.concat(
        [
            task_long[common_columns],
            source_long[common_columns],
        ],
        ignore_index=True,
    )


def build_counts_table(
    subgroup_long: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "grouping_variable",
        "source_column",
        "group_value",
        "multi_label_grouping",
    ]

    for keys, subset in subgroup_long.groupby(
        group_columns,
        observed=False,
        sort=True,
    ):
        grouping_variable, source_column, group_value, is_multilabel = keys

        subset = subset.drop_duplicates(subset=[ID_COLUMN])

        n_diverging = int(
            (subset[LABEL_COLUMN] == 1).sum()
        )
        n_high = int(
            (subset[LABEL_COLUMN] == 0).sum()
        )

        rows.append(
            {
                "grouping_variable": grouping_variable,
                "source_column": source_column,
                "group_value": str(group_value),
                "multi_label_grouping": bool(is_multilabel),
                "n_total": int(len(subset)),
                "n_diverging": n_diverging,
                "n_high_agreement": n_high,
                "eligible_for_auroc": (
                    n_diverging >= MIN_PER_CLASS_FOR_AUROC
                    and n_high >= MIN_PER_CLASS_FOR_AUROC
                ),
                "minimum_required_per_class":
                    MIN_PER_CLASS_FOR_AUROC,
            }
        )

    return pd.DataFrame(rows)


def build_auroc_table(
    subgroup_long: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "grouping_variable",
        "source_column",
        "group_value",
        "multi_label_grouping",
    ]

    for keys, subset in subgroup_long.groupby(
        group_columns,
        observed=False,
        sort=True,
    ):
        grouping_variable, source_column, group_value, is_multilabel = keys

        subset = subset.drop_duplicates(subset=[ID_COLUMN]).copy()
        labels = subset[LABEL_COLUMN].to_numpy(dtype=int)

        n_diverging = int((labels == 1).sum())
        n_high = int((labels == 0).sum())

        eligible = (
            n_diverging >= MIN_PER_CLASS_FOR_AUROC
            and n_high >= MIN_PER_CLASS_FOR_AUROC
        )

        for model_name, score_column in MODEL_SCORE_COLUMNS.items():
            if eligible:
                auroc_value = float(
                    roc_auc_score(
                        labels,
                        subset[score_column].to_numpy(dtype=float),
                    )
                )
                status = "computed"
            else:
                auroc_value = np.nan
                status = (
                    "not computed: fewer than "
                    f"{MIN_PER_CLASS_FOR_AUROC} "
                    "observations in at least one class"
                )

            rows.append(
                {
                    "grouping_variable": grouping_variable,
                    "source_column": source_column,
                    "group_value": str(group_value),
                    "multi_label_grouping": bool(is_multilabel),
                    "model": model_name,
                    "n_total": int(len(subset)),
                    "n_diverging": n_diverging,
                    "n_high_agreement": n_high,
                    "subgroup_auroc": auroc_value,
                    "status": status,
                    "positive_class": "diverging_id_label == 1",
                    "predictor": "-abs(score_a - score_b)",
                    "exploratory": True,
                    "multiple_comparison_adjusted": False,
                }
            )

    return pd.DataFrame(rows)


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

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(frame)}."
        )

    if LABEL_COLUMN not in frame.columns:
        raise ValueError(
            f"Missing label column: {LABEL_COLUMN}"
        )

    frame[LABEL_COLUMN] = normalize_labels(
        frame
    )

    frame = normalize_scores(frame)

    required_manual_columns = [
    ID_COLUMN,
    TASK_SCENARIO_COLUMN,
    *DISAGREEMENT_SOURCE_COLUMNS,
    ]

    missing_manual_columns = [
        column
        for column in required_manual_columns
        if column not in frame.columns
    ]
    
    if missing_manual_columns:
        raise ValueError(
            "Missing manual annotation columns:\n- "
            + "\n- ".join(missing_manual_columns)
        )
    
    frame[TASK_SCENARIO_COLUMN] = normalize_group_values(
        frame[TASK_SCENARIO_COLUMN],
        TASK_SCENARIO_COLUMN,
    )
    
    print("Detected manual subgroup structure:")
    print(f"Task scenario: {TASK_SCENARIO_COLUMN}")
    print(
      "Disagreement sources: "
        + ", ".join(DISAGREEMENT_SOURCE_COLUMNS)
    )
    print()

    subgroup_long = build_subgroup_long(frame)
    
    counts = build_counts_table(
        subgroup_long
    )

    auroc_results = build_auroc_table(
        subgroup_long
        )

    COUNTS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts.to_csv(
        COUNTS_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    auroc_results.to_csv(
        AUROC_OUTPUT_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    counts_sha = sha256_file(
        COUNTS_OUTPUT_PATH
    )
    auroc_sha = sha256_file(
        AUROC_OUTPUT_PATH
    )

    manifest = pd.DataFrame(
        [
            {
                "input_path": str(INPUT_PATH),
                "input_sha256": input_sha,
                "task_scenario_column": TASK_SCENARIO_COLUMN,
                "disagreement_source_columns": "|".join(
                    DISAGREEMENT_SOURCE_COLUMNS
                    ),
                "disagreement_source_is_multilabel": True,
                "counts_output_path": str(
                    COUNTS_OUTPUT_PATH
                ),
                "counts_output_sha256": counts_sha,
                "auroc_output_path": str(
                    AUROC_OUTPUT_PATH
                ),
                "auroc_output_sha256": auroc_sha,
                "minimum_per_class_for_auroc": (
                    MIN_PER_CLASS_FOR_AUROC
                ),
                "bootstrap_ci_computed": False,
                "multiple_comparison_adjusted": False,
                "exploratory": True,
                "created_at_local": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
            }
        ]
    )

    manifest.to_csv(
        MANIFEST_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("Manual subgroup counts:")
    print()
    print(
        counts.to_string(
            index=False,
        )
    )
    print()

    print("Eligible subgroup AUROCs:")
    print()

    eligible_results = auroc_results.loc[
        auroc_results["status"]
        == "computed"
    ]

    if eligible_results.empty:
        print(
            "No subgroup met the minimum "
            "per-class threshold."
        )
    else:
        print(
            eligible_results[
                [
                    "grouping_variable",
                    "group_value",
                    "model",
                    "n_diverging",
                    "n_high_agreement",
                    "subgroup_auroc",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

    print()
    print("Saved subgroup counts:")
    print(COUNTS_OUTPUT_PATH)
    print()
    print("Counts SHA-256:")
    print(counts_sha)
    print()

    print("Saved subgroup AUROC results:")
    print(AUROC_OUTPUT_PATH)
    print()
    print("AUROC results SHA-256:")
    print(auroc_sha)
    print()

    print("Saved manifest:")
    print(MANIFEST_OUTPUT_PATH)
    print()

    print("=" * 72)
    print(
        "Manual subgroup exploratory analysis "
        "completed successfully."
    )
    print(
        "No bootstrap CIs or multiplicity-adjusted "
        "tests were computed."
    )
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print(
            "MANUAL SUBGROUP ANALYSIS FAILED"
        )
        print("=" * 72)
        print(
            f"{type(error).__name__}: {error}"
        )
        sys.exit(1)
