from __future__ import annotations

import hashlib
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 38_compute_two_rm_preference_accuracy_v1.py
#
# Purpose:
# 1. Verify the fixed master table and final labeled analysis table.
# 2. Derive human strict-majority preference from n_A versus n_B.
# 3. Compute preference accuracy for Skywork and FsfairX.
# 4. Report results overall, by disagreement class, and by the
#    six original sampling strata.
#
# Human target:
# - A if n_A > n_B
# - B if n_B > n_A
# - no_strict_majority if n_A == n_B
#
# Model prediction:
# - A if score_a > score_b
# - B if score_b > score_a
# - model_tie if score_a == score_b
#
# Primary accuracy:
# - denominator: rows with a strict human majority
# - model ties count as incorrect
#
# Sensitivity accuracy:
# - excludes model ties from the denominator
#
# IMPORTANT:
# - This script does not define or compute
#   overconfident-disagreement rate.
# - It does not modify any input file.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

MASTER_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_stratified_master.csv"
)

ANALYSIS_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_two_rm_labeled_analysis_table.csv"
)

ROW_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_preference_predictions.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_preference_accuracy.csv"
)

MANIFEST_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_preference_accuracy_manifest.csv"
)

EXPECTED_MASTER_SHA256 = (
    "02bd5783ddd028b6ba849f02d1af28a5b51748320c387906fcdbbb439a7d4a60"
)

EXPECTED_ANALYSIS_SHA256 = (
    "d9447c490d613ee65b2618ed9025ca5e82227b925c87df2a5020d5d9b8da7f56"
)

EXPECTED_ROWS = 400

ID_COLUMNS = [
    "annotation_id",
    "comparison_id",
]

MODEL_COLUMNS = {
    "Skywork-Reward-Llama-3.1-8B-v0.2": {
        "score_a": "skywork_score_a",
        "score_b": "skywork_score_b",
        "prediction_column": "skywork_predicted_preference",
        "correct_column": "skywork_preference_correct",
    },
    "FsfairX-LLaMA3-RM-v0.1": {
        "score_a": "fsfairx_score_a",
        "score_b": "fsfairx_score_b",
        "prediction_column": "fsfairx_predicted_preference",
        "correct_column": "fsfairx_preference_correct",
    },
}

STRATUM_ORDER = [
    "Open QA",
    "Generation",
    "Coding",
    "Chat",
    "Brainstorm",
    "Closed QA",
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


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    table_name: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns:\n- "
            + "\n- ".join(missing)
        )


def normalize_id(series: pd.Series) -> pd.Series:
    values = (
        series
        .astype("string")
        .str.strip()
    )

    if values.isna().any() or values.eq("").any():
        raise ValueError(
            f"{series.name} contains missing or empty IDs."
        )

    return values


def finite_numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = pd.to_numeric(
        frame[column],
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
            f"{column} contains missing, non-numeric, "
            f"or infinite values at rows {bad_rows[:20]}."
        )

    return values.astype(float)


def nonnegative_integer(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    if values.isna().any():
        raise ValueError(
            f"{column} contains missing or non-numeric values."
        )

    if not np.all(
        np.equal(
            values.to_numpy(dtype=float),
            np.floor(
                values.to_numpy(dtype=float)
            ),
        )
    ):
        raise ValueError(
            f"{column} contains non-integer values."
        )

    if (values < 0).any():
        raise ValueError(
            f"{column} contains negative values."
        )

    return values.astype("int64")


def verify_id_pairs_and_order(
    master: pd.DataFrame,
    analysis: pd.DataFrame,
) -> None:
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
            "Master and analysis have different row order.\n"
            f"First mismatch at zero-based row {first}:\n"
            f"Master:   {master_pairs[first]}\n"
            f"Analysis: {analysis_pairs[first]}"
        )


def derive_human_preference(
    n_a: pd.Series,
    n_b: pd.Series,
) -> pd.Series:
    result = np.where(
        n_a > n_b,
        "A",
        np.where(
            n_b > n_a,
            "B",
            "no_strict_majority",
        ),
    )

    return pd.Series(
        result,
        index=n_a.index,
        dtype="string",
        name="human_majority_preference",
    )


def derive_model_preference(
    score_a: pd.Series,
    score_b: pd.Series,
    name: str,
) -> pd.Series:
    result = np.where(
        score_a > score_b,
        "A",
        np.where(
            score_b > score_a,
            "B",
            "model_tie",
        ),
    )

    return pd.Series(
        result,
        index=score_a.index,
        dtype="string",
        name=name,
    )


def safe_float(value: float) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"Encountered non-finite result: {value}"
        )

    return value


def summarize_scope(
    frame: pd.DataFrame,
    scope_type: str,
    scope_value: str,
    model_name: str,
    prediction_column: str,
) -> dict[str, object]:
    strict = frame.loc[
        frame["human_majority_preference"]
        != "no_strict_majority"
    ].copy()

    n_total_scope = int(len(frame))
    n_no_strict_majority = int(
        (
            frame["human_majority_preference"]
            == "no_strict_majority"
        ).sum()
    )
    n_strict_majority = int(len(strict))

    if n_strict_majority == 0:
        primary_accuracy = np.nan
        conditional_accuracy = np.nan
        n_model_ties = 0
        n_correct = 0
        n_non_tied_model_predictions = 0
    else:
        model_tie_mask = (
            strict[prediction_column]
            == "model_tie"
        )
        correct_mask = (
            strict[prediction_column]
            == strict[
                "human_majority_preference"
            ]
        )

        n_model_ties = int(
            model_tie_mask.sum()
        )
        n_correct = int(
            correct_mask.sum()
        )
        n_non_tied_model_predictions = int(
            (~model_tie_mask).sum()
        )

        primary_accuracy = safe_float(
            n_correct
            / n_strict_majority
        )

        if n_non_tied_model_predictions > 0:
            conditional_accuracy = safe_float(
                n_correct
                / n_non_tied_model_predictions
            )
        else:
            conditional_accuracy = np.nan

    return {
        "scope_type": scope_type,
        "scope_value": scope_value,
        "model": model_name,
        "n_total_scope": n_total_scope,
        "n_human_strict_majority": n_strict_majority,
        "n_human_no_strict_majority": n_no_strict_majority,
        "n_model_ties_among_strict_majority": n_model_ties,
        "n_non_tied_model_predictions": (
            n_non_tied_model_predictions
        ),
        "n_correct": n_correct,
        "preference_accuracy_primary": primary_accuracy,
        "preference_accuracy_excluding_model_ties": (
            conditional_accuracy
        ),
        "primary_denominator_rule": (
            "human strict-majority rows; model ties incorrect"
        ),
        "sensitivity_denominator_rule": (
            "human strict-majority rows; model ties excluded"
        ),
    }


def main() -> None:
    print("Current working directory:")
    print(Path.cwd())
    print()

    require_file(
        MASTER_PATH,
        "Stratified master table",
    )
    require_file(
        ANALYSIS_PATH,
        "Final labeled analysis table",
    )

    master_sha = sha256_file(MASTER_PATH)
    analysis_sha = sha256_file(ANALYSIS_PATH)

    print("Input SHA-256 values:")
    print(f"Master:   {master_sha}")
    print(f"Analysis: {analysis_sha}")
    print()

    if master_sha != EXPECTED_MASTER_SHA256:
        raise ValueError(
            "Master SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_MASTER_SHA256}\n"
            f"Actual:   {master_sha}"
        )

    if analysis_sha != EXPECTED_ANALYSIS_SHA256:
        raise ValueError(
            "Analysis SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_ANALYSIS_SHA256}\n"
            f"Actual:   {analysis_sha}"
        )

    print("Both input hashes verified.")
    print()

    master = pd.read_csv(MASTER_PATH)
    analysis = pd.read_csv(ANALYSIS_PATH)

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

    master_required = [
        *ID_COLUMNS,
        "n_A",
        "n_B",
        "n_Tie",
        "n_Invalid",
        "n_valid",
        "diverging_id_label",
        "sampling_stratum",
    ]

    analysis_required = [
        *ID_COLUMNS,
        "diverging_id_label",
        "sampling_stratum",
    ]

    for columns in MODEL_COLUMNS.values():
        analysis_required.extend(
            [
                columns["score_a"],
                columns["score_b"],
            ]
        )

    require_columns(
        master,
        master_required,
        "Master table",
    )
    require_columns(
        analysis,
        analysis_required,
        "Analysis table",
    )

    verify_id_pairs_and_order(
        master,
        analysis,
    )

    print("ID pairs and row order verified.")
    print()

    for column in (
        "n_A",
        "n_B",
        "n_Tie",
        "n_Invalid",
        "n_valid",
    ):
        master[column] = nonnegative_integer(
            master,
            column,
        )

    if not (
        master["n_A"]
        + master["n_B"]
        + master["n_Tie"]
        == master["n_valid"]
    ).all():
        bad_rows = np.flatnonzero(
            (
                master["n_A"]
                + master["n_B"]
                + master["n_Tie"]
                != master["n_valid"]
            ).to_numpy()
        ).tolist()

        raise ValueError(
            "n_A + n_B + n_Tie does not equal n_valid "
            f"at rows {bad_rows[:20]}."
        )

    master_label = pd.to_numeric(
        master["diverging_id_label"],
        errors="coerce",
    )
    analysis_label = pd.to_numeric(
        analysis["diverging_id_label"],
        errors="coerce",
    )

    if (
        master_label.isna().any()
        or analysis_label.isna().any()
        or not master_label.isin([0, 1]).all()
        or not analysis_label.isin([0, 1]).all()
    ):
        raise ValueError(
            "Invalid diverging_id_label values."
        )

    if not np.array_equal(
        master_label.to_numpy(dtype=int),
        analysis_label.to_numpy(dtype=int),
    ):
        raise ValueError(
            "diverging_id_label differs between "
            "master and analysis tables."
        )

    master_stratum = (
        master["sampling_stratum"]
        .astype("string")
        .str.strip()
    )
    analysis_stratum = (
        analysis["sampling_stratum"]
        .astype("string")
        .str.strip()
    )

    if not master_stratum.equals(
        analysis_stratum
    ):
        raise ValueError(
            "sampling_stratum differs between "
            "master and analysis tables."
        )

    print(
        "Vote counts, labels, and sampling strata verified."
    )
    print()

    output = analysis[
        [
            *ID_COLUMNS,
            "sampling_stratum",
            "diverging_id_label",
        ]
    ].copy()

    output["n_A"] = master["n_A"].to_numpy()
    output["n_B"] = master["n_B"].to_numpy()
    output["n_Tie"] = master["n_Tie"].to_numpy()
    output["n_Invalid"] = master[
        "n_Invalid"
    ].to_numpy()
    output["n_valid"] = master["n_valid"].to_numpy()

    output["human_majority_preference"] = (
        derive_human_preference(
            master["n_A"],
            master["n_B"],
        )
    )

    output["human_vote_margin_abs"] = np.abs(
        master["n_A"].to_numpy(dtype=int)
        - master["n_B"].to_numpy(dtype=int)
    )

    for model_name, columns in (
        MODEL_COLUMNS.items()
    ):
        score_a = finite_numeric(
            analysis,
            columns["score_a"],
        )
        score_b = finite_numeric(
            analysis,
            columns["score_b"],
        )

        output[columns["score_a"]] = (
            score_a.to_numpy(dtype=float)
        )
        output[columns["score_b"]] = (
            score_b.to_numpy(dtype=float)
        )

        output[
            columns["prediction_column"]
        ] = derive_model_preference(
            score_a,
            score_b,
            columns["prediction_column"],
        )

        strict_mask = (
            output["human_majority_preference"]
            != "no_strict_majority"
        )

        output[
            columns["correct_column"]
        ] = pd.Series(
            pd.NA,
            index=output.index,
            dtype="boolean",
        )

        output.loc[
            strict_mask,
            columns["correct_column"],
        ] = (
            output.loc[
                strict_mask,
                columns["prediction_column"],
            ]
            == output.loc[
                strict_mask,
                "human_majority_preference",
            ]
        ).to_numpy()

    n_a_majority = int(
        (
            output["human_majority_preference"]
            == "A"
        ).sum()
    )
    n_b_majority = int(
        (
            output["human_majority_preference"]
            == "B"
        ).sum()
    )
    n_no_majority = int(
        (
            output["human_majority_preference"]
            == "no_strict_majority"
        ).sum()
    )

    print("Human majority preference counts:")
    print(f"A majority:          {n_a_majority}")
    print(f"B majority:          {n_b_majority}")
    print(f"No strict majority:  {n_no_majority}")
    print()

    summary_rows: list[dict[str, object]] = []

    scopes: list[
        tuple[str, str, pd.Series]
    ] = [
        (
            "overall",
            "Overall",
            pd.Series(
                True,
                index=output.index,
            ),
        ),
        (
            "disagreement_class",
            "Diverging",
            output["diverging_id_label"] == 1,
        ),
        (
            "disagreement_class",
            "High-agreement",
            output["diverging_id_label"] == 0,
        ),
    ]

    for stratum in STRATUM_ORDER:
        scopes.append(
            (
                "sampling_stratum",
                stratum,
                output["sampling_stratum"] == stratum,
            )
        )

    for (
        scope_type,
        scope_value,
        mask,
    ) in scopes:
        subset = output.loc[mask].copy()

        for model_name, columns in (
            MODEL_COLUMNS.items()
        ):
            summary_rows.append(
                summarize_scope(
                    subset,
                    scope_type,
                    scope_value,
                    model_name,
                    columns["prediction_column"],
                )
            )

    summary = pd.DataFrame(
        summary_rows
    )

    ROW_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        ROW_OUTPUT_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    row_sha = sha256_file(
        ROW_OUTPUT_PATH
    )
    summary_sha = sha256_file(
        SUMMARY_OUTPUT_PATH
    )

    manifest = pd.DataFrame(
        [
            {
                "master_path": str(MASTER_PATH),
                "master_sha256": master_sha,
                "analysis_path": str(ANALYSIS_PATH),
                "analysis_sha256": analysis_sha,
                "row_output_path": str(
                    ROW_OUTPUT_PATH
                ),
                "row_output_sha256": row_sha,
                "summary_output_path": str(
                    SUMMARY_OUTPUT_PATH
                ),
                "summary_output_sha256": summary_sha,
                "human_target_definition": (
                    "A if n_A>n_B; B if n_B>n_A; "
                    "no strict majority if equal"
                ),
                "primary_accuracy_definition": (
                    "strict human-majority rows; "
                    "model ties count incorrect"
                ),
                "n_a_majority": n_a_majority,
                "n_b_majority": n_b_majority,
                "n_no_strict_majority": n_no_majority,
                "overconfident_disagreement_computed": False,
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

    print("Preference accuracy summary:")
    print()

    print(
        summary[
            [
                "scope_type",
                "scope_value",
                "model",
                "n_human_strict_majority",
                "n_human_no_strict_majority",
                "n_model_ties_among_strict_majority",
                "preference_accuracy_primary",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    print()

    print("Saved row-level predictions:")
    print(ROW_OUTPUT_PATH)
    print()
    print("Row output SHA-256:")
    print(row_sha)
    print()

    print("Saved preference accuracy summary:")
    print(SUMMARY_OUTPUT_PATH)
    print()
    print("Summary SHA-256:")
    print(summary_sha)
    print()

    print("Saved manifest:")
    print(MANIFEST_OUTPUT_PATH)
    print()

    print("=" * 72)
    print(
        "Two-model preference accuracy analysis "
        "completed successfully."
    )
    print(
        "Overconfident-disagreement rate was not computed."
    )
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print(
            "PREFERENCE ACCURACY ANALYSIS FAILED"
        )
        print("=" * 72)
        print(
            f"{type(error).__name__}: {error}"
        )
        sys.exit(1)
