from __future__ import annotations

import hashlib
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 39_compute_overconfident_disagreement_rate_v1.py
#
# Purpose:
# 1. Operationalize "overconfident disagreement" using each
#    model's own reward-gap scale.
# 2. Define a high-gap threshold from the high-agreement group:
#       primary:    75th percentile
#       sensitivity: 50th and 90th percentiles
# 3. Compute the proportion of diverging samples whose absolute
#    reward gap is at or above each threshold.
# 4. Report overall, by strict-majority status, and within each
#    original sampling stratum.
#
# Operational definition:
#   overconfident disagreement =
#       diverging_id_label == 1
#       AND abs(score_a - score_b) >= model-specific threshold
#
# Notes:
# - Reward gaps are not calibrated probabilities.
# - Thresholds are therefore model-specific and relative to the
#   model's high-agreement reference distribution.
# - The 75th percentile is the primary exploratory threshold.
# - The 50th and 90th percentiles are sensitivity thresholds.
# - Wilson intervals quantify binomial rate uncertainty while
#   treating the empirical threshold as fixed; they do not include
#   threshold-estimation uncertainty.
# - No frozen input file is modified.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

ANALYSIS_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_two_rm_labeled_analysis_table.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_preference_predictions.csv"
)

THRESHOLDS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_overconfident_disagreement_thresholds.csv"
)

RATES_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_overconfident_disagreement_rates.csv"
)

ROW_FLAGS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_overconfident_disagreement_flags.csv"
)

MANIFEST_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_overconfident_disagreement_manifest.csv"
)

EXPECTED_ANALYSIS_SHA256 = (
    "d9447c490d613ee65b2618ed9025ca5e82227b925c87df2a5020d5d9b8da7f56"
)

EXPECTED_PREDICTIONS_SHA256 = (
    "84d09230ce900a66a491f42d8168f2ec242869bf9929daa6e2bbb97d5b48c95e"
)

EXPECTED_ROWS = 400
LABEL_COLUMN = "diverging_id_label"
STRATUM_COLUMN = "sampling_stratum"
HUMAN_PREFERENCE_COLUMN = "human_majority_preference"

QUANTILES = [
    (0.50, "sensitivity_q50"),
    (0.75, "primary_q75"),
    (0.90, "sensitivity_q90"),
]

MODEL_COLUMNS = {
    "Skywork-Reward-Llama-3.1-8B-v0.2": {
        "score_a": "skywork_score_a",
        "score_b": "skywork_score_b",
        "stored_gap": "skywork_score_gap_abs",
        "prediction": "skywork_predicted_preference",
        "short_name": "skywork",
    },
    "FsfairX-LLaMA3-RM-v0.1": {
        "score_a": "fsfairx_score_a",
        "score_b": "fsfairx_score_b",
        "stored_gap": "fsfairx_score_gap_abs",
        "prediction": "fsfairx_predicted_preference",
        "short_name": "fsfairx",
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

EXPECTED_DIVERGING_PER_STRATUM = {
    "Open QA": 40,
    "Generation": 35,
    "Coding": 35,
    "Chat": 35,
    "Brainstorm": 35,
    "Closed QA": 20,
}

WILSON_Z_95 = 1.959963984540054


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
            f"{column} contains invalid values at rows "
            f"{bad_rows[:20]}."
        )

    return values.astype(float)


def safe_float(value: float) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"Encountered non-finite value: {value}"
        )

    return value


def wilson_interval(
    successes: int,
    total: int,
    z: float = WILSON_Z_95,
) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)

    proportion = successes / total
    z_squared = z * z

    denominator = 1.0 + z_squared / total

    center = (
        proportion
        + z_squared / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            (
                proportion * (1.0 - proportion)
                + z_squared / (4.0 * total)
            )
            / total
        )
        / denominator
    )

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)

    return (
        safe_float(lower),
        safe_float(upper),
    )


def verify_id_pairs_and_order(
    analysis: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    required_ids = [
        "annotation_id",
        "comparison_id",
    ]

    require_columns(
        analysis,
        required_ids,
        "Analysis table",
    )
    require_columns(
        predictions,
        required_ids,
        "Prediction table",
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

    prediction_pairs = list(
        zip(
            normalize_id(
                predictions["annotation_id"]
            ).tolist(),
            normalize_id(
                predictions["comparison_id"]
            ).tolist(),
        )
    )

    if set(analysis_pairs) != set(prediction_pairs):
        missing = sorted(
            set(analysis_pairs)
            - set(prediction_pairs)
        )
        extra = sorted(
            set(prediction_pairs)
            - set(analysis_pairs)
        )

        raise ValueError(
            "Analysis and prediction tables contain "
            "different ID pairs.\n"
            f"Missing from predictions: {missing[:10]}\n"
            f"Extra in predictions:     {extra[:10]}"
        )

    if analysis_pairs != prediction_pairs:
        first = next(
            index
            for index, (left, right) in enumerate(
                zip(
                    analysis_pairs,
                    prediction_pairs,
                )
            )
            if left != right
        )

        raise ValueError(
            "Analysis and prediction tables have "
            "different row order.\n"
            f"First mismatch at zero-based row {first}."
        )


def validate_core_columns(
    analysis: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    analysis_required = [
        LABEL_COLUMN,
        STRATUM_COLUMN,
    ]

    prediction_required = [
        LABEL_COLUMN,
        STRATUM_COLUMN,
        HUMAN_PREFERENCE_COLUMN,
    ]

    for columns in MODEL_COLUMNS.values():
        analysis_required.extend(
            [
                columns["score_a"],
                columns["score_b"],
                columns["stored_gap"],
            ]
        )

        prediction_required.extend(
            [
                columns["score_a"],
                columns["score_b"],
                columns["prediction"],
            ]
        )

    require_columns(
        analysis,
        analysis_required,
        "Analysis table",
    )

    require_columns(
        predictions,
        prediction_required,
        "Prediction table",
    )

    analysis_labels = pd.to_numeric(
        analysis[LABEL_COLUMN],
        errors="coerce",
    )
    prediction_labels = pd.to_numeric(
        predictions[LABEL_COLUMN],
        errors="coerce",
    )

    if (
        analysis_labels.isna().any()
        or prediction_labels.isna().any()
        or not analysis_labels.isin([0, 1]).all()
        or not prediction_labels.isin([0, 1]).all()
    ):
        raise ValueError(
            "Invalid diverging_id_label values."
        )

    if not np.array_equal(
        analysis_labels.to_numpy(dtype=int),
        prediction_labels.to_numpy(dtype=int),
    ):
        raise ValueError(
            "diverging_id_label differs between tables."
        )

    if int((analysis_labels == 1).sum()) != 200:
        raise ValueError(
            "Expected 200 diverging rows."
        )

    if int((analysis_labels == 0).sum()) != 200:
        raise ValueError(
            "Expected 200 high-agreement rows."
        )

    analysis_strata = (
        analysis[STRATUM_COLUMN]
        .astype("string")
        .str.strip()
    )
    prediction_strata = (
        predictions[STRATUM_COLUMN]
        .astype("string")
        .str.strip()
    )

    if not analysis_strata.equals(
        prediction_strata
    ):
        raise ValueError(
            "sampling_stratum differs between tables."
        )

    actual_strata = set(
        analysis_strata.unique()
    )

    if actual_strata != set(STRATUM_ORDER):
        raise ValueError(
            "Unexpected sampling strata.\n"
            f"Expected: {sorted(STRATUM_ORDER)}\n"
            f"Actual:   {sorted(actual_strata)}"
        )

    diverging_mask = (
        analysis_labels.to_numpy(dtype=int)
        == 1
    )

    for stratum, expected_count in (
        EXPECTED_DIVERGING_PER_STRATUM.items()
    ):
        actual_count = int(
            (
                diverging_mask
                & (
                    analysis_strata.to_numpy()
                    == stratum
                )
            ).sum()
        )

        if actual_count != expected_count:
            raise ValueError(
                f"{stratum}: expected "
                f"{expected_count} diverging rows, "
                f"found {actual_count}."
            )

    allowed_human_preferences = {
        "A",
        "B",
        "no_strict_majority",
    }

    human_values = set(
        predictions[
            HUMAN_PREFERENCE_COLUMN
        ]
        .astype("string")
        .str.strip()
        .unique()
    )

    if not human_values.issubset(
        allowed_human_preferences
    ):
        raise ValueError(
            "Unexpected human majority preference values: "
            f"{sorted(human_values)}"
        )


def build_working_table(
    analysis: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    output = predictions[
        [
            "annotation_id",
            "comparison_id",
            STRATUM_COLUMN,
            LABEL_COLUMN,
            HUMAN_PREFERENCE_COLUMN,
        ]
    ].copy()

    output[LABEL_COLUMN] = pd.to_numeric(
        output[LABEL_COLUMN],
        errors="raise",
    ).astype("int64")

    output[STRATUM_COLUMN] = (
        output[STRATUM_COLUMN]
        .astype("string")
        .str.strip()
    )

    output[HUMAN_PREFERENCE_COLUMN] = (
        output[HUMAN_PREFERENCE_COLUMN]
        .astype("string")
        .str.strip()
    )

    for model_name, columns in (
        MODEL_COLUMNS.items()
    ):
        analysis_score_a = finite_numeric(
            analysis,
            columns["score_a"],
        )
        analysis_score_b = finite_numeric(
            analysis,
            columns["score_b"],
        )
        stored_gap = finite_numeric(
            analysis,
            columns["stored_gap"],
        )

        prediction_score_a = finite_numeric(
            predictions,
            columns["score_a"],
        )
        prediction_score_b = finite_numeric(
            predictions,
            columns["score_b"],
        )

        if not np.allclose(
            analysis_score_a.to_numpy(dtype=float),
            prediction_score_a.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{model_name}: score_a differs "
                "between input tables."
            )

        if not np.allclose(
            analysis_score_b.to_numpy(dtype=float),
            prediction_score_b.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{model_name}: score_b differs "
                "between input tables."
            )

        recomputed_gap = np.abs(
            analysis_score_a.to_numpy(dtype=float)
            - analysis_score_b.to_numpy(dtype=float)
        )

        if not np.allclose(
            stored_gap.to_numpy(dtype=float),
            recomputed_gap,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{model_name}: stored absolute gap "
                "does not match score_a and score_b."
            )

        prediction_column = columns[
            "prediction"
        ]

        model_predictions = (
            predictions[prediction_column]
            .astype("string")
            .str.strip()
        )

        allowed_model_preferences = {
            "A",
            "B",
            "model_tie",
        }

        if not set(
            model_predictions.unique()
        ).issubset(
            allowed_model_preferences
        ):
            raise ValueError(
                f"{model_name}: unexpected prediction values."
            )

        short_name = columns["short_name"]

        output[
            f"{short_name}_score_gap_abs"
        ] = recomputed_gap

        output[
            f"{short_name}_predicted_preference"
        ] = model_predictions.to_numpy()

    return output


def build_threshold_table(
    working: pd.DataFrame,
) -> pd.DataFrame:
    high_agreement = working.loc[
        working[LABEL_COLUMN] == 0
    ].copy()

    if len(high_agreement) != 200:
        raise ValueError(
            "High-agreement reference group must "
            "contain 200 rows."
        )

    rows: list[dict[str, object]] = []

    for model_name, columns in (
        MODEL_COLUMNS.items()
    ):
        short_name = columns["short_name"]
        gap_column = (
            f"{short_name}_score_gap_abs"
        )

        reference_gaps = high_agreement[
            gap_column
        ].to_numpy(dtype=float)

        for quantile, threshold_role in (
            QUANTILES
        ):
            threshold = safe_float(
                np.quantile(
                    reference_gaps,
                    quantile,
                    method="linear",
                )
            )

            exceedance_count = int(
                (
                    reference_gaps
                    >= threshold
                ).sum()
            )

            exceedance_rate = safe_float(
                exceedance_count
                / len(reference_gaps)
            )

            rows.append(
                {
                    "model": model_name,
                    "model_short_name": short_name,
                    "threshold_role": threshold_role,
                    "reference_quantile": quantile,
                    "threshold_gap": threshold,
                    "quantile_method": "linear",
                    "comparison_operator": (
                        "gap >= threshold"
                    ),
                    "n_high_agreement_reference": int(
                        len(reference_gaps)
                    ),
                    "n_high_agreement_at_or_above_threshold": (
                        exceedance_count
                    ),
                    "high_agreement_reference_exceedance_rate": (
                        exceedance_rate
                    ),
                    "threshold_interpretation": (
                        "model-specific empirical reward-gap "
                        "quantile among high-agreement samples"
                    ),
                }
            )

    return pd.DataFrame(rows)


def attach_flags(
    working: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    output = working.copy()

    for _, row in thresholds.iterrows():
        short_name = str(
            row["model_short_name"]
        )
        threshold_role = str(
            row["threshold_role"]
        )
        threshold = float(
            row["threshold_gap"]
        )

        gap_column = (
            f"{short_name}_score_gap_abs"
        )
        flag_column = (
            f"{short_name}_{threshold_role}_high_gap"
        )

        output[flag_column] = (
            output[gap_column]
            >= threshold
        )

    return output


def build_scope_definitions(
    frame: pd.DataFrame,
) -> list[
    tuple[str, str, pd.Series]
]:
    diverging = (
        frame[LABEL_COLUMN] == 1
    )

    strict_majority = (
        frame[HUMAN_PREFERENCE_COLUMN]
        != "no_strict_majority"
    )

    no_strict_majority = (
        frame[HUMAN_PREFERENCE_COLUMN]
        == "no_strict_majority"
    )

    scopes: list[
        tuple[str, str, pd.Series]
    ] = [
        (
            "overall",
            "All diverging",
            diverging,
        ),
        (
            "human_majority_status",
            "Diverging with strict majority",
            diverging & strict_majority,
        ),
        (
            "human_majority_status",
            "Diverging with no strict majority",
            diverging & no_strict_majority,
        ),
    ]

    for stratum in STRATUM_ORDER:
        scopes.append(
            (
                "sampling_stratum",
                stratum,
                diverging
                & (
                    frame[STRATUM_COLUMN]
                    == stratum
                ),
            )
        )

    return scopes


def build_rate_table(
    flagged: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    scopes = build_scope_definitions(
        flagged
    )

    for _, threshold_row in (
        thresholds.iterrows()
    ):
        model_name = str(
            threshold_row["model"]
        )
        short_name = str(
            threshold_row[
                "model_short_name"
            ]
        )
        threshold_role = str(
            threshold_row[
                "threshold_role"
            ]
        )
        quantile = float(
            threshold_row[
                "reference_quantile"
            ]
        )
        threshold = float(
            threshold_row[
                "threshold_gap"
            ]
        )

        flag_column = (
            f"{short_name}_{threshold_role}_high_gap"
        )
        prediction_column = (
            f"{short_name}_predicted_preference"
        )

        for (
            scope_type,
            scope_value,
            mask,
        ) in scopes:
            subset = flagged.loc[
                mask
            ].copy()

            n_total = int(
                len(subset)
            )

            if n_total == 0:
                raise ValueError(
                    f"Empty scope: "
                    f"{scope_type}/{scope_value}"
                )

            n_high_gap = int(
                subset[flag_column].sum()
            )

            rate = safe_float(
                n_high_gap / n_total
            )

            ci_lower, ci_upper = (
                wilson_interval(
                    n_high_gap,
                    n_total,
                )
            )

            strict = subset.loc[
                subset[
                    HUMAN_PREFERENCE_COLUMN
                ]
                != "no_strict_majority"
            ].copy()

            overconfident_strict = (
                strict.loc[
                    strict[flag_column]
                ]
            )

            n_strict = int(
                len(strict)
            )
            n_high_gap_strict = int(
                len(overconfident_strict)
            )

            if n_high_gap_strict > 0:
                correct_mask = (
                    overconfident_strict[
                        prediction_column
                    ]
                    == overconfident_strict[
                        HUMAN_PREFERENCE_COLUMN
                    ]
                )

                n_high_gap_correct = int(
                    correct_mask.sum()
                )
                n_high_gap_wrong = int(
                    (
                        ~correct_mask
                    ).sum()
                )

                wrong_fraction = safe_float(
                    n_high_gap_wrong
                    / n_high_gap_strict
                )
            else:
                n_high_gap_correct = 0
                n_high_gap_wrong = 0
                wrong_fraction = math.nan

            rows.append(
                {
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "model": model_name,
                    "model_short_name": short_name,
                    "threshold_role": threshold_role,
                    "reference_quantile": quantile,
                    "threshold_gap": threshold,
                    "n_diverging_in_scope": n_total,
                    "n_high_gap_diverging": n_high_gap,
                    "operational_overconfident_disagreement_rate": (
                        rate
                    ),
                    "wilson_95_ci_lower_fixed_threshold": (
                        ci_lower
                    ),
                    "wilson_95_ci_upper_fixed_threshold": (
                        ci_upper
                    ),
                    "n_strict_majority_in_scope": n_strict,
                    "n_high_gap_strict_majority": (
                        n_high_gap_strict
                    ),
                    "n_high_gap_and_majority_correct": (
                        n_high_gap_correct
                    ),
                    "n_high_gap_and_majority_wrong": (
                        n_high_gap_wrong
                    ),
                    "wrong_fraction_among_high_gap_strict_majority": (
                        wrong_fraction
                    ),
                    "rate_definition": (
                        "P(gap >= model-specific "
                        "high-agreement quantile threshold "
                        "| diverging)"
                    ),
                    "threshold_uncertainty_in_ci": False,
                    "exploratory": True,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    print("Current working directory:")
    print(Path.cwd())
    print()

    require_file(
        ANALYSIS_PATH,
        "Final labeled analysis table",
    )
    require_file(
        PREDICTIONS_PATH,
        "Preference prediction table",
    )

    analysis_sha = sha256_file(
        ANALYSIS_PATH
    )
    predictions_sha = sha256_file(
        PREDICTIONS_PATH
    )

    print("Input SHA-256 values:")
    print(f"Analysis:    {analysis_sha}")
    print(f"Predictions: {predictions_sha}")
    print()

    if (
        analysis_sha
        != EXPECTED_ANALYSIS_SHA256
    ):
        raise ValueError(
            "Analysis SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_ANALYSIS_SHA256}\n"
            f"Actual:   {analysis_sha}"
        )

    if (
        predictions_sha
        != EXPECTED_PREDICTIONS_SHA256
    ):
        raise ValueError(
            "Prediction SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_PREDICTIONS_SHA256}\n"
            f"Actual:   {predictions_sha}"
        )

    print("Both input hashes verified.")
    print()

    analysis = pd.read_csv(
        ANALYSIS_PATH
    )
    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    if len(analysis) != EXPECTED_ROWS:
        raise ValueError(
            f"Analysis must contain "
            f"{EXPECTED_ROWS} rows, "
            f"found {len(analysis)}."
        )

    if len(predictions) != EXPECTED_ROWS:
        raise ValueError(
            f"Predictions must contain "
            f"{EXPECTED_ROWS} rows, "
            f"found {len(predictions)}."
        )

    verify_id_pairs_and_order(
        analysis,
        predictions,
    )

    validate_core_columns(
        analysis,
        predictions,
    )

    print(
        "IDs, labels, strata, scores, and "
        "preference predictions verified."
    )
    print()

    working = build_working_table(
        analysis,
        predictions,
    )

    thresholds = build_threshold_table(
        working
    )

    flagged = attach_flags(
        working,
        thresholds,
    )

    rates = build_rate_table(
        flagged,
        thresholds,
    )

    THRESHOLDS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds.to_csv(
        THRESHOLDS_OUTPUT_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    rates.to_csv(
        RATES_OUTPUT_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    flagged.to_csv(
        ROW_FLAGS_OUTPUT_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    thresholds_sha = sha256_file(
        THRESHOLDS_OUTPUT_PATH
    )
    rates_sha = sha256_file(
        RATES_OUTPUT_PATH
    )
    flags_sha = sha256_file(
        ROW_FLAGS_OUTPUT_PATH
    )

    manifest = pd.DataFrame(
        [
            {
                "analysis_path": str(
                    ANALYSIS_PATH
                ),
                "analysis_sha256": analysis_sha,
                "predictions_path": str(
                    PREDICTIONS_PATH
                ),
                "predictions_sha256": (
                    predictions_sha
                ),
                "thresholds_output_path": str(
                    THRESHOLDS_OUTPUT_PATH
                ),
                "thresholds_output_sha256": (
                    thresholds_sha
                ),
                "rates_output_path": str(
                    RATES_OUTPUT_PATH
                ),
                "rates_output_sha256": rates_sha,
                "row_flags_output_path": str(
                    ROW_FLAGS_OUTPUT_PATH
                ),
                "row_flags_output_sha256": (
                    flags_sha
                ),
                "primary_threshold": (
                    "model-specific 75th percentile "
                    "of high-agreement absolute gaps"
                ),
                "sensitivity_thresholds": (
                    "model-specific 50th and 90th "
                    "percentiles of high-agreement "
                    "absolute gaps"
                ),
                "comparison_operator": (
                    "gap >= threshold"
                ),
                "quantile_method": "linear",
                "rate_ci_method": (
                    "Wilson 95% interval with "
                    "empirical threshold treated as fixed"
                ),
                "threshold_uncertainty_in_ci": False,
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

    print("Model-specific gap thresholds:")
    print()

    print(
        thresholds[
            [
                "model",
                "threshold_role",
                "reference_quantile",
                "threshold_gap",
                "high_agreement_reference_exceedance_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    print()

    primary = rates.loc[
        rates["threshold_role"]
        == "primary_q75"
    ].copy()

    print(
        "Primary operational overconfident-"
        "disagreement rates (Q75 threshold):"
    )
    print()

    print(
        primary[
            [
                "scope_type",
                "scope_value",
                "model",
                "n_diverging_in_scope",
                "n_high_gap_diverging",
                "operational_overconfident_disagreement_rate",
                "wilson_95_ci_lower_fixed_threshold",
                "wilson_95_ci_upper_fixed_threshold",
                "n_high_gap_and_majority_wrong",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    print()

    print("Saved thresholds:")
    print(THRESHOLDS_OUTPUT_PATH)
    print()
    print("Thresholds SHA-256:")
    print(thresholds_sha)
    print()

    print("Saved rate summary:")
    print(RATES_OUTPUT_PATH)
    print()
    print("Rates SHA-256:")
    print(rates_sha)
    print()

    print("Saved row-level flags:")
    print(ROW_FLAGS_OUTPUT_PATH)
    print()
    print("Flags SHA-256:")
    print(flags_sha)
    print()

    print("Saved manifest:")
    print(MANIFEST_OUTPUT_PATH)
    print()

    print("=" * 72)
    print(
        "Operational overconfident-disagreement "
        "analysis completed successfully."
    )
    print(
        "The planned metric-analysis stage is now complete."
    )
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print(
            "OVERCONFIDENT-DISAGREEMENT "
            "ANALYSIS FAILED"
        )
        print("=" * 72)
        print(
            f"{type(error).__name__}: {error}"
        )
        sys.exit(1)
