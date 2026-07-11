from __future__ import annotations

import hashlib
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# 35_compare_two_rm_auroc_v1.py
#
# Purpose:
# 1. Use the paired bootstrap replicates produced by Step 34.
# 2. Estimate 95% percentile CIs for:
#       FsfairX AUROC - Skywork AUROC
#    overall and within each of the six original categories.
# 3. Summarize cross-category consistency between the two models.
#
# IMPORTANT:
# - This script reuses the existing 2,000 bootstrap replicates.
# - It does not rerun bootstrap.
# - It does not use overlap between separate model CIs as a test.
# - Category-level results are exploratory and unadjusted for
#   multiple comparisons.
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

OVERALL_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_overall_auroc.csv"
)

CATEGORY_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_category_auroc.csv"
)

BOOTSTRAP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_bootstrap_auroc_summary.csv"
)

BOOTSTRAP_REPLICATES_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_bootstrap_auroc_replicates.csv"
)

DIFFERENCE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_paired_auroc_differences.csv"
)

CONSISTENCY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_category_consistency.csv"
)

MANIFEST_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_comparison_manifest.csv"
)

EXPECTED_HASHES = {
    "overall": (
        "0fdf5b39aa11f3247c32bfddfedce710bf02566a1af8fc95e2e3006557a7ebd5"
    ),
    "category": (
        "7eddb14e687a49521874d1d72063c2ea982b1507ca2b71a1422569424c215d26"
    ),
    "bootstrap_summary": (
        "5c319909aeccca2872db2b7cd835d65c3f3f356685a3a180117aa1c5b35cde02"
    ),
    "bootstrap_replicates": (
        "cbc705cf2a1e565f8c31a69e73e02958137fcde482336c45c162b10d3b3b55d5"
    ),
}

SKYWORK_MODEL = "Skywork-Reward-Llama-3.1-8B-v0.2"
FSFAIRX_MODEL = "FsfairX-LLaMA3-RM-v0.1"

DIFFERENCE_COLUMN = "fsfairx_minus_skywork_auroc"

SCOPES = [
    "Overall",
    "Open QA",
    "Generation",
    "Coding",
    "Chat",
    "Brainstorm",
    "Closed QA",
]

CATEGORY_SCOPES = SCOPES[1:]

N_BOOTSTRAP = 2000
CI_LEVEL = 0.95


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
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


def safe_float(value: float) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"Encountered non-finite value: {value}"
        )

    return value


def observed_auroc_by_scope(
    overall: pd.DataFrame,
    category: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        overall,
        ["model", "overall_auroc"],
        "Overall AUROC table",
    )

    require_columns(
        category,
        [
            "sampling_stratum",
            "model",
            "category_auroc",
        ],
        "Category AUROC table",
    )

    rows: list[dict[str, object]] = []

    for model in (
        SKYWORK_MODEL,
        FSFAIRX_MODEL,
    ):
        match = overall.loc[
            overall["model"] == model,
            "overall_auroc",
        ]

        if len(match) != 1:
            raise ValueError(
                f"Could not uniquely identify overall AUROC "
                f"for model: {model}"
            )

        rows.append(
            {
                "scope": "Overall",
                "model": model,
                "observed_auroc": safe_float(
                    match.iloc[0]
                ),
            }
        )

    for scope in CATEGORY_SCOPES:
        for model in (
            SKYWORK_MODEL,
            FSFAIRX_MODEL,
        ):
            match = category.loc[
                (
                    category["sampling_stratum"]
                    == scope
                )
                & (
                    category["model"]
                    == model
                ),
                "category_auroc",
            ]

            if len(match) != 1:
                raise ValueError(
                    f"Could not uniquely identify category AUROC "
                    f"for scope={scope}, model={model}"
                )

            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "observed_auroc": safe_float(
                        match.iloc[0]
                    ),
                }
            )

    return pd.DataFrame(rows)


def validate_replicates(
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        replicates,
        [
            "bootstrap_id",
            "scope",
            SKYWORK_MODEL,
            FSFAIRX_MODEL,
            DIFFERENCE_COLUMN,
        ],
        "Bootstrap replicate table",
    )

    actual_scopes = set(
        replicates["scope"]
        .astype(str)
        .unique()
    )

    if actual_scopes != set(SCOPES):
        raise ValueError(
            "Unexpected bootstrap scopes.\n"
            f"Expected: {sorted(SCOPES)}\n"
            f"Actual:   {sorted(actual_scopes)}"
        )

    expected_rows = (
        N_BOOTSTRAP
        * len(SCOPES)
    )

    if len(replicates) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} bootstrap rows, "
            f"found {len(replicates)}."
        )

    result = replicates.copy()

    for column in (
        SKYWORK_MODEL,
        FSFAIRX_MODEL,
        DIFFERENCE_COLUMN,
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        finite = np.isfinite(
            result[column].to_numpy(dtype=float)
        )

        if not finite.all():
            bad_rows = np.flatnonzero(
                ~finite
            ).tolist()

            raise ValueError(
                f"{column} contains invalid values "
                f"at rows {bad_rows[:20]}."
            )

    recomputed_difference = (
        result[FSFAIRX_MODEL]
        - result[SKYWORK_MODEL]
    )

    if not np.allclose(
        result[DIFFERENCE_COLUMN].to_numpy(
            dtype=float
        ),
        recomputed_difference.to_numpy(
            dtype=float
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        max_error = float(
            np.max(
                np.abs(
                    result[
                        DIFFERENCE_COLUMN
                    ].to_numpy(dtype=float)
                    - recomputed_difference.to_numpy(
                        dtype=float
                    )
                )
            )
        )

        raise ValueError(
            "Stored paired AUROC difference does not equal "
            "FsfairX - Skywork. "
            f"Maximum error: {max_error}"
        )

    for scope in SCOPES:
        subset = result.loc[
            result["scope"] == scope
        ]

        if len(subset) != N_BOOTSTRAP:
            raise ValueError(
                f"{scope}: expected {N_BOOTSTRAP} "
                f"replicates, found {len(subset)}."
            )

        unique_ids = subset[
            "bootstrap_id"
        ].nunique()

        if unique_ids != N_BOOTSTRAP:
            raise ValueError(
                f"{scope}: expected {N_BOOTSTRAP} "
                f"unique bootstrap IDs, found "
                f"{unique_ids}."
            )

    return result


def build_difference_table(
    observed: pd.DataFrame,
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    alpha = 1.0 - CI_LEVEL
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q

    rows: list[dict[str, object]] = []

    for scope in SCOPES:
        observed_scope = observed.loc[
            observed["scope"] == scope
        ]

        skywork_match = observed_scope.loc[
            observed_scope["model"]
            == SKYWORK_MODEL,
            "observed_auroc",
        ]

        fsfairx_match = observed_scope.loc[
            observed_scope["model"]
            == FSFAIRX_MODEL,
            "observed_auroc",
        ]

        if (
            len(skywork_match) != 1
            or len(fsfairx_match) != 1
        ):
            raise ValueError(
                f"Could not uniquely obtain observed AUROCs "
                f"for scope={scope}."
            )

        skywork_auc = safe_float(
            skywork_match.iloc[0]
        )
        fsfairx_auc = safe_float(
            fsfairx_match.iloc[0]
        )
        observed_difference = (
            fsfairx_auc - skywork_auc
        )

        values = replicates.loc[
            replicates["scope"] == scope,
            DIFFERENCE_COLUMN,
        ].to_numpy(dtype=float)

        if len(values) != N_BOOTSTRAP:
            raise ValueError(
                f"{scope}: unexpected number "
                "of paired differences."
            )

        ci_lower = safe_float(
            np.quantile(
                values,
                lower_q,
            )
        )
        ci_upper = safe_float(
            np.quantile(
                values,
                upper_q,
            )
        )

        probability_positive = safe_float(
            np.mean(values > 0.0)
        )
        probability_negative = safe_float(
            np.mean(values < 0.0)
        )
        probability_equal = safe_float(
            np.mean(values == 0.0)
        )

        if ci_lower > 0.0:
            ci_conclusion = (
                "FsfairX higher"
            )
        elif ci_upper < 0.0:
            ci_conclusion = (
                "Skywork higher"
            )
        else:
            ci_conclusion = (
                "difference inconclusive"
            )

        rows.append(
            {
                "scope": scope,
                "skywork_observed_auroc": skywork_auc,
                "fsfairx_observed_auroc": fsfairx_auc,
                "observed_fsfairx_minus_skywork": (
                    observed_difference
                ),
                "bootstrap_mean_difference": safe_float(
                    np.mean(values)
                ),
                "bootstrap_se_difference": safe_float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),
                "ci_level": CI_LEVEL,
                "ci_method": (
                    "paired stratified percentile bootstrap"
                ),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_contains_zero": bool(
                    ci_lower <= 0.0 <= ci_upper
                ),
                "bootstrap_probability_fsfairx_higher": (
                    probability_positive
                ),
                "bootstrap_probability_skywork_higher": (
                    probability_negative
                ),
                "bootstrap_probability_equal": (
                    probability_equal
                ),
                "ci_based_conclusion": ci_conclusion,
                "n_bootstrap": N_BOOTSTRAP,
                "multiple_comparison_adjusted": False,
            }
        )

    return pd.DataFrame(rows)


def build_consistency_table(
    observed: pd.DataFrame,
    difference_table: pd.DataFrame,
) -> pd.DataFrame:
    category_observed = observed.loc[
        observed["scope"].isin(
            CATEGORY_SCOPES
        )
    ]

    pivot = category_observed.pivot(
        index="scope",
        columns="model",
        values="observed_auroc",
    ).reindex(CATEGORY_SCOPES)

    skywork_values = pivot[
        SKYWORK_MODEL
    ].to_numpy(dtype=float)

    fsfairx_values = pivot[
        FSFAIRX_MODEL
    ].to_numpy(dtype=float)

    pearson_result = pearsonr(
        skywork_values,
        fsfairx_values,
    )

    spearman_result = spearmanr(
        skywork_values,
        fsfairx_values,
    )

    category_differences = difference_table.loc[
        difference_table["scope"].isin(
            CATEGORY_SCOPES
        )
    ]

    n_fsfairx_higher_observed = int(
        (
            category_differences[
                "observed_fsfairx_minus_skywork"
            ]
            > 0.0
        ).sum()
    )

    n_skywork_higher_observed = int(
        (
            category_differences[
                "observed_fsfairx_minus_skywork"
            ]
            < 0.0
        ).sum()
    )

    n_equal_observed = int(
        (
            category_differences[
                "observed_fsfairx_minus_skywork"
            ]
            == 0.0
        ).sum()
    )

    n_difference_ci_excludes_zero = int(
        (
            ~category_differences[
                "ci_contains_zero"
            ]
        ).sum()
    )

    return pd.DataFrame(
        [
            {
                "n_categories": len(
                    CATEGORY_SCOPES
                ),
                "pearson_correlation_of_category_aurocs": (
                    safe_float(
                        pearson_result.statistic
                    )
                ),
                "pearson_p_value_exploratory": (
                    safe_float(
                        pearson_result.pvalue
                    )
                ),
                "spearman_correlation_of_category_aurocs": (
                    safe_float(
                        spearman_result.statistic
                    )
                ),
                "spearman_p_value_exploratory": (
                    safe_float(
                        spearman_result.pvalue
                    )
                ),
                "n_categories_fsfairx_higher_observed": (
                    n_fsfairx_higher_observed
                ),
                "n_categories_skywork_higher_observed": (
                    n_skywork_higher_observed
                ),
                "n_categories_equal_observed": (
                    n_equal_observed
                ),
                "n_category_difference_cis_excluding_zero": (
                    n_difference_ci_excludes_zero
                ),
                "category_scope_order": (
                    " | ".join(
                        CATEGORY_SCOPES
                    )
                ),
                "interpretation_note": (
                    "Cross-category correlations use only six "
                    "categories and are exploratory."
                ),
            }
        ]
    )


def main() -> None:
    print("Current working directory:")
    print(Path.cwd())
    print()

    paths = {
        "overall": OVERALL_RESULTS_PATH,
        "category": CATEGORY_RESULTS_PATH,
        "bootstrap_summary": (
            BOOTSTRAP_SUMMARY_PATH
        ),
        "bootstrap_replicates": (
            BOOTSTRAP_REPLICATES_PATH
        ),
    }

    for name, path in paths.items():
        require_file(path, name)

    actual_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
    }

    print("Input SHA-256 values:")
    for name in (
        "overall",
        "category",
        "bootstrap_summary",
        "bootstrap_replicates",
    ):
        print(
            f"{name:22s}: "
            f"{actual_hashes[name]}"
        )
    print()

    for name, expected_hash in (
        EXPECTED_HASHES.items()
    ):
        if (
            actual_hashes[name]
            != expected_hash
        ):
            raise ValueError(
                f"SHA-256 mismatch for {name}.\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hashes[name]}"
            )

    print("All input hashes verified.")
    print()

    overall = pd.read_csv(
        OVERALL_RESULTS_PATH
    )
    category = pd.read_csv(
        CATEGORY_RESULTS_PATH
    )
    bootstrap_summary = pd.read_csv(
        BOOTSTRAP_SUMMARY_PATH
    )
    replicates = pd.read_csv(
        BOOTSTRAP_REPLICATES_PATH
    )

    require_columns(
        bootstrap_summary,
        [
            "scope",
            "model",
            "observed_auroc",
            "ci_lower",
            "ci_upper",
        ],
        "Bootstrap summary",
    )

    observed = observed_auroc_by_scope(
        overall,
        category,
    )

    replicates = validate_replicates(
        replicates
    )

    print(
        "Observed results and paired bootstrap "
        "replicates verified."
    )
    print()

    difference_table = (
        build_difference_table(
            observed,
            replicates,
        )
    )

    consistency_table = (
        build_consistency_table(
            observed,
            difference_table,
        )
    )

    DIFFERENCE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    difference_table.to_csv(
        DIFFERENCE_OUTPUT_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    consistency_table.to_csv(
        CONSISTENCY_OUTPUT_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    difference_sha = sha256_file(
        DIFFERENCE_OUTPUT_PATH
    )
    consistency_sha = sha256_file(
        CONSISTENCY_OUTPUT_PATH
    )

    manifest = pd.DataFrame(
        [
            {
                "overall_results_path": str(
                    OVERALL_RESULTS_PATH
                ),
                "overall_results_sha256": (
                    actual_hashes["overall"]
                ),
                "category_results_path": str(
                    CATEGORY_RESULTS_PATH
                ),
                "category_results_sha256": (
                    actual_hashes["category"]
                ),
                "bootstrap_summary_path": str(
                    BOOTSTRAP_SUMMARY_PATH
                ),
                "bootstrap_summary_sha256": (
                    actual_hashes[
                        "bootstrap_summary"
                    ]
                ),
                "bootstrap_replicates_path": str(
                    BOOTSTRAP_REPLICATES_PATH
                ),
                "bootstrap_replicates_sha256": (
                    actual_hashes[
                        "bootstrap_replicates"
                    ]
                ),
                "difference_output_path": str(
                    DIFFERENCE_OUTPUT_PATH
                ),
                "difference_output_sha256": (
                    difference_sha
                ),
                "consistency_output_path": str(
                    CONSISTENCY_OUTPUT_PATH
                ),
                "consistency_output_sha256": (
                    consistency_sha
                ),
                "n_bootstrap": N_BOOTSTRAP,
                "ci_level": CI_LEVEL,
                "difference_definition": (
                    "FsfairX AUROC - Skywork AUROC"
                ),
                "category_results_exploratory": True,
                "multiple_comparison_adjusted": False,
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

    print("Paired AUROC differences:")
    print()

    print(
        difference_table[
            [
                "scope",
                "observed_fsfairx_minus_skywork",
                "ci_lower",
                "ci_upper",
                "ci_contains_zero",
                "ci_based_conclusion",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    print()

    print("Cross-category consistency:")
    print()

    print(
        consistency_table.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    print()

    print("Saved paired difference results:")
    print(DIFFERENCE_OUTPUT_PATH)
    print()
    print("Difference results SHA-256:")
    print(difference_sha)
    print()

    print("Saved category consistency results:")
    print(CONSISTENCY_OUTPUT_PATH)
    print()
    print("Consistency results SHA-256:")
    print(consistency_sha)
    print()

    print("Saved manifest:")
    print(MANIFEST_OUTPUT_PATH)
    print()

    print("=" * 72)
    print(
        "Paired model-difference and category-consistency "
        "analysis completed successfully."
    )
    print(
        "Category-level comparisons remain exploratory "
        "and unadjusted for multiplicity."
    )
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print(
            "TWO-RM COMPARISON ANALYSIS FAILED"
        )
        print("=" * 72)
        print(
            f"{type(error).__name__}: {error}"
        )
        sys.exit(1)
