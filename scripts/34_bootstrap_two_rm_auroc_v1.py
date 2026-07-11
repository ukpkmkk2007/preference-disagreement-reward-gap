from __future__ import annotations

import hashlib
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data_processed"
    / "formal_sample_v1_two_rm_labeled_analysis_table.csv"
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

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_bootstrap_auroc_summary.csv"
)

REPLICATES_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_bootstrap_auroc_replicates.csv"
)

MANIFEST_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "formal_sample_v1_two_rm_bootstrap_auroc_manifest.csv"
)

EXPECTED_INPUT_SHA256 = (
    "d9447c490d613ee65b2618ed9025ca5e82227b925c87df2a5020d5d9b8da7f56"
)
EXPECTED_OVERALL_RESULTS_SHA256 = (
    "0fdf5b39aa11f3247c32bfddfedce710bf02566a1af8fc95e2e3006557a7ebd5"
)
EXPECTED_CATEGORY_RESULTS_SHA256 = (
    "7eddb14e687a49521874d1d72063c2ea982b1507ca2b71a1422569424c215d26"
)

EXPECTED_ROWS = 400
N_BOOTSTRAP = 2000
RANDOM_SEED = 20260709
CI_LEVEL = 0.95

LABEL_COLUMN = "diverging_id_label"
STRATUM_COLUMN = "sampling_stratum"

STRATA = [
    "Open QA",
    "Generation",
    "Coding",
    "Chat",
    "Brainstorm",
    "Closed QA",
]

EXPECTED_PER_CLASS = {
    "Open QA": 40,
    "Generation": 35,
    "Coding": 35,
    "Chat": 35,
    "Brainstorm": 35,
    "Closed QA": 20,
}

MODEL_SCORE_COLUMNS = {
    "Skywork-Reward-Llama-3.1-8B-v0.2":
        "skywork_diverging_score",
    "FsfairX-LLaMA3-RM-v0.1":
        "fsfairx_diverging_score",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found:\n{path}")


def safe_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Encountered non-finite value: {value}")
    return value


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        LABEL_COLUMN,
        STRATUM_COLUMN,
        *MODEL_SCORE_COLUMNS.values(),
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            "Input table is missing required columns:\n- "
            + "\n- ".join(missing)
        )

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, found {len(frame)}."
        )

    frame = frame.copy()

    frame[LABEL_COLUMN] = pd.to_numeric(
        frame[LABEL_COLUMN],
        errors="coerce",
    )
    if frame[LABEL_COLUMN].isna().any():
        raise ValueError(f"{LABEL_COLUMN} contains invalid values.")
    if not frame[LABEL_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{LABEL_COLUMN} must contain only 0 and 1.")
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype("int64")

    frame[STRATUM_COLUMN] = (
        frame[STRATUM_COLUMN]
        .astype("string")
        .str.strip()
    )
    if (
        frame[STRATUM_COLUMN].isna().any()
        or frame[STRATUM_COLUMN].eq("").any()
    ):
        raise ValueError(f"{STRATUM_COLUMN} contains invalid values.")

    actual_strata = set(frame[STRATUM_COLUMN].unique())
    if actual_strata != set(STRATA):
        raise ValueError(
            "Unexpected sampling strata.\n"
            f"Expected: {sorted(STRATA)}\n"
            f"Actual:   {sorted(actual_strata)}"
        )

    for model_name, score_column in MODEL_SCORE_COLUMNS.items():
        values = pd.to_numeric(
            frame[score_column],
            errors="coerce",
        )
        finite = np.isfinite(values.to_numpy(dtype=float))
        if not finite.all():
            bad = np.flatnonzero(~finite).tolist()
            raise ValueError(
                f"{model_name}: invalid scores at rows {bad[:20]}."
            )
        frame[score_column] = values.astype(float)

    counts = (
        frame
        .groupby([STRATUM_COLUMN, LABEL_COLUMN])
        .size()
        .unstack(fill_value=0)
    )

    for stratum, expected in EXPECTED_PER_CLASS.items():
        for label in (0, 1):
            actual = int(counts.loc[stratum, label])
            if actual != expected:
                raise ValueError(
                    f"{stratum}, label={label}: "
                    f"expected {expected}, found {actual}."
                )

    return frame


def build_cells(
    frame: pd.DataFrame,
) -> dict[tuple[str, int], np.ndarray]:
    cells: dict[tuple[str, int], np.ndarray] = {}

    for stratum in STRATA:
        for label in (0, 1):
            mask = (
                (frame[STRATUM_COLUMN] == stratum)
                & (frame[LABEL_COLUMN] == label)
            )
            indices = np.flatnonzero(mask.to_numpy()).astype(np.int64)
            expected = EXPECTED_PER_CLASS[stratum]
            if len(indices) != expected:
                raise ValueError(
                    f"Unexpected cell size for "
                    f"{stratum}, label={label}: {len(indices)}."
                )
            cells[(stratum, label)] = indices

    return cells


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Bootstrap sample lost one outcome class.")
    return safe_float(roc_auc_score(labels, scores))


def compute_observed(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scope in ["Overall", *STRATA]:
        subset = (
            frame
            if scope == "Overall"
            else frame.loc[frame[STRATUM_COLUMN] == scope]
        )

        labels = subset[LABEL_COLUMN].to_numpy(dtype=int)

        for model_name, score_column in MODEL_SCORE_COLUMNS.items():
            rows.append(
                {
                    "scope": scope,
                    "model": model_name,
                    "observed_auroc": auc(
                        labels,
                        subset[score_column].to_numpy(dtype=float),
                    ),
                    "n_total": int(len(subset)),
                    "n_diverging": int((labels == 1).sum()),
                    "n_high_agreement": int((labels == 0).sum()),
                }
            )

    return pd.DataFrame(rows)


def verify_observed_results(
    observed: pd.DataFrame,
    overall_results: pd.DataFrame,
    category_results: pd.DataFrame,
) -> None:
    for _, row in observed.iterrows():
        scope = str(row["scope"])
        model = str(row["model"])
        recomputed = float(row["observed_auroc"])

        if scope == "Overall":
            match = overall_results.loc[
                overall_results["model"] == model,
                "overall_auroc",
            ]
        else:
            match = category_results.loc[
                (
                    category_results["sampling_stratum"] == scope
                )
                & (category_results["model"] == model),
                "category_auroc",
            ]

        if len(match) != 1:
            raise ValueError(
                f"Could not uniquely match scope={scope}, model={model}."
            )

        stored = float(match.iloc[0])
        if not math.isclose(
            recomputed,
            stored,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Observed AUROC mismatch for scope={scope}, "
                f"model={model}. Recomputed={recomputed}, stored={stored}"
            )


def run_bootstrap(
    frame: pd.DataFrame,
    cells: dict[tuple[str, int], np.ndarray],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    labels_all = frame[LABEL_COLUMN].to_numpy(dtype=int)
    score_arrays = {
        model_name: frame[score_column].to_numpy(dtype=float)
        for model_name, score_column in MODEL_SCORE_COLUMNS.items()
    }

    rows = []

    for bootstrap_id in range(1, N_BOOTSTRAP + 1):
        sampled = {
            cell: rng.choice(
                indices,
                size=len(indices),
                replace=True,
            )
            for cell, indices in cells.items()
        }

        overall_indices = np.concatenate(
            [
                sampled[(stratum, label)]
                for stratum in STRATA
                for label in (0, 1)
            ]
        )

        overall_labels = labels_all[overall_indices]
        overall_row = {
            "bootstrap_id": bootstrap_id,
            "scope": "Overall",
        }

        for model_name, scores_all in score_arrays.items():
            overall_row[model_name] = auc(
                overall_labels,
                scores_all[overall_indices],
            )

        rows.append(overall_row)

        for stratum in STRATA:
            stratum_indices = np.concatenate(
                [
                    sampled[(stratum, 0)],
                    sampled[(stratum, 1)],
                ]
            )
            stratum_labels = labels_all[stratum_indices]

            row = {
                "bootstrap_id": bootstrap_id,
                "scope": stratum,
            }

            for model_name, scores_all in score_arrays.items():
                row[model_name] = auc(
                    stratum_labels,
                    scores_all[stratum_indices],
                )

            rows.append(row)

        if (
            bootstrap_id == 1
            or bootstrap_id % 100 == 0
            or bootstrap_id == N_BOOTSTRAP
        ):
            print(
                f"Completed bootstrap replicate "
                f"{bootstrap_id}/{N_BOOTSTRAP}"
            )

    replicates = pd.DataFrame(rows)
    model_names = list(MODEL_SCORE_COLUMNS.keys())

    replicates["fsfairx_minus_skywork_auroc"] = (
        replicates[model_names[1]]
        - replicates[model_names[0]]
    )

    return replicates


def summarize(
    observed: pd.DataFrame,
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    lower_q = (1.0 - CI_LEVEL) / 2.0
    upper_q = 1.0 - lower_q

    rows = []

    for _, observed_row in observed.iterrows():
        scope = str(observed_row["scope"])
        model = str(observed_row["model"])

        values = replicates.loc[
            replicates["scope"] == scope,
            model,
        ].to_numpy(dtype=float)

        if len(values) != N_BOOTSTRAP:
            raise ValueError(
                f"Unexpected replicate count for {scope}/{model}: "
                f"{len(values)}."
            )

        ci_lower = safe_float(np.quantile(values, lower_q))
        ci_upper = safe_float(np.quantile(values, upper_q))

        rows.append(
            {
                "scope": scope,
                "model": model,
                "observed_auroc": safe_float(
                    observed_row["observed_auroc"]
                ),
                "bootstrap_mean_auroc": safe_float(np.mean(values)),
                "bootstrap_se": safe_float(np.std(values, ddof=1)),
                "ci_level": CI_LEVEL,
                "ci_method": (
                    "paired stratified percentile bootstrap"
                ),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_contains_0_5": bool(
                    ci_lower <= 0.5 <= ci_upper
                ),
                "n_bootstrap": N_BOOTSTRAP,
                "random_seed": RANDOM_SEED,
                "n_total": int(observed_row["n_total"]),
                "n_diverging": int(
                    observed_row["n_diverging"]
                ),
                "n_high_agreement": int(
                    observed_row["n_high_agreement"]
                ),
            }
        )

    summary = pd.DataFrame(rows)

    scope_order = {
        scope: i
        for i, scope in enumerate(["Overall", *STRATA])
    }
    model_order = {
        model: i
        for i, model in enumerate(MODEL_SCORE_COLUMNS.keys())
    }

    summary["_scope_order"] = summary["scope"].map(scope_order)
    summary["_model_order"] = summary["model"].map(model_order)

    return (
        summary
        .sort_values(["_scope_order", "_model_order"])
        .drop(columns=["_scope_order", "_model_order"])
        .reset_index(drop=True)
    )


def main() -> None:
    start = time.perf_counter()

    print("Current working directory:")
    print(Path.cwd())
    print()

    require_file(INPUT_PATH, "Final labeled analysis table")
    require_file(
        OVERALL_RESULTS_PATH,
        "Observed overall AUROC results",
    )
    require_file(
        CATEGORY_RESULTS_PATH,
        "Observed category AUROC results",
    )

    actual_hashes = {
        "input": sha256_file(INPUT_PATH),
        "overall": sha256_file(OVERALL_RESULTS_PATH),
        "category": sha256_file(CATEGORY_RESULTS_PATH),
    }

    print("Input SHA-256 values:")
    print(f"Labeled analysis table: {actual_hashes['input']}")
    print(f"Overall AUROC results:  {actual_hashes['overall']}")
    print(f"Category AUROC results: {actual_hashes['category']}")
    print()

    expected_hashes = {
        "input": EXPECTED_INPUT_SHA256,
        "overall": EXPECTED_OVERALL_RESULTS_SHA256,
        "category": EXPECTED_CATEGORY_RESULTS_SHA256,
    }

    for name, expected in expected_hashes.items():
        if actual_hashes[name] != expected:
            raise ValueError(
                f"SHA-256 mismatch for {name}.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual_hashes[name]}"
            )

    print("All input hashes verified.")
    print()

    frame = validate_input(pd.read_csv(INPUT_PATH))
    overall_results = pd.read_csv(OVERALL_RESULTS_PATH)
    category_results = pd.read_csv(CATEGORY_RESULTS_PATH)

    print("Input table and all 12 stratum-by-label cells verified.")
    print()

    observed = compute_observed(frame)
    verify_observed_results(
        observed,
        overall_results,
        category_results,
    )

    print("Observed AUROCs match Steps 32 and 33.")
    print()

    cells = build_cells(frame)

    print(
        f"Running {N_BOOTSTRAP} paired, stratified "
        "bootstrap replicates..."
    )
    print()

    replicates = run_bootstrap(frame, cells)
    summary = summarize(observed, replicates)

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    replicates.to_csv(
        REPLICATES_OUTPUT_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    summary_sha = sha256_file(SUMMARY_OUTPUT_PATH)
    replicates_sha = sha256_file(REPLICATES_OUTPUT_PATH)
    elapsed = time.perf_counter() - start

    manifest = pd.DataFrame(
        [
            {
                "input_path": str(INPUT_PATH),
                "input_sha256": actual_hashes["input"],
                "overall_results_path": str(
                    OVERALL_RESULTS_PATH
                ),
                "overall_results_sha256": actual_hashes["overall"],
                "category_results_path": str(
                    CATEGORY_RESULTS_PATH
                ),
                "category_results_sha256": actual_hashes["category"],
                "summary_output_path": str(
                    SUMMARY_OUTPUT_PATH
                ),
                "summary_output_sha256": summary_sha,
                "replicates_output_path": str(
                    REPLICATES_OUTPUT_PATH
                ),
                "replicates_output_sha256": replicates_sha,
                "n_bootstrap": N_BOOTSTRAP,
                "random_seed": RANDOM_SEED,
                "ci_level": CI_LEVEL,
                "ci_method": (
                    "paired stratified percentile bootstrap"
                ),
                "elapsed_seconds": elapsed,
                "created_at_local": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
        ]
    )

    manifest.to_csv(
        MANIFEST_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Bootstrap AUROC summary:")
    print()

    print(
        summary[
            [
                "scope",
                "model",
                "observed_auroc",
                "ci_lower",
                "ci_upper",
                "ci_contains_0_5",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()

    print(f"Elapsed time: {elapsed:.2f} seconds")
    print()

    print("Saved bootstrap summary:")
    print(SUMMARY_OUTPUT_PATH)
    print()
    print("Summary SHA-256:")
    print(summary_sha)
    print()

    print("Saved bootstrap replicates:")
    print(REPLICATES_OUTPUT_PATH)
    print()
    print("Replicates SHA-256:")
    print(replicates_sha)
    print()

    print("Saved manifest:")
    print(MANIFEST_OUTPUT_PATH)
    print()

    print("=" * 72)
    print("Paired stratified bootstrap completed successfully.")
    print("Model-difference interpretation has not yet been performed.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("BOOTSTRAP AUROC ANALYSIS FAILED")
        print("=" * 72)
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)
