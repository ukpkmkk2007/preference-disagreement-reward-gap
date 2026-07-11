from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\23624\Desktop\preference_disagreement_baseline")
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FILES = {
    "bootstrap": RESULTS / "formal_sample_v1_two_rm_bootstrap_auroc_summary.csv",
    "category": RESULTS / "formal_sample_v1_two_rm_category_auroc.csv",
    "paired": RESULTS / "formal_sample_v1_two_rm_paired_auroc_differences.csv",
    "preference": RESULTS / "formal_sample_v1_two_rm_preference_accuracy.csv",
    "overconfident": RESULTS / "formal_sample_v1_two_rm_overconfident_disagreement_rates.csv",
}

EXPECTED_HASHES = {
    "bootstrap": "5c319909aeccca2872db2b7cd835d65c3f3f356685a3a180117aa1c5b35cde02",
    "category": "7eddb14e687a49521874d1d72063c2ea982b1507ca2b71a1422569424c215d26",
    "paired": "8a6dbc21527b66243111321feb92ae8b5f5f2bd34149a07c0cd1731284b1cfc7",
    "preference": "474061aa20467ef555ab9dbf43a1ee1205128561e8429777ab05c4d0d3541ce0",
    "overconfident": "c39bd8fa37b71f03b9f5d7cbbacdc6b1d43687c12547114852c32730b90f0ea4",
}

SUMMARY_PATH = RESULTS / "formal_sample_v1_primary_results_summary.csv"
MANIFEST_PATH = RESULTS / "formal_sample_v1_primary_results_summary_and_figures_manifest.csv"

FIG_OVERALL = FIGURES / "formal_sample_v1_overall_auroc_ci.png"
FIG_CATEGORY = FIGURES / "formal_sample_v1_category_auroc.png"
FIG_PREF = FIGURES / "formal_sample_v1_preference_accuracy.png"
FIG_OVERCONF = FIGURES / "formal_sample_v1_overconfident_disagreement_rate.png"

SKYWORK = "Skywork-Reward-Llama-3.1-8B-v0.2"
FSFAIRX = "FsfairX-LLaMA3-RM-v0.1"
MODELS = [SKYWORK, FSFAIRX]
SHORT = {SKYWORK: "Skywork", FSFAIRX: "FsfairX"}

CATEGORIES = [
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
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs() -> dict[str, str]:
    hashes = {}

    print("Input SHA-256 values:")

    for name, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file:\n{path}")

        actual = sha256_file(path)
        hashes[name] = actual
        print(f"{name:16s}: {actual}")

        expected = EXPECTED_HASHES[name]
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {name}.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )

    print()
    print("All input hashes verified.")
    print()

    return hashes


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def build_summary(
    bootstrap: pd.DataFrame,
    paired: pd.DataFrame,
    preference: pd.DataFrame,
    overconfident: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in bootstrap.iterrows():
        rows.append(
            {
                "analysis_family": "AUROC",
                "scope": row["scope"],
                "model": row["model"],
                "estimate": row["observed_auroc"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
                "denominator_n": row.get("n_total", pd.NA),
                "note": "Positive class=diverging; predictor=-absolute reward gap.",
            }
        )

    for _, row in paired.iterrows():
        rows.append(
            {
                "analysis_family": "Paired AUROC difference",
                "scope": row["scope"],
                "model": "FsfairX minus Skywork",
                "estimate": row["observed_fsfairx_minus_skywork"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
                "denominator_n": pd.NA,
                "note": "Positive values favor FsfairX.",
            }
        )

    pref_keep = preference.loc[
        (
            (preference["scope_type"] == "overall")
            & (preference["scope_value"] == "Overall")
        )
        | (preference["scope_type"] == "disagreement_class")
    ]

    for _, row in pref_keep.iterrows():
        rows.append(
            {
                "analysis_family": "Preference accuracy",
                "scope": row["scope_value"],
                "model": row["model"],
                "estimate": row["preference_accuracy_primary"],
                "ci_lower": pd.NA,
                "ci_upper": pd.NA,
                "denominator_n": row["n_human_strict_majority"],
                "note": "Rows without a strict human majority are excluded.",
            }
        )

    over_keep = overconfident.loc[
        (overconfident["threshold_role"] == "primary_q75")
        & (overconfident["scope_type"] == "overall")
        & (overconfident["scope_value"] == "All diverging")
    ]

    for _, row in over_keep.iterrows():
        rows.append(
            {
                "analysis_family": "Operational overconfident disagreement",
                "scope": "All diverging",
                "model": row["model"],
                "estimate": row["operational_overconfident_disagreement_rate"],
                "ci_lower": row["wilson_95_ci_lower_fixed_threshold"],
                "ci_upper": row["wilson_95_ci_upper_fixed_threshold"],
                "denominator_n": row["n_diverging_in_scope"],
                "note": "Model-specific Q75 high-agreement gap threshold.",
            }
        )

    return pd.DataFrame(rows)


def plot_overall_auroc(bootstrap: pd.DataFrame) -> None:
    data = (
        bootstrap.loc[bootstrap["scope"] == "Overall"]
        .set_index("model")
        .reindex(MODELS)
        .reset_index()
    )

    estimates = data["observed_auroc"].to_numpy(float)
    lower = estimates - data["ci_lower"].to_numpy(float)
    upper = data["ci_upper"].to_numpy(float) - estimates
    x = np.arange(len(MODELS))

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        x,
        estimates,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=6,
        markersize=8,
    )
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xticks(x, [SHORT[m] for m in MODELS])
    plt.ylim(0.45, 0.75)
    plt.ylabel("AUROC")
    plt.title("Overall AUROC for detecting human disagreement")

    for index, value in enumerate(estimates):
        plt.text(index, value + 0.015, f"{value:.3f}", ha="center")

    save_figure(FIG_OVERALL)


def plot_category_auroc(category: pd.DataFrame) -> None:
    pivot = (
        category.pivot(
            index="sampling_stratum",
            columns="model",
            values="category_auroc",
        )
        .reindex(CATEGORIES)
    )

    x = np.arange(len(CATEGORIES))
    width = 0.36

    plt.figure(figsize=(10, 5.5))

    for model_index, model in enumerate(MODELS):
        offset = (model_index - 0.5) * width
        plt.bar(
            x + offset,
            pivot[model].to_numpy(float),
            width=width,
            label=SHORT[model],
        )

    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xticks(x, CATEGORIES, rotation=25, ha="right")
    plt.ylim(0.35, 0.85)
    plt.ylabel("AUROC")
    plt.title("Category-specific AUROC")
    plt.legend()

    save_figure(FIG_CATEGORY)


def plot_preference_accuracy(preference: pd.DataFrame) -> None:
    selected = preference.loc[
        (
            (preference["scope_type"] == "overall")
            & (preference["scope_value"] == "Overall")
        )
        | (preference["scope_type"] == "disagreement_class")
    ]

    scopes = ["Overall", "Diverging", "High-agreement"]

    pivot = (
        selected.pivot(
            index="scope_value",
            columns="model",
            values="preference_accuracy_primary",
        )
        .reindex(scopes)
    )

    x = np.arange(len(scopes))
    width = 0.36

    plt.figure(figsize=(8, 5.2))

    for model_index, model in enumerate(MODELS):
        offset = (model_index - 0.5) * width
        values = pivot[model].to_numpy(float)
        bars = plt.bar(
            x + offset,
            values,
            width=width,
            label=SHORT[model],
        )

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )

    plt.xticks(x, scopes)
    plt.ylim(0.45, 0.95)
    plt.ylabel("Preference accuracy")
    plt.title("Reward-model preference accuracy")
    plt.legend()

    save_figure(FIG_PREF)


def plot_overconfident_rate(overconfident: pd.DataFrame) -> None:
    selected = overconfident.loc[
        (overconfident["threshold_role"] == "primary_q75")
        & (
            (overconfident["scope_type"] == "overall")
            | (overconfident["scope_type"] == "sampling_stratum")
        )
    ]

    scopes = ["All diverging", *CATEGORIES]

    pivot = (
        selected.pivot(
            index="scope_value",
            columns="model",
            values="operational_overconfident_disagreement_rate",
        )
        .reindex(scopes)
    )

    x = np.arange(len(scopes))
    width = 0.36

    plt.figure(figsize=(11, 5.5))

    for model_index, model in enumerate(MODELS):
        offset = (model_index - 0.5) * width
        plt.bar(
            x + offset,
            pivot[model].to_numpy(float),
            width=width,
            label=SHORT[model],
        )

    plt.xticks(x, scopes, rotation=25, ha="right")
    plt.ylim(0.0, 0.30)
    plt.ylabel("Rate")
    plt.title(
        "Operational overconfident-disagreement rate "
        "(model-specific Q75 threshold)"
    )
    plt.legend()

    save_figure(FIG_OVERCONF)


def main() -> None:
    print("Current working directory:")
    print(Path.cwd())
    print()

    input_hashes = verify_inputs()

    bootstrap = pd.read_csv(FILES["bootstrap"])
    category = pd.read_csv(FILES["category"])
    paired = pd.read_csv(FILES["paired"])
    preference = pd.read_csv(FILES["preference"])
    overconfident = pd.read_csv(FILES["overconfident"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    summary = build_summary(
        bootstrap,
        paired,
        preference,
        overconfident,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        float_format="%.17g",
        encoding="utf-8-sig",
    )

    print("Generating figures...")
    plot_overall_auroc(bootstrap)
    plot_category_auroc(category)
    plot_preference_accuracy(preference)
    plot_overconfident_rate(overconfident)

    output_hashes = {
        "summary": sha256_file(SUMMARY_PATH),
        "overall_figure": sha256_file(FIG_OVERALL),
        "category_figure": sha256_file(FIG_CATEGORY),
        "preference_figure": sha256_file(FIG_PREF),
        "overconfident_figure": sha256_file(FIG_OVERCONF),
    }

    manifest = pd.DataFrame(
        [
            {
                **{
                    f"input_{name}_sha256": value
                    for name, value in input_hashes.items()
                },
                "summary_path": str(SUMMARY_PATH),
                "summary_sha256": output_hashes["summary"],
                "overall_figure_path": str(FIG_OVERALL),
                "overall_figure_sha256": output_hashes["overall_figure"],
                "category_figure_path": str(FIG_CATEGORY),
                "category_figure_sha256": output_hashes["category_figure"],
                "preference_figure_path": str(FIG_PREF),
                "preference_figure_sha256": output_hashes["preference_figure"],
                "overconfident_figure_path": str(FIG_OVERCONF),
                "overconfident_figure_sha256": output_hashes["overconfident_figure"],
                "new_statistical_tests_performed": False,
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

    print()
    print("Saved primary summary:")
    print(SUMMARY_PATH)
    print()
    print("Saved figures:")
    print(FIG_OVERALL)
    print(FIG_CATEGORY)
    print(FIG_PREF)
    print(FIG_OVERCONF)
    print()
    print("Saved manifest:")
    print(MANIFEST_PATH)
    print()
    print("=" * 72)
    print("Primary results summary and figures completed successfully.")
    print("No new statistical tests were performed.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("RESULT SUMMARY AND FIGURE GENERATION FAILED")
        print("=" * 72)
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)
