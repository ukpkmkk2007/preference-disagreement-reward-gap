# -*- coding: utf-8 -*-
"""
Step 24: Score the frozen 400-pair sample with Skywork Reward 8B.

This script performs inference only. It does not train any model and does not
read diverging/high-agreement labels, human vote counts, task categories, or
manual annotation labels during scoring.

Inputs
------
1. data_processed/formal_sample_v1_stratified_annotation_frozen.csv
2. results/formal_sample_v1_annotation_freeze_manifest.csv

Outputs
-------
1. results/formal_sample_v1_skywork8b_scores_partial.csv
2. results/formal_sample_v1_skywork8b_scores.csv
3. results/formal_sample_v1_skywork8b_scoring_manifest.csv
"""

import gc
import hashlib
import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# =========================
# 0. Fixed configuration
# =========================

PROJECT_DIR = Path(
    r"C:\Users\23624\Desktop\preference_disagreement_baseline"
)

FROZEN_INPUT_PATH = (
    PROJECT_DIR
    / "data_processed"
    / "formal_sample_v1_stratified_annotation_frozen.csv"
)

FREEZE_MANIFEST_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_annotation_freeze_manifest.csv"
)

PARTIAL_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_skywork8b_scores_partial.csv"
)

FINAL_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_skywork8b_scores.csv"
)

SCORING_MANIFEST_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_skywork8b_scoring_manifest.csv"
)

MODEL_NAME = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Skywork--Skywork-Reward-Llama-3.1-8B-v0.2"
    / "snapshots"
    / "d4117fbfd81b72f41b96341238baa1e3e90a4ce1"
)
MAX_LENGTH = 2048
EXPECTED_ROWS = 400
SAVE_EVERY = 1
OVERWRITE_FINAL = False


# =========================
# 1. Utility functions
# =========================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_partial(results: list[dict]) -> None:
    pd.DataFrame(results).to_csv(
        PARTIAL_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )


# =========================
# 2. Setup and file checks
# =========================

if not PROJECT_DIR.exists():
    raise FileNotFoundError(
        f"Project directory does not exist:\n{PROJECT_DIR}"
    )

os.chdir(PROJECT_DIR)
(PROJECT_DIR / "results").mkdir(exist_ok=True)

print("Current working directory:")
print(Path.cwd())

if not FROZEN_INPUT_PATH.exists():
    raise FileNotFoundError(
        "Cannot find the frozen annotation file:\n"
        f"{FROZEN_INPUT_PATH}"
    )

if not FREEZE_MANIFEST_PATH.exists():
    raise FileNotFoundError(
        "Cannot find the annotation freeze manifest:\n"
        f"{FREEZE_MANIFEST_PATH}"
    )

if FINAL_OUTPUT_PATH.exists() and not OVERWRITE_FINAL:
    raise FileExistsError(
        "The final Skywork score file already exists:\n"
        f"{FINAL_OUTPUT_PATH}\n\n"
        "Do not silently overwrite completed model scores."
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This script requires an NVIDIA GPU."
    )

print("\nCUDA device:")
print(torch.cuda.get_device_name(0))


# =========================
# 3. Verify the frozen annotation hash
# =========================

freeze_manifest = pd.read_csv(FREEZE_MANIFEST_PATH)

if len(freeze_manifest) != 1:
    raise ValueError(
        "The annotation freeze manifest must contain exactly one row."
    )

if "frozen_sha256" not in freeze_manifest.columns:
    raise ValueError(
        "The annotation freeze manifest has no frozen_sha256 column."
    )

expected_frozen_hash = str(
    freeze_manifest.loc[0, "frozen_sha256"]
).strip()

actual_frozen_hash = sha256_file(FROZEN_INPUT_PATH)

if actual_frozen_hash != expected_frozen_hash:
    raise ValueError(
        "Frozen annotation hash mismatch.\n"
        f"Expected: {expected_frozen_hash}\n"
        f"Actual:   {actual_frozen_hash}\n\n"
        "The frozen file may have been edited after freezing."
    )

print("\nFrozen annotation SHA-256 verified:")
print(actual_frozen_hash)


# =========================
# 4. Load and validate the scoring input
# =========================

sample_df = pd.read_csv(FROZEN_INPUT_PATH)

required_columns = [
    "annotation_id",
    "comparison_id",
    "prompt",
    "response_a",
    "response_b",
]

missing_columns = [
    col for col in required_columns if col not in sample_df.columns
]

if missing_columns:
    raise ValueError(
        "Frozen input is missing required columns:\n- "
        + "\n- ".join(missing_columns)
    )

if len(sample_df) != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} rows, found {len(sample_df)}."
    )

if sample_df["annotation_id"].duplicated().any():
    raise ValueError("Duplicate annotation_id values found.")

if sample_df["comparison_id"].duplicated().any():
    raise ValueError("Duplicate comparison_id values found.")

sample_df = sample_df[required_columns].copy()
sample_df["_original_order"] = range(len(sample_df))

print("\nLoaded frozen scoring input.")
print("Rows:", len(sample_df))


# =========================
# 5. Resume from partial results if available
# =========================

score_columns = [
    "annotation_id",
    "comparison_id",
    "skywork_score_a",
    "skywork_score_b",
    "score_gap_abs",
    "diverging_score",
]

results = []

if PARTIAL_OUTPUT_PATH.exists():
    partial_df = pd.read_csv(PARTIAL_OUTPUT_PATH)

    missing_partial_columns = [
        col for col in score_columns if col not in partial_df.columns
    ]

    if missing_partial_columns:
        raise ValueError(
            "Existing partial file is missing columns:\n- "
            + "\n- ".join(missing_partial_columns)
        )

    if partial_df["annotation_id"].duplicated().any():
        raise ValueError(
            "Existing partial file contains duplicate annotation IDs."
        )

    valid_ids = set(sample_df["annotation_id"].astype(str))
    partial_ids = set(partial_df["annotation_id"].astype(str))

    if not partial_ids.issubset(valid_ids):
        raise ValueError(
            "Existing partial file contains IDs outside the frozen sample."
        )

    expected_id_map = dict(
        zip(
            sample_df["annotation_id"].astype(str),
            sample_df["comparison_id"].astype(str),
        )
    )

    for _, row in partial_df.iterrows():
        annotation_id = str(row["annotation_id"])
        comparison_id = str(row["comparison_id"])

        if expected_id_map[annotation_id] != comparison_id:
            raise ValueError(
                "Existing partial file has an annotation/comparison "
                f"ID mismatch for {annotation_id}."
            )

    results = (
        partial_df[score_columns]
        .to_dict(orient="records")
    )

    print("\nResuming from partial results.")
    print("Already scored:", len(results))
else:
    print("\nNo partial result found. Starting from row 1.")

completed_ids = {
    str(row["annotation_id"]) for row in results
}

remaining_df = sample_df[
    ~sample_df["annotation_id"].astype(str).isin(completed_ids)
].copy()

print("Remaining:", len(remaining_df))


# =========================
# 6. Load Skywork Reward 8B in 4-bit
# =========================

print("\nLoading tokenizer...")
if not MODEL_NAME.exists():
    raise FileNotFoundError(
        "Cannot find the local Skywork model snapshot:\n"
        f"{MODEL_NAME}"
    )

print("Using local model snapshot:")
print(MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_NAME),
    local_files_only=True,
)

print("\nPreparing 4-bit quantization configuration...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("\nLoading Skywork Reward 8B in 4-bit...")
print(
    "The model will be loaded strictly from the local cache. "
    "No Hugging Face network request will be made."
)

reward_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        str(MODEL_NAME),
        num_labels=1,
        local_files_only=True,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="eager",
    )
)

reward_model.eval()

print("\nModel loaded.")


# =========================
# 7. Reward-scoring function
# =========================

def get_reward_score(
    prompt: str,
    response: str,
) -> float:
    """
    Return one scalar reward for one prompt-response pair.

    The same chat template and inference settings are used for
    response A and response B.
    """

    conversation = [
        {
            "role": "user",
            "content": str(prompt),
        },
        {
            "role": "assistant",
            "content": str(response),
        },
    ]

    input_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to("cuda:0")

    with torch.inference_mode():
        output = reward_model(input_ids)
        score = float(
            output.logits[0][0].item()
        )

    del input_ids
    del output

    return score


# =========================
# 8. Score every remaining pair
# =========================

start_time = time.time()
initial_count = len(results)

for _, row in remaining_df.iterrows():
    example_no = len(results) + 1

    print("\n==============================")
    print(
        "Scoring example",
        example_no,
        "/",
        len(sample_df),
    )
    print("annotation_id:", row["annotation_id"])
    print("comparison_id:", row["comparison_id"])

    try:
        score_a = get_reward_score(
            row["prompt"],
            row["response_a"],
        )

        score_b = get_reward_score(
            row["prompt"],
            row["response_b"],
        )

        score_gap_abs = abs(
            score_a - score_b
        )

        # Smaller gap means more consistent with human disagreement.
        diverging_score = -score_gap_abs

        result = {
            "annotation_id": row["annotation_id"],
            "comparison_id": row["comparison_id"],
            "skywork_score_a": score_a,
            "skywork_score_b": score_b,
            "score_gap_abs": score_gap_abs,
            "diverging_score": diverging_score,
        }

        results.append(result)

        print("score_a:", score_a)
        print("score_b:", score_b)
        print("absolute gap:", score_gap_abs)

        if len(results) % SAVE_EVERY == 0:
            save_partial(results)

        gc.collect()
        torch.cuda.empty_cache()

    except Exception:
        save_partial(results)
        gc.collect()
        torch.cuda.empty_cache()

        print("\nScoring stopped after an error.")
        print(
            "Completed results were preserved in:\n"
            f"{PARTIAL_OUTPUT_PATH}"
        )
        raise


# =========================
# 9. Validate and save final score table
# =========================

result_df = pd.DataFrame(results)

if len(result_df) != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} completed scores, "
        f"but found {len(result_df)}."
    )

if result_df["annotation_id"].duplicated().any():
    raise ValueError(
        "Duplicate annotation IDs found in completed scores."
    )

result_df = result_df.merge(
    sample_df[
        [
            "annotation_id",
            "_original_order",
        ]
    ],
    on="annotation_id",
    how="left",
    validate="one_to_one",
)

if result_df["_original_order"].isna().any():
    raise ValueError(
        "Some completed score rows do not match the frozen sample."
    )

result_df = (
    result_df
    .sort_values("_original_order")
    .drop(columns=["_original_order"])
    .reset_index(drop=True)
)

result_df = result_df[score_columns]

result_df.to_csv(
    FINAL_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

final_score_hash = sha256_file(
    FINAL_OUTPUT_PATH
)

elapsed_seconds = time.time() - start_time
newly_scored = len(results) - initial_count


# =========================
# 10. Save scoring manifest
# =========================

scoring_manifest = pd.DataFrame(
    [
        {
            "model_name": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
            "local_model_snapshot": str(MODEL_NAME),
            "inference_mode": "4-bit NF4, no training",
            "input_file": str(FROZEN_INPUT_PATH),
            "input_sha256": actual_frozen_hash,
            "output_file": str(FINAL_OUTPUT_PATH),
            "output_sha256": final_score_hash,
            "n_expected": EXPECTED_ROWS,
            "n_scored": len(result_df),
            "n_newly_scored_this_run": newly_scored,
            "max_length": MAX_LENGTH,
            "cuda_device": torch.cuda.get_device_name(0),
            "elapsed_seconds_this_run": elapsed_seconds,
            "seconds_per_new_pair": (
                elapsed_seconds / newly_scored
                if newly_scored > 0
                else float("nan")
            ),
            "human_outcome_labels_used_during_scoring": False,
            "manual_annotation_labels_used_during_scoring": False,
        }
    ]
)

scoring_manifest.to_csv(
    SCORING_MANIFEST_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 11. Final report
# =========================

print("\n==============================")
print("Skywork Reward 8B scoring completed.")
print("Expected rows:", EXPECTED_ROWS)
print("Scored rows:", len(result_df))
print("Newly scored this run:", newly_scored)
print(
    "Elapsed seconds this run:",
    round(elapsed_seconds, 2),
)
print(
    "Mean absolute reward gap:",
    float(result_df["score_gap_abs"].mean()),
)
print(
    "Median absolute reward gap:",
    float(result_df["score_gap_abs"].median()),
)
print("==============================")

print("\nFinal score SHA-256:")
print(final_score_hash)

print("\nSaved:")
print(PARTIAL_OUTPUT_PATH)
print(FINAL_OUTPUT_PATH)
print(SCORING_MANIFEST_PATH)

print("\nIMPORTANT:")
print(
    "Do not analyze AUROC or scenario differences yet. "
    "First confirm that all 400 model scores were produced."
)
