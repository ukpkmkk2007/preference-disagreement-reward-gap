# -*- coding: utf-8 -*-
"""
Step 27: Score the frozen 400-pair sample with FsfairX-LLaMA3-RM-v0.1.

This script performs inference only. It does not train any model and does not
read diverging/high-agreement labels, human vote counts, task categories, or
manual annotation labels during scoring.

Inputs
------
1. data_processed/formal_sample_v1_stratified_annotation_frozen.csv
2. results/formal_sample_v1_annotation_freeze_manifest.csv

Outputs
-------
1. results/formal_sample_v1_fsfairx_rm_scores_partial.csv
2. results/formal_sample_v1_fsfairx_rm_scores.csv
3. results/formal_sample_v1_fsfairx_rm_scoring_manifest.csv

Important
---------
- The model is loaded strictly from the fixed local Hugging Face snapshot.
- The script supports resuming from the partial score file.
- The frozen annotation file is verified by SHA-256 before scoring.
- No formal AUROC or scenario analysis is performed here.
"""

from __future__ import annotations

import gc
import hashlib
import os
import time
from pathlib import Path

# Force local/offline loading before importing Transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =========================
# 0. Fixed configuration
# =========================

PROJECT_DIR = Path(
    os.environ.get(
        "PREFERENCE_DISAGREEMENT_PROJECT_DIR",
        Path(__file__).resolve().parents[1],
    )
).expanduser().resolve()

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
    / "formal_sample_v1_fsfairx_rm_scores_partial.csv"
)

FINAL_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_fsfairx_rm_scores.csv"
)

SCORING_MANIFEST_PATH = (
    PROJECT_DIR
    / "results"
    / "formal_sample_v1_fsfairx_rm_scoring_manifest.csv"
)

MODEL_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--sfairXC--FsfairX-LLaMA3-RM-v0.1"
    / "snapshots"
    / "94fad49f1b3227aa8b566f415a335adb68ec544c"
)

MODEL_REPO_ID = "sfairXC/FsfairX-LLaMA3-RM-v0.1"
MODEL_REVISION = "94fad49f1b3227aa8b566f415a335adb68ec544c"

DEVICE = "cuda:0"
MAX_LENGTH = 2048
EXPECTED_ROWS = 400
SAVE_EVERY = 1
OVERWRITE_FINAL = False

REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
)

REQUIRED_INPUT_COLUMNS = [
    "annotation_id",
    "comparison_id",
    "prompt",
    "response_a",
    "response_b",
]

SCORE_COLUMNS = [
    "annotation_id",
    "comparison_id",
    "fsfairx_score_a",
    "fsfairx_score_b",
    "score_gap_abs",
    "diverging_score",
]


# =========================
# 1. Utility functions
# =========================

def repository_path(path: Path) -> str:
    """Return a repository-relative path for manifests when possible."""
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return str(path).replace(str(Path.home()), "~")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_partial(results: list[dict]) -> None:
    """Persist completed rows so the run can resume after interruption."""
    if not results:
        return

    partial_df = pd.DataFrame(results)
    partial_df = partial_df[SCORE_COLUMNS]

    partial_df.to_csv(
        PARTIAL_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def validate_model_snapshot(model_dir: Path) -> None:
    """Check that all local model files needed for inference are present."""
    if not model_dir.exists():
        raise FileNotFoundError(
            "Cannot find the local FsfairX model snapshot:\n"
            f"{model_dir}"
        )

    missing = [
        name
        for name in REQUIRED_MODEL_FILES
        if not (model_dir / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "The local FsfairX model snapshot is incomplete:\n- "
            + "\n- ".join(missing)
        )


def format_chat(
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
) -> str:
    """
    Format one prompt-response pair using the model's chat template.

    This matches the formatting used in the successful smoke test.
    """
    messages = [
        {
            "role": "user",
            "content": str(prompt),
        },
        {
            "role": "assistant",
            "content": str(response),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Avoid a duplicated leading BOS token if the template inserts one.
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]

    return text


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

for path in [
    FROZEN_INPUT_PATH,
    FREEZE_MANIFEST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file does not exist:\n{path}"
        )

if FINAL_OUTPUT_PATH.exists() and not OVERWRITE_FINAL:
    raise FileExistsError(
        "The final FsfairX score file already exists:\n"
        f"{FINAL_OUTPUT_PATH}\n\n"
        "Do not silently overwrite completed model scores."
    )

validate_model_snapshot(MODEL_DIR)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This script requires an NVIDIA GPU."
    )

if not torch.cuda.is_bf16_supported():
    raise RuntimeError(
        "The active GPU/PyTorch setup does not support BF16."
    )

print("\nCUDA device:")
print(torch.cuda.get_device_name(0))
print("CUDA version:", torch.version.cuda)
print("BF16 supported:", torch.cuda.is_bf16_supported())


# =========================
# 3. Verify frozen annotation hash
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
# 4. Load and validate scoring input
# =========================

sample_df = pd.read_csv(FROZEN_INPUT_PATH)

missing_columns = [
    col
    for col in REQUIRED_INPUT_COLUMNS
    if col not in sample_df.columns
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

sample_df = sample_df[REQUIRED_INPUT_COLUMNS].copy()
sample_df["_original_order"] = range(len(sample_df))

print("\nLoaded frozen scoring input.")
print("Rows:", len(sample_df))


# =========================
# 5. Resume from partial results
# =========================

results: list[dict] = []

if PARTIAL_OUTPUT_PATH.exists():
    partial_df = pd.read_csv(PARTIAL_OUTPUT_PATH)

    missing_partial_columns = [
        col
        for col in SCORE_COLUMNS
        if col not in partial_df.columns
    ]

    if missing_partial_columns:
        raise ValueError(
            "Existing partial file is missing columns:\n- "
            + "\n- ".join(missing_partial_columns)
        )

    partial_df = partial_df[SCORE_COLUMNS].copy()

    if partial_df["annotation_id"].duplicated().any():
        raise ValueError(
            "Existing partial file contains duplicate annotation IDs."
        )

    if partial_df["comparison_id"].duplicated().any():
        raise ValueError(
            "Existing partial file contains duplicate comparison IDs."
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

    numeric_columns = [
        "fsfairx_score_a",
        "fsfairx_score_b",
        "score_gap_abs",
        "diverging_score",
    ]

    for col in numeric_columns:
        partial_df[col] = pd.to_numeric(
            partial_df[col],
            errors="coerce",
        )

        if partial_df[col].isna().any():
            raise ValueError(
                f"Existing partial file contains invalid values in {col}."
            )

    results = partial_df.to_dict(orient="records")

    print("\nResuming from partial results.")
    print("Already scored:", len(results))
else:
    print("\nNo partial result found. Starting from row 1.")

completed_ids = {
    str(row["annotation_id"])
    for row in results
}

remaining_df = sample_df[
    ~sample_df["annotation_id"].astype(str).isin(completed_ids)
].copy()

print("Remaining:", len(remaining_df))


# =========================
# 6. Load tokenizer and model
# =========================

print("\nLoading tokenizer from local snapshot:")
print(MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is None:
        raise RuntimeError(
            "Tokenizer has neither pad_token_id nor eos_token_id."
        )
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")
print("Tokenizer vocabulary size:", len(tokenizer))
print("Pad token ID:", tokenizer.pad_token_id)

print("\nLoading FsfairX Reward Model in BF16...")
print(
    "The model will be loaded strictly from the local cache. "
    "No Hugging Face network request will be made."
)

reward_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
)

reward_model.config.pad_token_id = tokenizer.pad_token_id
reward_model.eval()

first_parameter = next(reward_model.parameters())

print("\nModel loaded.")
print("Model class:", reward_model.__class__.__name__)
print("Model device:", first_parameter.device)
print("Model dtype:", first_parameter.dtype)
print("Number of labels:", reward_model.config.num_labels)

allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
reserved_gb = torch.cuda.memory_reserved(0) / 1024**3

print(f"CUDA allocated after load: {allocated_gb:.3f} GB")
print(f"CUDA reserved after load:  {reserved_gb:.3f} GB")


# =========================
# 7. Reward-scoring function
# =========================

def get_reward_score(
    prompt: str,
    response: str,
) -> float:
    """
    Return one raw scalar reward for one prompt-response pair.

    The same chat template, maximum length, precision, and inference settings
    are used for response A and response B.
    """
    text = format_chat(
        tokenizer=tokenizer,
        prompt=prompt,
        response=response,
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        output = reward_model(**encoded)
        logits = output.logits

    if logits.numel() != 1:
        raise RuntimeError(
            "Expected one scalar reward, but received logits shape "
            f"{tuple(logits.shape)}."
        )

    score = float(
        logits.reshape(-1)[0]
        .float()
        .cpu()
        .item()
    )

    del encoded
    del output
    del logits

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
            prompt=row["prompt"],
            response=row["response_a"],
        )

        score_b = get_reward_score(
            prompt=row["prompt"],
            response=row["response_b"],
        )

        score_gap_abs = abs(score_a - score_b)

        # A smaller reward gap is more consistent with human disagreement.
        diverging_score = -score_gap_abs

        result = {
            "annotation_id": row["annotation_id"],
            "comparison_id": row["comparison_id"],
            "fsfairx_score_a": score_a,
            "fsfairx_score_b": score_b,
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

missing_score_columns = [
    col
    for col in SCORE_COLUMNS
    if col not in result_df.columns
]

if missing_score_columns:
    raise ValueError(
        "Completed result table is missing columns:\n- "
        + "\n- ".join(missing_score_columns)
    )

if result_df["annotation_id"].duplicated().any():
    raise ValueError(
        "Duplicate annotation IDs found in completed scores."
    )

if result_df["comparison_id"].duplicated().any():
    raise ValueError(
        "Duplicate comparison IDs found in completed scores."
    )

result_df = result_df.merge(
    sample_df[
        [
            "annotation_id",
            "comparison_id",
            "_original_order",
        ]
    ],
    on=[
        "annotation_id",
        "comparison_id",
    ],
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

result_df = result_df[SCORE_COLUMNS].copy()

recomputed_gap = (
    result_df["fsfairx_score_a"]
    - result_df["fsfairx_score_b"]
).abs()

if not torch.allclose(
    torch.tensor(
        recomputed_gap.to_numpy(),
        dtype=torch.float64,
    ),
    torch.tensor(
        result_df["score_gap_abs"].to_numpy(),
        dtype=torch.float64,
    ),
    rtol=0.0,
    atol=1e-12,
):
    raise ValueError(
        "Stored score_gap_abs does not equal abs(score_a-score_b)."
    )

recomputed_diverging = -result_df["score_gap_abs"]

if not torch.allclose(
    torch.tensor(
        recomputed_diverging.to_numpy(),
        dtype=torch.float64,
    ),
    torch.tensor(
        result_df["diverging_score"].to_numpy(),
        dtype=torch.float64,
    ),
    rtol=0.0,
    atol=1e-12,
):
    raise ValueError(
        "Stored diverging_score does not equal -score_gap_abs."
    )

result_df.to_csv(
    FINAL_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

final_score_hash = sha256_file(FINAL_OUTPUT_PATH)

elapsed_seconds = time.time() - start_time
newly_scored = len(results) - initial_count


# =========================
# 10. Save scoring manifest
# =========================

scoring_manifest = pd.DataFrame(
    [
        {
            "model_name": MODEL_REPO_ID,
            "model_revision": MODEL_REVISION,
            "local_model_snapshot": repository_path(MODEL_DIR),
            "inference_mode": "BF16, no training",
            "input_file": repository_path(FROZEN_INPUT_PATH),
            "input_sha256": actual_frozen_hash,
            "output_file": repository_path(FINAL_OUTPUT_PATH),
            "output_sha256": final_score_hash,
            "n_expected": EXPECTED_ROWS,
            "n_scored": len(result_df),
            "n_newly_scored_this_run": newly_scored,
            "max_length": MAX_LENGTH,
            "chat_template_used": True,
            "device": DEVICE,
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_dtype": str(first_parameter.dtype),
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

peak_allocated_gb = (
    torch.cuda.max_memory_allocated(0)
    / 1024**3
)

print("\n==============================")
print("FsfairX Reward Model scoring completed.")
print("Expected rows:", EXPECTED_ROWS)
print("Scored rows:", len(result_df))
print("Newly scored this run:", newly_scored)
print(
    "Elapsed seconds this run:",
    round(elapsed_seconds, 2),
)
print(
    "Seconds per newly scored pair:",
    (
        round(elapsed_seconds / newly_scored, 4)
        if newly_scored > 0
        else "N/A"
    ),
)
print(
    "Mean absolute reward gap:",
    float(result_df["score_gap_abs"].mean()),
)
print(
    "Median absolute reward gap:",
    float(result_df["score_gap_abs"].median()),
)
print(
    "Peak CUDA allocated:",
    round(peak_allocated_gb, 3),
    "GB",
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
    "First validate and freeze all 400 FsfairX scores."
)


if __name__ == "__main__":
    pass
