# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:39:28 2026

@author: 23624
"""
import os
import gc
import time
import pandas as pd
import torch

# =========================
# 0. 固定项目目录
# =========================

from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_DIR)

os.environ["HF_HOME"] = os.path.join(PROJECT_DIR, "hf_cache")
os.environ["HF_HUB_CACHE"] = os.path.join(PROJECT_DIR, "hf_cache", "hub")

print("Current working directory:")
print(os.getcwd())

os.makedirs("results", exist_ok=True)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)


# =========================
# 1. 检查 CUDA
# =========================

print("\nChecking CUDA...")
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. Stop here.")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory GB:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
)


# =========================
# 2. 读取输入数据
# =========================

input_path = "data_processed/skywork_baseline_inputs.csv"

if not os.path.exists(input_path):
    raise FileNotFoundError(
        "Cannot find data_processed/skywork_baseline_inputs.csv. "
        "Run scripts/06_prepare_skywork_baseline_inputs.py first."
    )

df = pd.read_csv(input_path)

print("\nLoaded baseline input data.")
print("Rows:", len(df))


# =========================
# 3. 构造随机 100 条样本
# =========================

# 只保留有明确人类多数偏好的样本
df = df[df["has_majority_preference"] == True].copy()

df["total_text_chars"] = (
    df["prompt"].fillna("").astype(str).str.len()
    + df["response_a"].fillna("").astype(str).str.len()
    + df["response_b"].fillna("").astype(str).str.len()
)

# 为了避免 8GB 显存被极端长文本炸掉，先过滤掉特别长的样本
# 这一步是小规模可行性版本，不是最终全量复现。
MAX_TOTAL_CHARS = 4000

candidate_df = df[df["total_text_chars"] <= MAX_TOTAL_CHARS].copy()

print("\nCandidate pool after length filter:")
print("Rows:", len(candidate_df))
print("Max total chars:", candidate_df["total_text_chars"].max())
print("Mean total chars:", candidate_df["total_text_chars"].mean())

N_SAMPLES = 100

if len(candidate_df) < N_SAMPLES:
    raise ValueError("Not enough candidate examples. Increase MAX_TOTAL_CHARS or reduce N_SAMPLES.")

sample_df = candidate_df.sample(
    n=N_SAMPLES,
    random_state=42
).copy()

print("\nSelected random sample.")
print("N:", len(sample_df))
print("Average total chars:", sample_df["total_text_chars"].mean())
print("Majority label counts:")
print(sample_df["majority_label"].value_counts())


# =========================
# 4. 加载 Skywork 8B reward model
# =========================

model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("\nPreparing 4-bit quantization config...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("\nLoading Skywork reward model in 4-bit...")
print("If the model was already downloaded, this should be faster.")

rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    quantization_config=bnb_config,
    device_map={"": 0},
    attn_implementation="eager",
)

rm.eval()

print("\nModel loaded.")


# =========================
# 5. 定义打分函数
# =========================

def get_reward_score(prompt, response):
    """
    给一个 prompt-response pair 打 reward score。
    这一步只做 inference，不训练模型。
    """

    conv = [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(response)},
    ]

    input_ids = tokenizer.apply_chat_template(
        conv,
        tokenize=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda:0")

    with torch.no_grad():
        output = rm(input_ids)
        score = output.logits[0][0].item()

    del input_ids
    gc.collect()
    torch.cuda.empty_cache()

    return score


# =========================
# 6. 逐条样本打分
# =========================

results = []
start_time = time.time()

partial_path = "results/step5_skywork_8b_random100_partial.csv"

for idx, row in sample_df.iterrows():
    example_no = len(results) + 1

    print("\n==============================")
    print("Scoring example", example_no, "/", len(sample_df))
    print("comparison_id:", row["comparison_id"])
    print("human majority:", row["majority_label"])
    print("total chars:", row["total_text_chars"])

    prompt = row["prompt"]
    response_a = row["response_a"]
    response_b = row["response_b"]

    try:
        score_a = get_reward_score(prompt, response_a)
        score_b = get_reward_score(prompt, response_b)

        if score_a > score_b:
            pred_label = "A"
        elif score_b > score_a:
            pred_label = "B"
        else:
            pred_label = "Tie"

        correct = pred_label == row["majority_label"]
        score_gap_abs = abs(score_a - score_b)

        result = {
            "comparison_id": row["comparison_id"],
            "category": row["category"],
            "majority_label": row["majority_label"],
            "skywork_score_a": score_a,
            "skywork_score_b": score_b,
            "skywork_pred_label": pred_label,
            "correct": correct,
            "score_gap_abs": score_gap_abs,

            "n_A": row["n_A"],
            "n_B": row["n_B"],
            "n_Tie": row["n_Tie"],

            "diverging_simple": row["diverging_simple"],
            "diverging_paper_like": row["diverging_paper_like"],
            "diverging_substantial": row["diverging_substantial"],
            "high_agreement_pref": row["high_agreement_pref"],

            "total_text_chars": row["total_text_chars"],
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
        }

        results.append(result)

        print("score_a:", score_a)
        print("score_b:", score_b)
        print("pred:", pred_label)
        print("correct:", correct)
        print("gap:", score_gap_abs)

    except RuntimeError as e:
        print("RuntimeError on this example:")
        print(e)

        if "out of memory" in str(e).lower():
            print("CUDA out of memory. Saving partial results and stopping early.")
            break
        else:
            raise e

    # 每条都保存一次，防止中途崩掉
    pd.DataFrame(results).to_csv(
        partial_path,
        index=False,
        encoding="utf-8-sig"
    )


# =========================
# 7. 保存最终结果
# =========================

result_df = pd.DataFrame(results)

out_path = "results/step5_skywork_8b_random100_results.csv"
result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

elapsed = time.time() - start_time

print("\n==============================")
print("Finished random100 scoring.")
print("Scored examples:", len(result_df))
print("Elapsed seconds:", round(elapsed, 2))

if len(result_df) > 0:
    accuracy = result_df["correct"].mean()

    print("Random100 preference accuracy:", accuracy)

    summary = {
        "n_target": N_SAMPLES,
        "n_scored": len(result_df),
        "max_total_chars_filter": MAX_TOTAL_CHARS,
        "accuracy": float(accuracy),
        "mean_score_gap_abs": float(result_df["score_gap_abs"].mean()),
        "median_score_gap_abs": float(result_df["score_gap_abs"].median()),
        "elapsed_seconds": float(elapsed),
        "seconds_per_example": float(elapsed / len(result_df)),
        "n_correct": int(result_df["correct"].sum()),
        "n_wrong": int((~result_df["correct"]).sum()),
    }

    summary_df = pd.DataFrame([summary])

    summary_path = "results/step5_skywork_8b_random100_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nSummary:")
    print(summary_df.T)

    print("\nSaved results:")
    print(out_path)

    print("\nSaved summary:")
    print(summary_path)
else:
    print("No examples were scored.")
