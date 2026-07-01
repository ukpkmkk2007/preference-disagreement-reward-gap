# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:11:34 2026

@author: 23624
"""
import os
import gc
import time
import pandas as pd
import torch

# =========================
# 0. 固定项目目录和缓存目录
# =========================

PROJECT_DIR = r"C:\Users\23624\Desktop\preference_disagreement_baseline"
os.chdir(PROJECT_DIR)

# 使用项目目录下的 Hugging Face cache
# 你之前 09 已经重新下载过一次模型，所以这里继续用同一个 cache。
os.environ["HF_HOME"] = os.path.join(PROJECT_DIR, "hf_cache")
os.environ["HF_HUB_CACHE"] = os.path.join(PROJECT_DIR, "hf_cache", "hub")

print("Current working directory:")
print(os.getcwd())

print("\nHF cache:")
print(os.environ["HF_HOME"])

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
# 3. bool 列处理
# =========================

def to_bool_series(s):
    """
    把 True/False 或 'True'/'False' 统一转成 bool。
    """
    if s.dtype == bool:
        return s

    return s.astype(str).str.lower().map({
        "true": True,
        "false": False,
        "1": True,
        "0": False
    })


bool_cols = [
    "diverging_paper_like",
    "diverging_substantial",
    "high_agreement_pref",
    "diverging_simple",
]

for col in bool_cols:
    if col in df.columns:
        df[col] = to_bool_series(df[col])


# =========================
# 4. 构造 Diverging ID 数据集
# =========================

df["total_text_chars"] = (
    df["prompt"].fillna("").astype(str).str.len()
    + df["response_a"].fillna("").astype(str).str.len()
    + df["response_b"].fillna("").astype(str).str.len()
)

# 为了保护 8GB 显存，先过滤极端长文本。
MAX_TOTAL_CHARS = 4000

eligible_df = df[df["total_text_chars"] <= MAX_TOTAL_CHARS].copy()

diverging_df = eligible_df[eligible_df["diverging_paper_like"] == True].copy()
high_agree_df = eligible_df[eligible_df["high_agreement_pref"] == True].copy()

print("\nEligible examples after length filter:")
print("All eligible:", len(eligible_df))
print("Diverging eligible:", len(diverging_df))
print("High-agreement eligible:", len(high_agree_df))

# balanced sample: 50 diverging + 50 high-agreement
N_PER_CLASS = 50

n_div = min(N_PER_CLASS, len(diverging_df))
n_high = min(N_PER_CLASS, len(high_agree_df))

if n_div == 0 or n_high == 0:
    raise ValueError("Not enough examples for diverging or high-agreement class.")

sample_div = diverging_df.sample(n=n_div, random_state=42).copy()
sample_high = high_agree_df.sample(n=n_high, random_state=42).copy()

sample_div["diverging_id_label"] = 1
sample_high["diverging_id_label"] = 0

sample_df = pd.concat([sample_div, sample_high], axis=0)
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nSelected Diverging ID sample:")
print("Total:", len(sample_df))
print("Diverging:", int((sample_df["diverging_id_label"] == 1).sum()))
print("High-agreement:", int((sample_df["diverging_id_label"] == 0).sum()))
print("Average total chars:", sample_df["total_text_chars"].mean())


# =========================
# 5. 加载 Skywork 8B reward model
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
print("If the model is already cached, this should not re-download the full model.")

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
# 6. 定义 reward scoring 函数
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
# 7. 手写 AUROC 函数
# =========================

def compute_auc(y_true, score):
    """
    手写 AUROC，避免安装 sklearn。
    y_true: 1 = diverging, 0 = high-agreement
    score: 分数越高，越倾向 diverging
    """

    temp = pd.DataFrame({
        "y": y_true,
        "score": score
    }).dropna().copy()

    n_pos = int((temp["y"] == 1).sum())
    n_neg = int((temp["y"] == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # rank 越大，score 越高
    temp["rank"] = temp["score"].rank(method="average")

    sum_ranks_pos = temp.loc[temp["y"] == 1, "rank"].sum()

    auc = (
        sum_ranks_pos
        - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)

    return float(auc)


# =========================
# 8. 逐条打分
# =========================

results = []
start_time = time.time()

partial_path = "results/step6_skywork_8b_diverging_id_partial.csv"

for idx, row in sample_df.iterrows():
    example_no = len(results) + 1

    print("\n==============================")
    print("Scoring example", example_no, "/", len(sample_df))
    print("comparison_id:", row["comparison_id"])
    print("diverging_id_label:", row["diverging_id_label"])
    print("diverging_paper_like:", row["diverging_paper_like"])
    print("high_agreement_pref:", row["high_agreement_pref"])
    print("total chars:", row["total_text_chars"])

    prompt = row["prompt"]
    response_a = row["response_a"]
    response_b = row["response_b"]

    try:
        score_a = get_reward_score(prompt, response_a)
        score_b = get_reward_score(prompt, response_b)

        score_gap_abs = abs(score_a - score_b)

        # gap 越小，越像 diverging
        diverging_score = -score_gap_abs

        result = {
            "comparison_id": row["comparison_id"],
            "category": row["category"],

            "diverging_id_label": row["diverging_id_label"],
            "diverging_paper_like": row["diverging_paper_like"],
            "diverging_substantial": row["diverging_substantial"],
            "high_agreement_pref": row["high_agreement_pref"],

            "skywork_score_a": score_a,
            "skywork_score_b": score_b,
            "score_gap_abs": score_gap_abs,
            "diverging_score": diverging_score,

            "n_A": row["n_A"],
            "n_B": row["n_B"],
            "n_Tie": row["n_Tie"],

            "majority_label": row["majority_label"],
            "total_text_chars": row["total_text_chars"],

            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
        }

        results.append(result)

        print("score_a:", score_a)
        print("score_b:", score_b)
        print("gap:", score_gap_abs)
        print("diverging_score:", diverging_score)

    except RuntimeError as e:
        print("RuntimeError on this example:")
        print(e)

        if "out of memory" in str(e).lower():
            print("CUDA out of memory. Saving partial results and stopping early.")
            break
        else:
            raise e

    # 每条保存一次，防止中途崩掉
    pd.DataFrame(results).to_csv(
        partial_path,
        index=False,
        encoding="utf-8-sig"
    )


# =========================
# 9. 计算 Diverging ID AUROC
# =========================

result_df = pd.DataFrame(results)

out_path = "results/step6_skywork_8b_diverging_id_results.csv"
result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

elapsed = time.time() - start_time

print("\n==============================")
print("Finished Diverging ID scoring.")
print("Scored examples:", len(result_df))
print("Elapsed seconds:", round(elapsed, 2))

if len(result_df) > 0:
    auc = compute_auc(
        y_true=result_df["diverging_id_label"],
        score=result_df["diverging_score"]
    )

    gap_by_class = result_df.groupby("diverging_id_label")["score_gap_abs"].agg(
        ["count", "mean", "median", "std"]
    )

    print("\nScore gap by class:")
    print(gap_by_class)

    print("\nDiverging ID AUROC:")
    print(auc)

    summary = {
        "n_target_per_class": N_PER_CLASS,
        "n_scored": len(result_df),
        "n_diverging": int((result_df["diverging_id_label"] == 1).sum()),
        "n_high_agreement": int((result_df["diverging_id_label"] == 0).sum()),
        "max_total_chars_filter": MAX_TOTAL_CHARS,
        "diverging_id_auroc": float(auc),
        "mean_gap_diverging": float(
            result_df[result_df["diverging_id_label"] == 1]["score_gap_abs"].mean()
        ),
        "mean_gap_high_agreement": float(
            result_df[result_df["diverging_id_label"] == 0]["score_gap_abs"].mean()
        ),
        "median_gap_diverging": float(
            result_df[result_df["diverging_id_label"] == 1]["score_gap_abs"].median()
        ),
        "median_gap_high_agreement": float(
            result_df[result_df["diverging_id_label"] == 0]["score_gap_abs"].median()
        ),
        "elapsed_seconds": float(elapsed),
        "seconds_per_example": float(elapsed / len(result_df)),
    }

    summary_df = pd.DataFrame([summary])

    summary_path = "results/step6_skywork_8b_diverging_id_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nSummary:")
    print(summary_df.T)

    print("\nSaved results:")
    print(out_path)

    print("\nSaved summary:")
    print(summary_path)
else:
    print("No examples were scored.")
