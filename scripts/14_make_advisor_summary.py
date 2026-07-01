# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:55:50 2026

@author: 23624
"""
import os
import pandas as pd

# =========================
# 0. 固定项目目录
# =========================

PROJECT_DIR = r"C:\Users\23624\Desktop\preference_disagreement_baseline"
os.chdir(PROJECT_DIR)

os.makedirs("results", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("Current working directory:")
print(os.getcwd())


# =========================
# 1. 稳健读取 CSV
# =========================

def read_csv_with_fallback(path):
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            print(f"Trying to read {path} with encoding={enc}")
            df = pd.read_csv(path, encoding=enc)
            print(f"Successfully read {path} with encoding={enc}")
            return df
        except UnicodeDecodeError as e:
            print(f"Failed with encoding={enc}")
            last_error = e

    raise last_error


# =========================
# 2. 读取已有结果
# =========================

metrics_path = "results/step7_baseline_reproduction_metrics.csv"
case_path = "results/step8_combined_case_study_annotations.csv"

if not os.path.exists(metrics_path):
    raise FileNotFoundError(metrics_path)

if not os.path.exists(case_path):
    raise FileNotFoundError(case_path)

metrics_df = read_csv_with_fallback(metrics_path)
case_df = read_csv_with_fallback(case_path)

print("\nLoaded:")
print(metrics_path, len(metrics_df))
print(case_path, len(case_df))


# =========================
# 3. 提取核心指标
# =========================

m = metrics_df.iloc[0]

model = m["model"]
preference_accuracy = float(m["preference_accuracy"])
majority_baseline = float(m["majority_class_baseline_accuracy"])
diverging_auroc = float(m["diverging_id_auroc"])

mean_gap_div = float(m["mean_gap_diverging"])
mean_gap_high = float(m["mean_gap_high_agreement"])
median_gap_div = float(m["median_gap_diverging"])
median_gap_high = float(m["median_gap_high_agreement"])

n_pref = int(m["preference_accuracy_n"])
n_div_id = int(m["diverging_id_n"])


# =========================
# 4. 把细标签合并成 broad category
# =========================

def map_to_broad_label(label):
    label = str(label).lower()

    if "verbosity" in label or "concision" in label or "detail" in label or "over_elaboration" in label:
        return "verbosity_detail_concision"

    if "safety" in label or "refusal" in label or "redirection" in label or "deescalation" in label:
        return "safety_refusal_redirection"

    if "factuality" in label or "instruction" in label or "accuracy" in label:
        return "factuality_instruction_accuracy"

    if "technical" in label or "medical" in label or "complexity" in label:
        return "technical_medical_complexity"

    if "ambiguous" in label or "ambiguity" in label or "authenticity" in label or "puzzle" in label:
        return "task_ambiguity"

    if "recipe" in label or "polished" in label or "reliability" in label:
        return "practicality_vs_polish"

    if "sensitive" in label or "framing" in label:
        return "sensitive_topic_framing"

    return "other"


case_df["broad_reason_label"] = case_df["manual_reason_label"].apply(map_to_broad_label)

broad_counts = (
    case_df.groupby(["case_type", "broad_reason_label"])
    .size()
    .reset_index(name="count")
    .sort_values(["case_type", "count"], ascending=[True, False])
)

overall_broad_counts = (
    case_df.groupby("broad_reason_label")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

broad_counts_path = "results/step9_case_study_broad_label_counts_by_type.csv"
overall_broad_counts_path = "results/step9_case_study_broad_label_counts_overall.csv"
case_broad_path = "results/step9_combined_case_study_with_broad_labels.csv"

broad_counts.to_csv(broad_counts_path, index=False, encoding="utf-8-sig")
overall_broad_counts.to_csv(overall_broad_counts_path, index=False, encoding="utf-8-sig")
case_df.to_csv(case_broad_path, index=False, encoding="utf-8-sig")

print("\nSaved broad label outputs:")
print(broad_counts_path)
print(overall_broad_counts_path)
print(case_broad_path)

print("\nOverall broad label counts:")
print(overall_broad_counts)


# =========================
# 5. 生成 advisor summary
# =========================

advisor_report_path = "reports/advisor_update_summary.md"

summary = f"""# Short Research Update: MultiPref Diverging Preference Baseline Reproduction

## 1. Project Goal

I am doing a small-scale reproduction of the baseline evaluation pipeline from the paper *Diverging Preferences: When do Annotators Disagree and do Models Know?*

The main question is:

> When human annotators disagree on a preference pair, does a reward model also show uncertainty, or does it still make a confident scalar judgment?

The focus is on a no-training baseline using an existing reward model.

## 2. What I Have Reproduced

I reproduced the single-value reward model evaluation pipeline using:

- Dataset: MultiPref
- Model: `{model}`
- Setting: 4-bit inference, no model training
- Local hardware: RTX 5070 Laptop GPU
- Constraint: local 8GB GPU memory, so experiments are run on sampled subsets rather than full-scale 27B evaluation

For each prompt-response pair, I compute:

- `score_a = RM(prompt, response_a)`
- `score_b = RM(prompt, response_b)`

Then I use:

- `score_a > score_b` to predict A
- `score_b > score_a` to predict B
- `abs(score_a - score_b)` as the reward gap

## 3. Preference Accuracy Result

I evaluated preference prediction on a random subset of {n_pref} MultiPref examples with clear human majority preference.

Result:

- Preference Accuracy: **{preference_accuracy:.4f}**
- Majority-class baseline: **{majority_baseline:.4f}**

Interpretation:

The Skywork 8B reward model predicts the human majority preference better than a simple majority-class baseline on this sampled subset.

## 4. Diverging ID Result

I evaluated whether reward gap can identify diverging preference pairs.

Setup:

- Positive class: paper-like diverging examples
- Negative class: high-agreement preference examples
- Score used for Diverging ID: `-abs(score_a - score_b)`

The intuition is:

- smaller reward gap = lower model certainty
- lower model certainty may indicate human preference disagreement

Result:

- Diverging ID AUROC: **{diverging_auroc:.4f}**

Reward gap comparison:

| Group | Mean gap | Median gap |
|---|---:|---:|
| Diverging examples | {mean_gap_div:.4f} | {median_gap_div:.4f} |
| High-agreement examples | {mean_gap_high:.4f} | {median_gap_high:.4f} |

Interpretation:

The reward gap is smaller on diverging examples than on high-agreement examples. This suggests that the reward model has some signal for identifying human preference disagreement, but the AUROC is only moderate.

## 5. Qualitative Case Study

I manually inspected two sets of cases:

1. High-confidence preference errors:
   - Skywork disagrees with the human majority.
   - The reward gap is large.

2. Diverging examples with large reward gaps:
   - Human annotators disagree.
   - Skywork still gives a very decisive scalar preference.

The broad causes are summarized below:

## 6. Broad Error / Disagreement Categories

Overall case-study label counts:

{overall_broad_counts.to_string(index=False)}

By case type:

{broad_counts.to_string(index=False)}

Main qualitative findings:

- Many confident errors involve conflicts between factuality, instruction-following, verbosity, and practical usefulness.
- Some safety-related cases show a conflict between hard refusal and safe redirection.
- Some diverging examples have genuine task ambiguity, where multiple responses are defensible under different interpretations.
- In several cases, the reward model strongly prefers more detailed or more polished responses, even when humans are split or prefer a simpler answer.

## 7. Main Finding So Far

The current small-scale reproduction suggests:

1. Skywork 8B can recover majority human preference above a simple baseline.
2. Its reward gap has a weak-to-moderate signal for detecting human preference disagreement.
3. However, in some diverging cases, the model still produces a very large reward gap, meaning that a single-value reward model can collapse multi-dimensional human disagreement into a confident scalar judgment.

## 8. Limitations

This is not an exact reproduction of the paper's full Table 3 result.

Main limitations:

1. I use Skywork 8B instead of the paper's larger Skywork 27B model.
2. I use 4-bit quantized inference due to local GPU memory limits.
3. I use sampled subsets rather than full MultiPref evaluation.
4. Long examples are filtered for local feasibility.
5. The case-study labels are manually assigned and currently small-scale.

## 9. Possible Next Steps

Possible next steps are:

1. Scale the sample size from 100 to 300 or 500 if compute allows.
2. Run the same evaluation on another open reward model for comparison.
3. Analyze whether certain categories, such as safety, verbosity, or task ambiguity, are especially likely to produce reward-model overconfidence.
4. Compare reward gap with annotation entropy or disagreement strength more directly.
5. Build a small diagnostic benchmark of cases where single-value reward models are confidently wrong or overconfident on diverging preferences.
"""

with open(advisor_report_path, "w", encoding="utf-8") as f:
    f.write(summary)

print("\nSaved advisor summary:")
print(advisor_report_path)


# =========================
# 6. 生成中文汇报版本
# =========================

chinese_report_path = "reports/advisor_update_summary_zh.md"

summary_zh = f"""# 阶段性科研汇报：MultiPref 分歧偏好 baseline 复现

## 1. 当前目标

我目前在做论文 *Diverging Preferences: When do Annotators Disagree and do Models Know?* 的一个小规模 baseline 复现。

核心问题是：

> 当人类 annotators 对一个 preference pair 存在分歧时，reward model 是否也表现出不确定性，还是仍然给出很确定的单一标量判断？

当前重点是不训练模型，只复现 single-value reward model 的 evaluation pipeline。

## 2. 已完成内容

我使用了：

- 数据集：MultiPref
- 模型：`{model}`
- 设置：4-bit inference，不训练模型
- 本地硬件：RTX 5070 Laptop GPU
- 约束：本地显存约 8GB，因此目前是 sampled subset 小规模复现，不是完整 27B 全量复现

对每个 preference pair，计算：

- `score_a = RM(prompt, response_a)`
- `score_b = RM(prompt, response_b)`

然后：

- 如果 `score_a > score_b`，预测 A
- 如果 `score_b > score_a`，预测 B
- 用 `abs(score_a - score_b)` 作为 reward gap

## 3. Preference Accuracy 结果

在 {n_pref} 条有明确 human majority preference 的随机样本上：

- Preference Accuracy：**{preference_accuracy:.4f}**
- Majority-class baseline：**{majority_baseline:.4f}**

解释：

Skywork 8B 在该 sampled subset 上预测人类多数偏好的能力高于简单的 majority-class baseline。

## 4. Diverging ID 结果

我进一步评估 reward gap 是否能识别人类偏好分歧。

设置：

- 正类：paper-like diverging examples
- 负类：high-agreement preference examples
- Diverging score：`-abs(score_a - score_b)`

直觉是：

- reward gap 越小，说明模型越不确定
- 模型越不确定的 pair，可能越接近人类分歧样本

结果：

- Diverging ID AUROC：**{diverging_auroc:.4f}**

Reward gap 对比：

| 样本类型 | Mean gap | Median gap |
|---|---:|---:|
| Diverging examples | {mean_gap_div:.4f} | {median_gap_div:.4f} |
| High-agreement examples | {mean_gap_high:.4f} | {median_gap_high:.4f} |

解释：

Diverging examples 的 reward gap 确实小于 high-agreement examples，说明 reward gap 对识别人类分歧有一定信号，但 AUROC 只有中等水平，信号不强。

## 5. 定性 case study

我人工检查了两类样本：

1. High-confidence preference errors：
   - Skywork 和 human majority 不一致；
   - 且 reward gap 很大。

2. Diverging examples with large reward gaps：
   - 人类 annotators 有分歧；
   - 但 Skywork 仍然给出很大的 A/B 分数差。

## 6. 粗粒度原因统计

总体 case-study 原因统计：

{overall_broad_counts.to_string(index=False)}

按 case type 统计：

{broad_counts.to_string(index=False)}

主要观察：

- 很多高置信错误来自 factuality、instruction-following、verbosity、practical usefulness 之间的冲突。
- 一些 safety case 中，模型会在 hard refusal 和 safe redirection 之间表现出偏好差异。
- 一些 diverging examples 本身存在 task ambiguity，多个回答在不同理解下都合理。
- 在若干样本中，reward model 明显偏好更详细、更正式或更 polished 的回答，即使人类 annotators 有分歧。

## 7. 当前主要结论

当前小规模复现显示：

1. Skywork 8B 能够在 sampled subset 上超过简单 majority-class baseline，恢复部分人类多数偏好。
2. Reward gap 对识别人类分歧有弱到中等程度的信号。
3. 但在一些 diverging cases 上，模型仍然会给出很大的 reward gap，说明 single-value reward model 可能把多维人类偏好分歧压缩成一个非常确定的标量偏好。

## 8. 局限性

这不是论文 Table 3 的严格全量复现。

主要限制：

1. 使用 Skywork 8B，而不是论文中的 Skywork 27B。
2. 使用 4-bit quantized inference。
3. 当前只在 sampled subset 上评估，不是完整 MultiPref 全量。
4. 出于本地显存限制过滤了较长样本。
5. case study 标注目前规模较小，仍是初步 qualitative analysis。

## 9. 后续方向

可能的下一步包括：

1. 如果算力允许，将样本从 100 扩大到 300 或 500。
2. 换另一个 open reward model 做横向对比。
3. 分析 safety、verbosity、task ambiguity 等类别是否更容易导致 reward model overconfidence。
4. 更直接地比较 reward gap 与 annotation entropy / disagreement strength。
5. 构建一个小型 diagnostic benchmark，专门收集 single-value reward model 高置信错误或对 diverging preference 过度确定的样本。
"""

with open(chinese_report_path, "w", encoding="utf-8") as f:
    f.write(summary_zh)

print("\nSaved Chinese advisor summary:")
print(chinese_report_path)

print("\nStep 9 finished successfully.")
