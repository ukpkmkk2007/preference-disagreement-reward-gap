import pandas as pd
import ast
import os

os.makedirs("results", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)

df = pd.read_csv("data_processed/multipref_parsed_labels.csv")

print("Loaded parsed labels.")
print("Number of rows:", len(df))
print(df[["n_A", "n_B", "n_Tie", "n_clear", "n_slight", "n_valid"]].head())
def safe_parse_list(x):
    """
    把 CSV 里保存的字符串形式 list 还原成 Python list。
    例如：
    "['A', 'B', 'Tie']" -> ['A', 'B', 'Tie']
    """
    if isinstance(x, list):
        return x

    if pd.isna(x):
        return []

    try:
        return ast.literal_eval(x)
    except Exception:
        return []
def count_clear_by_direction(row):
    directions = safe_parse_list(row["parsed_directions"])
    strengths = safe_parse_list(row["parsed_strengths"])

    n_A_clear = 0
    n_B_clear = 0
    n_A_slight = 0
    n_B_slight = 0

    for d, s in zip(directions, strengths):
        if d == "A" and s == "clear":
            n_A_clear += 1
        elif d == "B" and s == "clear":
            n_B_clear += 1
        elif d == "A" and s == "slight":
            n_A_slight += 1
        elif d == "B" and s == "slight":
            n_B_slight += 1

    return pd.Series({
        "n_A_clear": n_A_clear,
        "n_B_clear": n_B_clear,
        "n_A_slight": n_A_slight,
        "n_B_slight": n_B_slight,
    })


clear_counts = df.apply(count_clear_by_direction, axis=1)
df = pd.concat([df, clear_counts], axis=1)
df["diverging_simple"] = (df["n_A"] > 0) & (df["n_B"] > 0)
df["tie_only"] = (df["n_A"] == 0) & (df["n_B"] == 0) & (df["n_Tie"] > 0)
df["only_slight_or_tie"] = (
    (df["n_clear"] == 0)
    & (df["n_slight"] > 0)
)
df["diverging_paper_like"] = (
    (df["n_A"] > 0)
    & (df["n_B"] > 0)
    & (~df["tie_only"])
    & (~df["only_slight_or_tie"])
)
df["diverging_substantial"] = (
    (df["n_A_clear"] > 0)
    & (df["n_B_clear"] > 0)
)
df["high_agreement_A"] = (
    (df["n_A"] > 0)
    & (df["n_B"] == 0)
    & (df["n_Tie"] == 0)
)

df["high_agreement_B"] = (
    (df["n_B"] > 0)
    & (df["n_A"] == 0)
    & (df["n_Tie"] == 0)
)

df["high_agreement_pref"] = df["high_agreement_A"] | df["high_agreement_B"]
out_path = "data_processed/multipref_with_diverging_labels.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("Saved:", out_path)
print("Rows:", len(df))
valid_df = df[df["n_valid"] > 0].copy()

summary = {
    "total_examples": len(df),
    "valid_examples": len(valid_df),

    "tie_only_examples": int(valid_df["tie_only"].sum()),
    "only_slight_or_tie_examples": int(valid_df["only_slight_or_tie"].sum()),

    "diverging_simple_examples": int(valid_df["diverging_simple"].sum()),
    "diverging_simple_rate": float(valid_df["diverging_simple"].mean()),

    "diverging_paper_like_examples": int(valid_df["diverging_paper_like"].sum()),
    "diverging_paper_like_rate": float(valid_df["diverging_paper_like"].mean()),

    "diverging_substantial_examples": int(valid_df["diverging_substantial"].sum()),
    "diverging_substantial_rate": float(valid_df["diverging_substantial"].mean()),

    "high_agreement_pref_examples": int(valid_df["high_agreement_pref"].sum()),
    "high_agreement_pref_rate": float(valid_df["high_agreement_pref"].mean()),
}

summary_df = pd.DataFrame([summary])

summary_path = "results/step5_diverging_definition_summary.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\nSummary:")
print(summary_df.T)
print("Saved summary:", summary_path)
by_category = valid_df.groupby("category").agg(
    n=("comparison_id", "count"),
    diverging_paper_like_rate=("diverging_paper_like", "mean"),
    diverging_simple_rate=("diverging_simple", "mean"),
    substantial_rate=("diverging_substantial", "mean"),
    high_agreement_rate=("high_agreement_pref", "mean"),
).reset_index().sort_values("diverging_paper_like_rate", ascending=False)

category_path = "results/step5_diverging_by_category.csv"
by_category.to_csv(category_path, index=False, encoding="utf-8-sig")

print("\nBy category:")
print(by_category.head(20))
print("Saved category summary:", category_path)