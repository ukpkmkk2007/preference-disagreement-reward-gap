import pandas as pd
import os

os.makedirs("case_studies", exist_ok=True)

df = pd.read_csv("data_processed/multipref_with_diverging_labels.csv")

print("Loaded data.")
print("Number of rows:", len(df))

valid_df = df[df["n_valid"] > 0].copy()

print("Valid rows:", len(valid_df))

paper_like_diverging = valid_df[valid_df["diverging_paper_like"] == True].copy()

substantial_diverging = valid_df[valid_df["diverging_substantial"] == True].copy()

high_agreement = valid_df[valid_df["high_agreement_pref"] == True].copy()

weak_diverging = valid_df[
    (valid_df["diverging_simple"] == True)
    & (valid_df["diverging_paper_like"] == False)
].copy()

print("paper-like diverging:", len(paper_like_diverging))
print("substantial diverging:", len(substantial_diverging))
print("high agreement:", len(high_agreement))
print("weak diverging:", len(weak_diverging))

cols = [
    "comparison_id",
    "category",
    "prompt",
    "model_a",
    "response_a",
    "model_b",
    "response_b",
    "raw_overall_labels",
    "parsed_directions",
    "parsed_strengths",
    "n_A",
    "n_B",
    "n_Tie",
    "n_clear",
    "n_slight",
    "diverging_simple",
    "diverging_paper_like",
    "diverging_substantial",
    "high_agreement_pref",
    "tie_only",
    "only_slight_or_tie"
]

paper_like_diverging[cols].sample(
    n=min(30, len(paper_like_diverging)),
    random_state=42
).to_csv(
    "case_studies/paper_like_diverging_30.csv",
    index=False,
    encoding="utf-8-sig"
)

substantial_diverging[cols].sample(
    n=min(30, len(substantial_diverging)),
    random_state=42
).to_csv(
    "case_studies/substantial_diverging_30.csv",
    index=False,
    encoding="utf-8-sig"
)

high_agreement[cols].sample(
    n=min(30, len(high_agreement)),
    random_state=42
).to_csv(
    "case_studies/high_agreement_30.csv",
    index=False,
    encoding="utf-8-sig"
)

weak_diverging[cols].sample(
    n=min(30, len(weak_diverging)),
    random_state=42
).to_csv(
    "case_studies/weak_diverging_30.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Saved case study files.")