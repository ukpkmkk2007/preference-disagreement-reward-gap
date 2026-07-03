# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:21:28 2026

@author: 23624
"""
import pandas as pd
import os

os.makedirs("results", exist_ok=True)

summary = pd.read_csv("results/step12_scenario_pilot_manual_summary.csv")
cases = pd.read_csv("results/step14_strong_failure_cases.csv")

strong_cases = cases[cases["failure_strength"] == "strong_failure"].copy()
strong_cases = strong_cases.sort_values("score_gap_abs", ascending=False)

text = []

text.append("Pilot: Scenario-conditioned failure analysis of reward-gap diagnostics")
text.append("")
text.append("Research question:")
text.append("When does reward gap fail as a diagnostic signal for human preference disagreement across different task scenarios?")
text.append("")

text.append("1. Scenario-level summary")
text.append(summary.to_string(index=False))
text.append("")

text.append("2. Main preliminary observation")
text.append(
    "The strongest failure cases are concentrated in verbosity/concision, "
    "with additional cases in technical/expert-advice and factuality/instruction-following. "
    "This suggests that reward gap may not be equally reliable across task scenarios."
)
text.append("")

text.append("3. Strong failure cases")
for _, row in strong_cases.iterrows():
    text.append(
        f"- scenario={row['scenario_manual']}, "
        f"gap={row['score_gap_abs']:.2f}, "
        f"model_pred={row['model_pred']}, "
        f"majority_label={row['majority_label']}, "
        f"votes=A:{row['n_A']}, B:{row['n_B']}, Tie:{row['n_Tie']}"
    )

text.append("")
text.append("4. Current limitation")
text.append(
    "This is only a 100-example pilot. Scenario labels are manually corrected but still preliminary. "
    "The next step would be to expand the sample size and, if useful, compare multiple reward models."
)

out_path = "results/step15_pilot_summary_for_advisor.txt"

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(text))

print("Saved:", out_path)
print("\n".join(text))
