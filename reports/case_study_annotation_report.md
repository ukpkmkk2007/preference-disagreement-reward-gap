# Case Study Annotation Report

## 1. Purpose

This report summarizes two qualitative case studies based on the Skywork 8B reward model baseline.

The goal is to understand when the reward model makes confident but potentially problematic judgments.

The two case-study sets are:

1. High-confidence preference errors:
   cases where Skywork disagrees with the human majority and has a large reward gap.

2. Diverging examples with large reward gaps:
   cases where human annotators disagree, but Skywork still gives a highly decisive scalar preference.

## 2. Case Study 1: High-confidence preference errors

File:

`results/step8_annotated_high_confidence_errors.csv`

These are cases where:

- `correct = False`
- `score_gap_abs` is large

Interpretation:

Skywork not only disagrees with the human majority, but does so confidently.

Common causes include:

- factuality vs instruction-following
- verbosity vs concision
- hard refusal vs safe redirection
- medical nuance vs simple answer
- polished detail vs practical usefulness

## 3. Case Study 2: Diverging examples with large reward gaps

File:

`results/step8_annotated_diverging_large_gap.csv`

These are cases where:

- `diverging_paper_like = True`
- `score_gap_abs` is large

Interpretation:

Human annotators disagree, but Skywork makes a decisive judgment.

Common causes include:

- verbosity and depth
- technical complexity
- safety redirection vs direct compliance
- sensitive-topic framing
- task ambiguity
- practical reliability vs polished presentation

## 4. Label Counts

                       case_type                     manual_reason_label  count
 diverging_with_large_reward_gap generic_troubleshooting_vs_specific_fix      1
 diverging_with_large_reward_gap   medical_completeness_and_risk_factors      1
 diverging_with_large_reward_gap            minor_accuracy_and_concision      1
 diverging_with_large_reward_gap   recipe_reliability_vs_polished_detail      1
 diverging_with_large_reward_gap      safety_redirection_vs_user_request      1
 diverging_with_large_reward_gap                 sensitive_topic_framing      1
 diverging_with_large_reward_gap         task_ambiguity_and_authenticity      1
 diverging_with_large_reward_gap    technical_correctness_and_complexity      1
 diverging_with_large_reward_gap             verbosity_and_actionability      1
 diverging_with_large_reward_gap            verbosity_and_literary_depth      1
high_confidence_preference_error                ambiguous_puzzle_pattern      1
high_confidence_preference_error   current_factuality_vs_polished_detail      1
high_confidence_preference_error    detail_vs_concision_in_creative_task      1
high_confidence_preference_error     email_concision_vs_over_elaboration      1
high_confidence_preference_error     factuality_vs_instruction_following      1
high_confidence_preference_error         medical_nuance_vs_simple_answer      1
high_confidence_preference_error         oververbosity_vs_concise_answer      1
high_confidence_preference_error             refusal_vs_safe_redirection      1
high_confidence_preference_error            safety_caveat_vs_direct_list      1
high_confidence_preference_error            safety_tone_and_deescalation      1

## 5. Main Interpretation

The manual annotations suggest that Skywork's confident mistakes and decisive judgments over diverging preferences are not random.

They often occur when multiple quality dimensions conflict:

- factual correctness vs instruction following
- safety vs helpfulness
- detail vs concision
- polished formatting vs practical reliability
- direct compliance vs safe redirection
- task ambiguity vs single-answer scoring

This supports the broader finding that a single-value reward model can collapse multi-dimensional human preference disagreement into one confident scalar preference.

## 6. Outputs

Generated files:

- `results/step8_annotated_high_confidence_errors.csv`
- `results/step8_annotated_diverging_large_gap.csv`
- `results/step8_combined_case_study_annotations.csv`
- `results/step8_case_study_label_counts.csv`
- `reports/case_study_annotation_report.md`
