# Annotation Guidelines v1

## 1. Purpose

This document defines the manual annotation rules for the stratified v1 experiment.

The confirmatory task-scenario analysis uses the original MultiPref task categories already stored in the internal master table:

- Open QA
- Generation
- Coding
- Chat
- Brainstorm
- Closed QA

These original categories are fixed by the dataset and are not manually changed.

The blinded annotation table is used for two secondary annotations:

1. a finer semantic task scenario (`task_scenario`);
2. one or more plausible disagreement sources (`disagreement_source_1` to `disagreement_source_3`).

The manual annotations must be assigned without access to:

- human vote counts;
- diverging/high-agreement labels;
- reward scores;
- reward gaps;
- reward-model predictions.

---

## 2. Files

Use only:

```text
data_annotation/formal_sample_v1_stratified_annotation.csv
```

Do not use the internal master table while annotating:

```text
data_processed/formal_sample_v1_stratified_master.csv
```

The master table contains outcome labels and would break the blinded design.

---

## 3. Primary and Secondary Analyses

### 3.1 Primary confirmatory scenario analysis

The primary scenario variable is the original MultiPref category:

- Open QA
- Generation
- Coding
- Chat
- Brainstorm
- Closed QA

These six strata were fixed before reward scoring, and each contains both diverging and high-agreement pairs.

### 3.2 Secondary exploratory analysis

The manually assigned `task_scenario` and `disagreement_source_*` fields are exploratory.

They are useful for interpreting why the reward-gap diagnostic succeeds or fails, but they should not be presented as fully confirmatory unless sample sizes are sufficient and the taxonomy remains unchanged.

---

## 4. General Annotation Procedure

For every row:

1. Read the prompt.
2. Read response A and response B completely.
3. Assign exactly one `task_scenario`.
4. Assign zero to three disagreement-source labels.
5. Add a short note only when the decision is unclear or unusual.
6. Set `annotation_status` to `completed`.

Do not infer which response the human majority preferred.

Do not judge the pair using reward-model behavior.

---

# Part A: Task Scenario

## 5. Task-Scenario Labels

Assign exactly one primary semantic task scenario.

### 5.1 `factual_qa`

Use when the user mainly asks for factual information, explanation, identification, or a knowledge-based answer.

Examples:

- historical facts;
- scientific explanations;
- definitions;
- general knowledge;
- factual comparisons.

Do not use when substantial mathematical derivation, coding, medical advice, or personal advice is central.

---

### 5.2 `technical_or_coding`

Use when the task primarily concerns:

- programming;
- debugging;
- software tools;
- technical implementation;
- computer systems;
- data processing;
- engineering procedures.

Use this even if the response also contains general explanation.

---

### 5.3 `medical_or_health`

Use when the user asks about:

- symptoms;
- treatment;
- medication;
- health risks;
- exercise injury;
- nutrition in a health context;
- mental-health support or medical interpretation.

This category concerns the content domain, not whether the answer is safe or correct.

---

### 5.4 `safety_or_sensitive`

Use when the central task involves:

- self-harm;
- violence;
- illegal or harmful activity;
- weapons;
- dangerous instructions;
- privacy invasion;
- severe misinformation;
- requests likely to trigger refusal or safety redirection.

Use this category only when safety handling is central. A normal medical question should remain `medical_or_health`.

---

### 5.5 `general_advice_or_chat`

Use when the user seeks:

- personal advice;
- relationship advice;
- social guidance;
- emotional support;
- opinions;
- ordinary conversation;
- practical life advice without a more specific technical domain.

---

### 5.6 `writing_or_creative`

Use when the user asks to create or revise:

- stories;
- poems;
- fictional scenes;
- emails;
- essays;
- social-media copy;
- advertisements;
- stylistic rewrites.

Use this category when the main output is authored text rather than factual explanation.

---

### 5.7 `reasoning_or_math`

Use when the main task requires:

- mathematical calculation;
- formal proof;
- logic;
- puzzle solving;
- multi-step quantitative reasoning.

A factual question containing numbers is not automatically `reasoning_or_math`.

---

### 5.8 `summarization_or_transformation`

Use when the user provides source text and asks for:

- summarization;
- extraction;
- classification;
- translation;
- reformatting;
- structured transformation;
- direct rewriting that preserves content.

Use `writing_or_creative` instead when the main goal is original composition or stylistic creation.

---

### 5.9 `brainstorming_or_planning`

Use when the main task is to generate multiple ideas, options, plans, strategies, names, or alternatives.

Examples:

- product ideas;
- trip plans;
- research-topic ideas;
- event concepts;
- lists of possible approaches.

---

### 5.10 `other`

Use only when none of the above categories reasonably fits.

Add a short explanation in `annotation_notes`.

---

## 6. Task-Scenario Tie-Breaking Rules

When several labels appear plausible, select the label that best describes the user's requested operation.

Priority examples:

- "Write Python code to calculate..." → `technical_or_coding`
- "Explain whether this symptom is dangerous" → `medical_or_health`
- "Write a poem about depression" → `writing_or_creative`
- "Advise me how to handle my partner" → `general_advice_or_chat`
- "Summarize this medical article" → `summarization_or_transformation`
- "Give ten app ideas" → `brainstorming_or_planning`

---

# Part B: Disagreement Source

## 7. General Principle

A disagreement source is a feature of the prompt-response contrast that could reasonably lead competent annotators to prefer different responses.

Do not force a disagreement-source label onto every pair.

A pair may receive:

- zero labels;
- one label;
- two labels;
- at most three labels.

Use the most central source first.

---

## 8. Disagreement-Source Labels

### 8.1 `verbosity_or_concision`

Use when the main trade-off is:

- detailed vs concise;
- comprehensive vs efficient;
- explanatory vs direct;
- more examples vs less redundancy.

Do not use merely because one response is longer. Length must create a meaningful quality trade-off.

---

### 8.2 `format_or_style`

Use when preferences may differ because of:

- bullets vs paragraphs;
- headings;
- markdown;
- tone;
- formality;
- rhetorical style;
- organization;
- readability;
- presentation conventions.

Use only when content quality is otherwise reasonably comparable.

---

### 8.3 `task_ambiguity`

Use when the prompt permits multiple reasonable interpretations, and the responses answer different interpretations.

Examples:

- unclear target audience;
- unspecified constraints;
- ambiguous reference;
- multiple plausible meanings;
- underspecified desired level of detail.

---

### 8.4 `expertise_or_complexity`

Use when responses trade off:

- technical rigor vs accessibility;
- advanced terminology vs plain language;
- nuanced caveats vs simplified explanation;
- specialist depth vs beginner usability.

This label is about level and complexity, not factual correctness.

---

### 8.5 `refusal_strategy`

Use when responses differ in:

- whether to refuse;
- how strongly to refuse;
- whether to provide a safe alternative;
- how much benign information to retain;
- whether the response is over-refusing or under-refusing.

This label is most common in safety-sensitive tasks.

---

### 8.6 `factuality`

Use when the central difference concerns:

- correctness;
- hallucination;
- unsupported claims;
- inaccurate details;
- citation reliability;
- factual precision.

Use only when the factual difference is identifiable from the text or common domain knowledge. Do not guess.

---

### 8.7 `instruction_following`

Use when responses differ in satisfying explicit user requirements, such as:

- requested format;
- requested length;
- required number of items;
- language;
- constraints;
- inclusion or exclusion rules.

This differs from `format_or_style`: instruction following concerns compliance with an explicit requirement.

---

### 8.8 `value_or_subjective_preference`

Use when the task has no single objectively best answer and preferences may depend on:

- moral values;
- political or cultural norms;
- aesthetic taste;
- interpersonal priorities;
- risk tolerance;
- subjective judgment.

Do not use merely because annotators could theoretically disagree. The response contrast must involve a genuine value trade-off.

---

### 8.9 `clear_quality_difference`

Use when one response appears plainly better on the central task and there is no meaningful trade-off that would reasonably explain disagreement.

Examples:

- one response is correct and the other is clearly wrong;
- one follows the request and the other ignores it;
- one is coherent and the other is unusable;
- one contains obvious fabrication with no compensating benefit.

When this label is used, normally leave the other disagreement-source fields blank unless a genuine secondary trade-off also exists.

---

### 8.10 `other`

Use when a plausible disagreement source exists but does not fit the listed categories.

Explain it briefly in `annotation_notes`.

---

## 9. Multi-Label Rules

Assign multiple labels only when each one independently contributes to the response contrast.

Good example:

```text
disagreement_source_1 = expertise_or_complexity
disagreement_source_2 = verbosity_or_concision
```

Use when one response is both more technical and substantially longer, and both differences matter.

Bad example:

```text
verbosity_or_concision
format_or_style
instruction_following
```

when the only real difference is that one response is longer.

Do not fill three labels merely because three slots exist.

---

## 10. Ordering Rules

Place the most important disagreement source in:

```text
disagreement_source_1
```

Use `_2` and `_3` only for genuine secondary sources.

Do not duplicate the same label within a row.

---

## 11. Annotation Notes

Use `annotation_notes` only when:

- the pair is hard to classify;
- two scenarios are nearly tied;
- a disagreement source is unusual;
- the source text appears corrupted;
- the responses are nearly identical;
- the row may require later review.

Recommended note style:

```text
Technical explanation, but the user mainly requests a rewrite.
```

or:

```text
Possible factuality issue; cannot verify from the text alone.
```

Keep notes short.

---

## 12. Annotation Status

Use:

```text
completed
```

when the row has been reviewed and all chosen fields are filled.

Use:

```text
review
```

when the row needs a second pass.

Do not leave completed rows as `unlabeled`.

---

# Part C: Quality Control

## 13. Pilot Before Full Annotation

Do not immediately annotate all 400 rows.

First annotate 30 rows using these guidelines.

After the pilot:

1. count task-scenario labels;
2. count disagreement-source labels;
3. inspect all rows marked `review`;
4. identify labels that are being confused;
5. revise the guidelines only if necessary;
6. freeze the taxonomy before annotating the remaining 370 rows.

The pilot is for testing annotation usability, not for testing reward-model results.

---

## 14. Consistency Check

After completing the full annotation:

1. randomly select 30 previously completed rows;
2. hide the original manual labels;
3. annotate them again;
4. compare the two rounds.

Report at least:

- task-scenario agreement rate;
- first-source agreement rate;
- any systematic confusion.

This is an intra-annotator consistency check, not inter-annotator reliability.

---

## 15. Prohibited Information During Annotation

Do not inspect:

- `formal_sample_v1_stratified_master.csv`;
- `formal_sample_v1_stratified_ids.csv`;
- diverging labels;
- high-agreement labels;
- vote counts;
- majority labels;
- reward-model outputs;
- reward gaps;
- per-scenario AUROC results.

These should be merged only after manual annotation is complete and frozen.

---

## 16. Final Interpretation Rule

The original MultiPref categories are the primary task-scenario variable because the sample was stratified on them before scoring.

The finer manual task scenarios and disagreement sources are secondary exploratory variables.

This distinction must be maintained in the final report.
