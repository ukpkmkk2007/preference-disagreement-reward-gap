# Data and Release Policy

## 1. Purpose

This policy keeps the repository reproducible without redistributing raw dataset content, private annotations, model weights, credentials, or unnecessary large files.

## 2. Files That May Be Published

The following artifacts are suitable for public release after inspection:

### Source code

```text
scripts/*.py
```

Publish only final scripts that contributed to the reported pipeline. Remove obsolete, failed, backup, and duplicate versions.

### Documentation

```text
README.md
docs/*.md
reports/*.md
```

Reports must not contain full raw prompts, full responses, private paths, tokens, or identifying information.

### Aggregate results

```text
results/*.csv
```

Suitable examples include:

- overall AUROC summaries;
- task-specific AUROC summaries;
- bootstrap confidence-interval summaries;
- paired model-difference summaries;
- preference-accuracy summaries;
- operational high-gap disagreement rates;
- manifests.

Large bootstrap replicate tables are optional because they can be regenerated from code.

### Figures

```text
figures/*.png
```

Figures should contain only aggregate statistics.

### Frozen IDs and text-free score tables

Suitable public fields may include:

```text
annotation_id
comparison_id
sampling_stratum
task_scenario
diverging_id_label
n_A
n_B
n_Tie
skywork_score_a
skywork_score_b
skywork_score_gap_abs
fsfairx_score_a
fsfairx_score_b
fsfairx_score_gap_abs
```

Before release, confirm that these fields comply with the original dataset license.

## 3. Files That Must Remain Local

Do not publish:

- raw MultiPref files;
- full prompt text;
- full response A text;
- full response B text;
- unsanitized free-form annotation notes;
- local absolute file paths;
- Hugging Face access tokens;
- API keys;
- model weights;
- model caches;
- Conda environments;
- private logs;
- temporary download fragments.

The following local files should remain ignored if they contain raw text:

```text
data_processed/formal_sample_v1_stratified_annotation_frozen.csv
data_processed/formal_sample_v1_two_rm_labeled_analysis_table.csv
```

## 4. Recommended Public Analysis Table

Create a separate compact file such as:

```text
data_processed/formal_sample_v1_two_rm_labeled_analysis_table_public.csv
```

Recommended columns:

```text
annotation_id
comparison_id
sampling_stratum
diverging_id_label
task_scenario
disagreement_source_1
disagreement_source_2
disagreement_source_3
n_A
n_B
n_Tie
skywork_score_a
skywork_score_b
skywork_score_gap_abs
fsfairx_score_a
fsfairx_score_b
fsfairx_score_gap_abs
```

Remove any free-text field that could reproduce the original prompt, responses, or private annotation notes.

## 5. Pre-Commit Inspection

Before each public commit:

1. run `git status`;
2. inspect every new data file;
3. search for `prompt`, `response_a`, `response_b`, `token`, `secret`, and local username fields;
4. confirm that no model files are staged;
5. confirm that no raw dataset file is staged;
6. confirm that no absolute Windows path is present;
7. confirm that public tables contain only approved columns;
8. confirm that figures do not expose raw examples;
9. verify output hashes when manifests are available.

Useful commands:

```bash
git status
git diff --cached
git ls-files
```

## 6. Commit Organization

A clean publication sequence is:

```text
1. Add formal-sample preparation and validation scripts
2. Add second reward-model scoring pipeline
3. Add statistical analysis scripts
4. Add aggregate result tables
5. Add final figures
6. Add documentation and public release policy
7. Add text-free public analysis table
```

Avoid mixing model caches, raw data, and source-code changes in the same commit.

## 7. Licensing and Attribution

The repository should include:

- attribution to MultiPref;
- attribution to the reward-model checkpoints;
- a project code license, if appropriate;
- a statement that third-party datasets and model checkpoints retain their original licenses.

This repository should not imply ownership of MultiPref or of the reward-model checkpoints.
