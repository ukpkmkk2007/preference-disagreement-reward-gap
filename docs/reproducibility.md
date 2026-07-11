# Reproducibility Guide

## 1. Environment

The project was developed with Python 3.11 in a Conda environment.

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Core packages include:

```text
torch
transformers
pandas
numpy
scikit-learn
scipy
matplotlib
huggingface_hub
```

GPU inference is required for reward-model scoring. Statistical analysis and figure generation do not require a GPU.

## 2. Directory Layout

Expected repository layout:

```text
.
├── scripts/
├── data_processed/
├── results/
├── figures/
├── reports/
├── docs/
├── README.md
├── requirements.txt
└── .gitignore
```

Raw dataset files and model caches should remain outside the public repository or in ignored local directories.

## 3. Execution Order

Run the formal pipeline in numerical order.

### Phase A: sample inspection and stratification

```text
20 → 21 → 22
```

These scripts:

- inspect the original formal-sample distribution;
- audit category-level eligible pools;
- prepare the balanced 400-example stratified sample.

Expected final category totals:

```text
Open QA       80
Generation    70
Coding        70
Chat          70
Brainstorm    70
Closed QA     40
Total        400
```

Expected disagreement-stratum totals:

```text
Diverging        200
High agreement   200
```

### Phase B: first reward-model scoring and validation

```text
23 → 24 → 25
```

Scripts 23–24 are the repository's formal Skywork input-preparation and scoring steps. Keep their existing numeric filenames unchanged.

Script 25 validates and freezes the Skywork score table.

### Phase C: second reward-model scoring and validation

```text
26 → 27 → 28
```

These scripts:

- smoke-test FsfairX;
- score the frozen formal sample;
- validate and freeze the FsfairX score table.

### Phase D: frozen analysis table construction

```text
29 → 30 → 31
```

These scripts:

- merge the two reward-model score tables;
- validate and freeze the stratified sample IDs;
- attach frozen disagreement and task labels.

### Phase E: primary statistical analysis

```text
32 → 33 → 34 → 35
```

These scripts compute:

- overall AUROC;
- task-specific AUROC;
- 2,000 paired stratified bootstrap replicates;
- paired reward-model AUROC differences;
- cross-task model consistency.

### Phase F: supplementary analyses

```text
36 → 37 → 38 → 39
```

These scripts compute:

- exploratory manual subgroup analyses;
- preference-target audit;
- preference accuracy;
- operational high-gap disagreement rates.

Only the final corrected version of script 36 should be published.

### Phase G: final summary and figures

```text
40
```

Script 40 consolidates the primary result table and creates the final figures.

## 4. Frozen Inputs and Integrity Checks

The following SHA-256 values identify the frozen local artifacts used for the reported results.

### Frozen annotation table

```text
File:
data_processed/formal_sample_v1_stratified_annotation_frozen.csv

SHA-256:
7e8c614e9f9dec12d48e74a31142a2b25fc9d01ec7070570745ef08925a7e10c
```

This file may contain non-public text or annotations and should remain local unless a sanitized version is created.

### Frozen Skywork scores

```text
File:
data_processed/formal_sample_v1_skywork8b_scores_frozen.csv

SHA-256:
b65688cc2039a99b26ad0a6acb349f2fa5bfb5368d63b84260e3ff49e5f73fe7
```

### Frozen FsfairX scores

```text
File:
data_processed/formal_sample_v1_fsfairx_rm_scores_frozen.csv

SHA-256:
87896f1aabf778f78c8aa1a082fa6ec86014d749e2a014fe90eadc279e6cab68
```

### Frozen sample IDs

```text
File:
data_processed/formal_sample_v1_stratified_ids_frozen.csv

SHA-256:
981edcd63732d41e7bfd3e5ef7a18f2076ab95dd0e85b5ff4bd73debef344bbb
```

### Master source file

```text
SHA-256:
02bd5783ddd028b6ba849f02d1af28a5b51748320c387906fcdbbb439a7d4a60
```

### Final labeled analysis table

```text
File:
data_processed/formal_sample_v1_two_rm_labeled_analysis_table.csv

SHA-256:
d9447c490d613ee65b2618ed9025ca5e82227b925c87df2a5020d5d9b8da7f56
```

The full final table should remain local if it contains raw prompt or response text. Publish a compact text-free table instead.

## 5. Expected Primary Outputs

### Overall AUROC

```text
results/formal_sample_v1_two_rm_overall_auroc.csv
```

Expected values:

```text
Skywork:  0.6022125
FsfairX:  0.6303750
```

### Task-specific AUROC

```text
results/formal_sample_v1_two_rm_category_auroc.csv
```

Expected values:

| Task scenario | Skywork | FsfairX |
|---|---:|---:|
| Open QA | 0.736875 | 0.758750 |
| Generation | 0.560000 | 0.564082 |
| Coding | 0.537143 | 0.668571 |
| Chat | 0.537143 | 0.555918 |
| Brainstorm | 0.648163 | 0.704898 |
| Closed QA | 0.580000 | 0.422500 |

### Bootstrap output

Expected files include:

```text
results/formal_sample_v1_two_rm_bootstrap_auroc_summary.csv
results/formal_sample_v1_two_rm_bootstrap_auroc_replicates.csv
```

The replicate table is optional for GitHub because it can be reconstructed by rerunning the bootstrap script.

### Preference accuracy

```text
results/formal_sample_v1_two_rm_preference_accuracy.csv
```

Expected overall values:

```text
Skywork: 0.703488
FsfairX: 0.781977
```

### Operational high-gap disagreement

Expected outputs include:

```text
results/formal_sample_v1_two_rm_overconfident_disagreement_thresholds.csv
results/formal_sample_v1_two_rm_overconfident_disagreement_rates.csv
results/formal_sample_v1_two_rm_overconfident_disagreement_flags.csv
```

### Final summary and figures

Expected outputs:

```text
results/formal_sample_v1_primary_results_summary.csv
results/formal_sample_v1_primary_results_summary_and_figures_manifest.csv

figures/formal_sample_v1_overall_auroc_ci.png
figures/formal_sample_v1_category_auroc.png
figures/formal_sample_v1_preference_accuracy.png
figures/formal_sample_v1_overconfident_disagreement_rate.png
```

## 6. Determinism and Validation

Before accepting a run:

1. confirm that each frozen input hash matches;
2. confirm that the sample contains 400 unique rows;
3. confirm a 200/200 diverging versus high-agreement split;
4. confirm the six task-category totals;
5. confirm that score tables have identical sample IDs;
6. confirm that all expected output columns are present;
7. rerun deterministic scripts and compare output hashes;
8. do not overwrite frozen inputs.

The operational high-gap disagreement script was rerun with identical output hashes, which supports deterministic behavior for that stage.

## 7. GPU Requirements

GPU usage is required only for reward-model inference.

Scripts 32–40 operate on frozen score tables and can be run on CPU.

Before requesting shared GPU resources, specify:

- model checkpoint;
- number of examples;
- maximum sequence length;
- precision or quantization setting;
- batch size;
- expected output file;
- estimated runtime.

## 8. Public Reproduction Path

A public user should be able to reproduce the statistical results from:

1. frozen sample IDs;
2. text-free frozen score tables;
3. a text-free public label table;
4. scripts 29–40;
5. the documented package environment.

A full raw-data reproduction may require separate access to MultiPref and local execution of the scoring scripts.
