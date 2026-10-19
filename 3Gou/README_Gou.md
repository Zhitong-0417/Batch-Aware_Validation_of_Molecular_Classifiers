# Gou

This directory contains the Python implementation and results for the **Gou cfRNA lung-cancer dataset** used in the batch-aware molecular classifier validation project.

## Data Source

The dataset is derived from:

**Wang, J. et al. (2024). Terminal modifications independent cell-free RNA sequencing enables sensitive early cancer detection and classification. Nature Communications, 15, 156.**

- Article: https://doi.org/10.1038/s41467-023-44461-y
- Original code: https://github.com/JinyongHuang/SLiPiR-seq
- GSA-Human accession: **HRA004959**

The study generated plasma cell-free RNA (cfRNA) sequencing profiles and developed machine-learning models for cancer detection and classification.

The current project uses the processed molecular data and selected lung-cancer features for downstream batch-aware validation rather than reproducing the raw sequencing pipeline.

---

## Data Files

The Gou workflow is based on the following source files:

### `41467_2023_44461_MOESM12_ESM.xlsx`

Supplementary data associated with the original publication.

It provides study-level information used to recover and verify the feature definitions and supporting information required by the current analysis.

### `HRA004959.xlsx`

Sample-level information associated with the clinical sequencing dataset.

It is used to support sample identification, cohort information, and alignment between molecular measurements and study metadata.

### `combination_selected_mi_sn_sno_lung.csv`

Main analysis-ready expression dataset used directly by the current Python workflow.

**Data type:** processed plasma cfRNA expression data restricted to the lung-cancer feature set used in the downstream analysis.

This is the primary molecular input for the batch-aware validation workflow.

> The large source datasets are not stored directly in this repository.  
> They should be placed in the `3Gou/` directory when reproducing the analysis locally.

---

## Original Preprocessing and Feature Selection

The original study processed raw cfRNA sequencing reads before constructing the expression matrix. Major upstream steps included:

- adapter and low-quality read removal;
- RNA-type-specific read mapping and counting;
- sample-level sequencing quality control;
- construction of cfRNA expression matrices;
- RPM normalization and log transformation.

Low-quality samples were excluded before downstream analysis.

For lung-cancer detection, the original study further filtered candidate cfRNAs using expression and differential-expression criteria and evaluated three feature-selection strategies:

- **Top N filtering**
- **Boruta**
- **LASSO**

The current Gou workflow does **not** repeat the complete raw sequencing or original feature-discovery pipeline. Instead, it starts from the processed and selected molecular features contained in `combination_selected_mi_sn_sno_lung.csv`.

---

## Main Files

```text
3Gou/
├── README_Gou.md
├── SSV_Gou_Python_pro.ipynb
├── SSV_Gou_Python_pro.py
├── gou.py
└── results_pro_gou/
```

### `SSV_Gou_Python_pro.py`

Main executable Gou batch-aware validation pipeline.

It performs:

- loading and validating the selected-feature dataset;
- sample and class alignment;
- batch / meta-batch construction;
- Natural and Class-stratified validation;
- normalization and classifier evaluation;
- bidirectional batch-aware testing;
- result aggregation and diagnostic generation.

### `gou.py`

Contains Gou-specific helper functions, including:

- selected-feature filtering;
- Gou-specific normalization helpers;
- feature-level diagnostics;
- Gou-specific visualization functions.

Generic SSV functionality shared across datasets is stored in the repository-level `src/` directory.

### `SSV_Gou_Python_pro.ipynb`

Notebook version of the Gou analysis, retained for interactive inspection and result visualization.

---

## Classification Task

The current Gou analysis focuses on:

| Positive class | Negative class | Interpretation |
|---|---|---|
| Lung cancer | Cancer-free controls | Lung-cancer detection |

Unlike Yokoi and JAMA, the current Gou implementation does not contain multiple medical scenarios. Instead, the analysis focuses on how validation performance for this fixed classification problem changes as train/test batch separation becomes stronger.

---

## Validation Design

Two complementary validation protocols are used.

### Natural

Preserves the observed class composition within the batch structure.

This represents the primary deployment-oriented analysis and includes both:

- batch/domain shift;
- class-composition differences.

### Class-stratified

Makes class composition across batches more comparable.

This serves as a sensitivity analysis, reducing the contribution of prevalence differences so that remaining degradation can be interpreted more directly as batch/domain-associated shift.

---

## Results Structure

All Gou Pro outputs are stored under:

```text
results_pro_gou/
├── class_stratified/
├── natural/
├── Gou_Pro_dual_track_result_interpretation.md
└── ssv_pro_manifest.csv
```

### `class_stratified/`

Contains classifier-performance results under the Class-stratified protocol.

### `natural/`

Contains classifier-performance results under the Natural protocol.

### `ssv_pro_manifest.csv`

Records the generated Pro analysis outputs and their locations.

---

## Main Result Interpretation

The primary scientific interpretation of the Gou analysis is:

[`results_pro_gou/Gou_Pro_dual_track_result_interpretation.md`](results_pro_gou/Gou_Pro_dual_track_result_interpretation.md)

This file should be treated as the main entry point for interpreting the Gou results.

It integrates:

- Natural and Class-stratified validation;
- raw error and balanced error;
- performance changes as batch separation becomes stronger;
- classifier and normalization sensitivity;
- bidirectional batch-transfer behavior;
- the distinction between baseline classification difficulty and additional batch-associated degradation.

The interpretation file should be read together with the corresponding figures and numerical outputs in `natural/` and `class_stratified/`.

---

## Analysis Perspective

The main question of the Gou analysis is not whether the selected cfRNA features can classify lung cancer under a conventional random split.

Instead, the project asks:

```text
How stable is classifier performance
when training and testing data become increasingly separated
by batch-associated structure?
```

Conceptually:

```text
observed validation difficulty
=
baseline molecular classification difficulty
+
batch-associated degradation
+
class-composition effects
```

The Natural and Class-stratified tracks are therefore interpreted jointly when assessing robustness and reproducibility.
