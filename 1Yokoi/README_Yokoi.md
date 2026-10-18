# Yokoi

This directory contains the Python implementation and results for the **Yokoi** dataset used in the batch-aware molecular classifier validation project.

## Data Source

The dataset is derived from:

**Yokoi, A. et al. (2018). Integrated extracellular microRNA profiling for ovarian cancer screening. Nature Communications, 9, 4319.**

- Article: https://doi.org/10.1038/s41467-018-06434-4
- GEO accession: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE106817

The original study generated serum miRNA microarray expression profiles. The current project starts from processed expression data and focuses on downstream classifier validation rather than raw microarray processing.

### Local data files

The analysis expects the following files:

- `combined_metadata_expression_GSE106817.csv`  
  Main processed miRNA expression dataset used by the Python workflow.

- `Yokoi_meta.csv`  
  Sample-level metadata used for outcome definition, batch construction, and sample alignment.

The large expression dataset is not stored directly in this GitHub repository because of file-size constraints. It should be placed in the `1Yokoi/` directory before running the notebooks.

---

## Main Analysis Files

```text
1Yokoi/
├── README.md
├── SSV_Python_Reproduction_pro.ipynb
├── SSV_Yokoi_Python_pro.ipynb
└── results_pro/
```

### `SSV_Python_Reproduction_pro.ipynb`

Contains the general Python reproduction of the SSV analysis framework shared across datasets.

### `SSV_Yokoi_Python_pro.ipynb`

Contains the Yokoi-specific implementation, including:

- dataset loading and metadata alignment;
- medical scenario construction;
- batch definition;
- train/test split generation;
- normalization and classifier evaluation;
- Natural and Class-stratified validation;
- result generation and quality-control outputs.

---

## Medical Classification Scenarios

The Yokoi analysis contains three related but clinically distinct classification tasks.

| Scenario | Positive class | Negative class | Interpretation |
|---|---|---|---|
| **S1** | Ovarian cancer | Non-cancer | General ovarian-cancer detection |
| **S2** | Ovarian cancer | Other cancers | Cancer-type discrimination / differential diagnosis |
| **S3** | Ovarian cancer | Benign ovarian disease + borderline ovarian tumors + non-cancer | Clinically heterogeneous differential diagnosis |

These scenarios should not be compared only by their absolute error levels because they represent different underlying medical classification problems.

---

## Validation Design

Each scenario is analyzed using two complementary protocols.

### Natural

Preserves the observed class composition within each batch.

This is the primary real-world analysis and reflects the total difficulty of cross-batch generalization, including both:

- batch/domain shift;
- class-prevalence differences.

### Class-stratified

Makes class composition across batches more comparable.

This serves as a sensitivity / mechanism analysis by reducing the effect of class-prevalence shift and allowing the remaining degradation to be interpreted more directly in terms of batch/domain shift.

The comparison between these two tracks is therefore an important part of the analysis.

---

## Results Structure

All Yokoi outputs are stored under:

```text
results_pro/
├── yokoi_s1_positive/
├── yokoi_s2_positive/
├── yokoi_s3_positive/
├── final_pro_summary.csv
├── ssv_pro_manifest.csv
└── yokoi_qc_figure_manifest.csv
```

### Summary files

- `final_pro_summary.csv`  
  Consolidated numerical results across scenarios, split settings, normalization methods, and classifiers.

- `ssv_pro_manifest.csv`  
  Manifest of the generated SSV analysis outputs.

- `yokoi_qc_figure_manifest.csv`  
  Index of Yokoi quality-control figures and diagnostics.

---

## Scenario-Level Results

All three scenarios follow the same directory structure.

Using **S1** as an example:

```text
results_pro/
└── yokoi_s1_positive/
    ├── class_stratified/
    ├── natural/
    ├── qc/
    └── S1_dual_track_result_interpretation.md
```

### `class_stratified/`

Contains results from the Class-stratified protocol, including performance curves and split-level outputs.

### `natural/`

Contains results from the Natural protocol.

### `qc/`

Contains quality-control and diagnostic outputs used to verify the split construction and analysis behavior.

### Result interpretation `.md`

The scenario-level Markdown file is the **main scientific interpretation of the corresponding results**.

It integrates the Natural and Class-stratified analyses and discusses:

- changes in error as batch separation becomes stronger;
- raw error versus balanced error;
- the effect of class composition;
- classifier-specific behavior;
- the distinction between baseline medical classification difficulty and additional batch-associated degradation.

The S2 and S3 directories follow the same structure as S1.

---

## Key Result Interpretation Files

For interpretation of the main results, start with these files:

- **Scenario 1:**  
  [`results_pro/yokoi_s1_positive/S1_dual_track_result_interpretation.md`](results_pro/yokoi_s1_positive/S1_dual_track_result_interpretation.md)

- **Scenario 2:**  
  [`results_pro/yokoi_s2_positive/S2_dual_track_result_interpretation.md`](results_pro/yokoi_s2_positive/S2_dual_track_result_interpretation.md)

- **Scenario 3:**  
  [`results_pro/yokoi_s3_positive/S3_dual_track_result_interpretation.md`](results_pro/yokoi_s3_positive/S3_dual_track_result_interpretation.md)

These interpretation files should be read together with the corresponding numerical outputs and figures.

---

## Cross-Scenario Interpretation

The three scenarios address two related but distinct questions:

1. **How difficult is the underlying medical classification problem?**
2. **How much additional degradation occurs as train/test batches become more separated?**

The current analysis therefore distinguishes:

```text
baseline classification difficulty
                +
additional degradation associated with batch separation
```

rather than interpreting the absolute error level alone as a measure of batch effect.

At a high level:

- **S1** represents the broad cancer-vs-non-cancer detection setting.
- **S2** is a more difficult cancer-type discrimination problem.
- **S3** is a heterogeneous differential-diagnosis problem in which class composition plays a particularly important role.

The Natural and Class-stratified analyses are considered jointly when interpreting these effects.
