# JAMA 

This directory contains the Python implementation and results for the **JAMA lung-cancer miRNA dataset** used in the batch-aware molecular classifier validation project.

## Data Source

The dataset is derived from:

**Fehlmann, T. et al. (2020). Evaluating the Use of Circulating MicroRNA Profiles for Lung Cancer Detection in Symptomatic Patients. JAMA Oncology.**

- Article: https://doi.org/10.1001/jamaoncol.2020.0001
- Expression data: https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-8026

The original study analyzed genome-wide miRNA expression profiles from **3,046 whole-blood samples** and evaluated three lung-cancer classification scenarios.

### Local data files

The current workflow uses:

- `expression_matrix.csv`  
  Main miRNA expression matrix used by the Python analysis.

- `expressed_mirnas.txt`  
  List of expressed miRNAs retained for downstream analysis.

- `eTable1.xlsx`  
  Sample-level clinical metadata, including disease group and tumor stage information used to define the three classification scenarios.

These input files are expected in the `2Jama/` directory when running the pipeline.

---

## Preprocessing

In the original study, the microarray expression data were:

1. background-subtracted;
2. summarized across replicate measurements using the median;
3. filtered to expressed miRNAs;
4. quantile-normalized.

The published analysis retained **1,183 expressed miRNAs**.

The original study also evaluated technical batch effects related to sampling/study site, profiling site, and biochip ID.

The current repository starts from the processed expression matrix and focuses on downstream **batch-aware validation**, rather than reproducing the raw microarray preprocessing pipeline.

---

## Main Files

```text
2Jama/
├── README_Jama.md
├── SSV_Jama_Python_pro.ipynb
├── SSV_Jama_Python_pro.py
├── jama.py
└── results_pro_jama/
```

### `SSV_Jama_Python_pro.py`

Main executable JAMA analysis pipeline.

It performs:

- data and metadata alignment;
- construction of the three medical scenarios;
- JAMA batch / meta-batch construction;
- Natural and Class-stratified splitting;
- normalization and classifier evaluation;
- batch-aware validation;
- QC and result generation.

### `jama.py`

Contains JAMA-specific helper functions, including:

- expression-data loading;
- sample and clinical-group parsing;
- JAMA-specific batch inference;
- meta-batch construction and validation;
- JAMA-specific diagnostic plotting.

Generic SSV functions shared across datasets are stored separately in the repository-level `src/` directory.

### `SSV_Jama_Python_pro.ipynb`

Notebook version of the JAMA analysis, retained mainly for interactive inspection and visualization.

---

## Medical Classification Scenarios

The analysis follows the three diagnostic settings defined in the original study.

| Scenario | Positive class | Negative class | Interpretation |
|---|---|---|---|
| **S1** | Lung cancer | NTLD + other diseases + unaffected controls | Broad lung-cancer detection |
| **S2** | Lung cancer | Nontumor lung disease (NTLD) | Clinically focused differential diagnosis |
| **S3** | Stage I/II lung cancer | All individuals without lung cancer | Early-stage lung-cancer detection |

The original comparisons were approximately:

- **S1:** 606 vs 2440
- **S2:** 606 vs 593
- **S3:** 194 vs 2440

Because these scenarios represent different medical classification problems, absolute error levels should not be interpreted as batch effects alone.

---

## Validation Design

Each scenario is evaluated using two complementary protocols.

### Natural

Preserves the observed class composition within the batch structure.

It represents the primary deployment-oriented analysis and therefore reflects the combined effect of:

- batch/domain shift;
- class-prevalence differences.

### Class-stratified

Makes class composition across batches more comparable.

It is used as a sensitivity / mechanism analysis to reduce the contribution of class-prevalence shift and isolate the remaining batch/domain-associated degradation more clearly.

---

## Results Structure

All JAMA Pro results are stored under:

```text
results_pro_jama/
├── jama_pro_s1_broad/
├── jama_pro_s2_broad/
├── jama_pro_s3_broad/
└── ssv_pro_manifest.csv
```

- `ssv_pro_manifest.csv`  
  Records the generated analysis outputs and their locations.

All three scenario directories follow the same general structure.

Using **Scenario 1** as an example:

```text
results_pro_jama/
└── jama_pro_s1_broad/
    ├── class_stratified/
    ├── natural/
    ├── qc/
    └── JAMA_Pro_S1_Result_Interpretation.md
```

### `class_stratified/`

Contains performance results from the Class-stratified protocol.

### `natural/`

Contains performance results under the observed class composition.

### `qc/`

Contains batch-structure, split, and quality-control diagnostics.

### Scenario result interpretation

The Markdown file stored directly inside each scenario directory is the **main scientific interpretation of that scenario**.

It integrates:

- Natural and Class-stratified results;
- raw error and balanced error;
- changes as train/test batch separation becomes stricter;
- classifier and normalization behavior;
- batch-structure diagnostics;
- the distinction between intrinsic medical classification difficulty and additional batch-associated degradation.

S2 and S3 follow the same directory structure as S1.

---

## Key Result Interpretation Files

For the scientific interpretation of the JAMA results, start with:

- **Scenario 1 — Broad lung-cancer detection**  
  [`results_pro_jama/jama_pro_s1_broad/JAMA_Pro_S1_Result_Interpretation.md`](results_pro_jama/jama_pro_s1_broad/JAMA_Pro_S1_Result_Interpretation.md)

- **Scenario 2 — Lung cancer vs nontumor lung disease**  
  [`results_pro_jama/jama_pro_s2_broad/JAMA_Pro_S2_Result_Interpretation.md`](results_pro_jama/jama_pro_s2_broad/JAMA_Pro_S2_Result_Interpretation.md)

- **Scenario 3 — Early-stage lung-cancer detection**  
  [`results_pro_jama/jama_pro_s3_broad/JAMA_Pro_S3_Result_Interpretation.md`](results_pro_jama/jama_pro_s3_broad/JAMA_Pro_S3_Result_Interpretation.md)

These files should be read together with the corresponding figures and numerical outputs in each scenario directory.

---

## Cross-Scenario Interpretation

The three JAMA scenarios separate two related questions:

1. **How difficult is the underlying medical classification task?**
2. **How much additional degradation occurs as train/test batches become more separated?**

Conceptually:

```text
observed validation difficulty
=
baseline classification difficulty
+
batch-associated degradation
+
class-composition effects
```

The Natural and Class-stratified protocols are therefore interpreted jointly rather than relying on a single validation curve.
