# Batch-Aware_Validation_of_Molecular_Classifiers
Python reproduction and methodological extension of the earlier **CrossBatch-ML (R workflow)** for studying how stronger train/test batch separation affects molecular-classifier performance.

The original framework showed that classification error can increase as validation moves from mixed/random-like splits toward strict cross-batch evaluation. The current Python Pro workflow keeps this core design but adds several robustness improvements to better separate batch effects from other sources of distribution shift. 

## Datasets

The framework is currently applied to three published molecular datasets:

- **Yokoi et al. (2018)** — *Integrated extracellular microRNA profiling for ovarian cancer screening*  
  https://doi.org/10.1038/s41467-018-06434-4

- **Fehlmann et al. (2020)** — *Evaluating the Use of Circulating MicroRNA Profiles for Lung Cancer Detection in Symptomatic Patients*  
  https://doi.org/10.1001/jamaoncol.2020.0001

- **Wang et al. (2024)** — *Terminal modifications independent cell-free RNA sequencing enables sensitive early cancer detection and classification*  
  https://doi.org/10.1038/s41467-023-44461-y

Dataset-specific inputs, preprocessing, tasks, and results are documented in `1Yokoi/`, `2Jama/`, and `3Gou/`.

---

## 🍎 Main Methodological Improvements

The original R workflow controlled overall batch size but did not explicitly separate class-prevalence differences from batch/domain shift. The Python Pro version adds four main extensions: 

### 1. Natural + Class-Stratified Validation

Two complementary protocols are used:

- **Natural**: preserves the observed class composition and measures real-world cross-batch difficulty.
- **Class-stratified**: approximately balances class prevalence across batches to isolate batch/domain-associated degradation more clearly.

Together they distinguish:

```text
overall deployment degradation
vs.
degradation remaining after controlling class composition
```

### 2. Batch–Class QC

The workflow explicitly checks whether batch membership and class label are associated, helping identify potential batch–class confounding.

### 3. Bidirectional Batch Transfer

Both directions are evaluated separately:

```text
Batch 1 → Batch 2
Batch 2 → Batch 1
```

Their difference is summarized by a **direction gap**, making asymmetric cross-batch generalization visible.

### 4. Equal-Weight Direction Aggregation + Balanced Error

The two transfer directions are aggregated with fixed equal weight rather than being randomly mixed across repetitions, reducing seed-driven direction imbalance.

The workflow also reports both:

- **Raw error** — overall misclassification rate
- **Balanced error** — equal-weight performance across classes

This is especially important when the class distribution is imbalanced. 

---

## Evaluation Views

The Pro workflow combines two validation protocols with two error metrics:

| Protocol | Metric | Interpretation |
|---|---|---|
| Class-stratified | Balanced error | Batch-associated degradation with class composition controlled |
| Class-stratified | Raw error | Conventional error after class balancing |
| Natural | Balanced error | Realistic transfer with equal class weighting |
| Natural | Raw error | Overall real-world deployment error |

This creates a more interpretable framework for separating:

```text
batch/domain shift
class-prevalence shift
class imbalance
directional transfer asymmetry
```

---

## Repository Structure

```text
Batch-Aware_Validation_of_Molecular_Classifiers/
├── src/                     # shared SSV Python modules
├── 1Yokoi/                  # Yokoi analysis and results
├── 2Jama/                   # JAMA analysis and results
├── 3Gou/                    # Gou analysis and results
├── SSV_Python_Reproduction_pro.ipynb
├── README.md
└── .gitignore
```

Shared methodology is stored in `src/`, while dataset-specific loading, preprocessing, analysis, and interpretation remain inside each dataset directory.

---

## Scope

This repository focuses on **validation methodology**, not on reproducing the original predictive models from the source publications.

The main objective is to determine whether molecular-classifier performance remains reliable when validation becomes increasingly independent of the batch structure present in the training data.

In short:

```text
CrossBatch-ML
    ↓
Python reproduction
    ↓
dual-track validation
    ↓
bidirectional transfer
    ↓
balanced evaluation
    ↓
multi-dataset reproducibility analysis
```
