# Yokoi Dataset

## Source

The dataset used in this analysis is derived from:

**Yokoi, A. et al. (2018). Integrated extracellular microRNA profiling for ovarian cancer screening. Nature Communications, 9, 4319.**

- Article: https://doi.org/10.1038/s41467-018-06434-4
- GEO accession: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE106817

The original study generated serum miRNA microarray expression profiles and made the corresponding data publicly available through GEO. :chatgpt-content-reference{index="0"}

## Files Used in This Repository

### `combined_metadata_expression_GSE106817.csv`

Main analysis dataset containing the processed miRNA expression matrix together with sample-level information required by the Python workflow.

**Data type:** processed miRNA microarray expression data.

### `Yokoi_meta.csv`

Sample metadata used to define:

- outcome labels;
- batch/group variables;
- sample alignment for downstream analysis.

## Preprocessing

The original study performed microarray quality control, background correction, and cross-array normalization before downstream analysis. :chatgpt-content-reference{index="1"}

The current Python workflow starts from these processed expression values rather than raw microarray data.

Additional preprocessing in this repository is limited to the steps required for the downstream validation workflow, including:

- metadata alignment;
- train/test splitting;
- normalization or transformation used by the classifier pipeline;
- batch-aware validation.

## Role in This Project

The Yokoi dataset is used to evaluate how batch structure and data-splitting strategies affect molecular classifier performance, repeatability, and reproducibility.
