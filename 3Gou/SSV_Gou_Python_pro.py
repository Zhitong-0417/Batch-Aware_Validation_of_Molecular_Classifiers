#!/usr/bin/env python3
"""Gou Pro analysis converted from SSV_Gou_Python_pro.ipynb.

The script preserves the notebook execution order and Gou-specific analysis logic.
It keeps the shared SSV runtime as a repository-level dependency and converts
notebook-only cells to standard Python where needed.
"""

from __future__ import annotations
# %% [markdown] Cell 1
# # Gou Pro SSV Workflow
# 
# Gou Pro extension and cross-dataset reproducibility framework. This notebook preserves the Gou selected-feature scientific workflow, adds batch/class structure diagnostics, and uses the shared Pro SSV runtime for bidirectional robustness analysis.

# %% [markdown] Cell 2
# ## Setup and paths

# %% Cell 3
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent if HERE.name.lower().startswith(("3gou", "03_gou")) else HERE
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import *
from gou import *

import re
from time import perf_counter

try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown = None

    def display(value):
        print(value)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def resolve_base_dir() -> Path:
    cwd = Path.cwd().resolve()
    required = {"combination_selected_mi_sn_sno_lung.csv", "SSV Gou.Rmd", "SSV summary.R"}
    if cwd.name == "3Gou" and required <= {path.name for path in cwd.iterdir()}:
        return cwd
    candidate = cwd / "3Gou"
    if candidate.exists() and required <= {path.name for path in candidate.iterdir()}:
        return candidate.resolve()
    raise RuntimeError("Launch this notebook from 3Gou or from its parent RNASeq_with_Python directory.")


BASE_DIR = resolve_base_dir()
DATA_PATH = BASE_DIR / "combination_selected_mi_sn_sno_lung.csv"
PRO_NOTEBOOK_PATH = BASE_DIR / "SSV_Gou_Python_pro.ipynb"
BASELINE_SOURCE_NOTEBOOK = BASE_DIR / "SSV_Gou_Python.ipynb"
PRO_RUNTIME_NOTEBOOK = BASE_DIR.parent / "SSV_Python_Reproduction_pro.ipynb"
GOU_PRO_OUTPUT_ROOT = BASE_DIR / "results_pro_gou"
GOU_PRO_QC_DIR = GOU_PRO_OUTPUT_ROOT / "qc"

GOU_STRUCTURE_SEED = 12345
GOU_QC_BOXPLOT_MAX_SAMPLES_PER_STRATUM = 40
DATA_TYPE_RNASEQ = False


def assert_inside_base(path):
    path = Path(path).resolve()
    if path != BASE_DIR and BASE_DIR not in path.parents:
        raise ValueError(f"Refusing path outside BASE_DIR: {path}")
    return path


for expected_path in (DATA_PATH, BASELINE_SOURCE_NOTEBOOK, PRO_RUNTIME_NOTEBOOK):
    assert expected_path.exists(), f"Missing required input: {expected_path}"
assert_inside_base(GOU_PRO_OUTPUT_ROOT)
assert_inside_base(GOU_PRO_QC_DIR)

print(f"BASE_DIR: {BASE_DIR}")
print(f"Baseline Gou source: {BASELINE_SOURCE_NOTEBOOK.name}")
print(f"Shared Pro runtime: {PRO_RUNTIME_NOTEBOOK.name}")
print(f"Gou Pro output root: {GOU_PRO_OUTPUT_ROOT}")

# %% [markdown] Cell 4
# ## Load shared Pro runtime
# 
# The generic Pro runtime is loaded from `SSV_Python_Reproduction_pro.ipynb`. This notebook does not execute any other dataset-specific notebook as a runtime.

# %% Cell 5

_RUNTIME_REQUIRED = (
    "MolecularDataset",
    "validate_molecular_dataset",
    "summarize_batch_structure",
    "plot_poor_marker_samplewise_boxplot",
    "run_ssv_pro_experiment_batch",
    "run_ssv_pro_bidirectional",
    "aggregate_equal_weight_directions",
    "save_ssv_pro_artifacts",
)




load_shared_pro_runtime(PRO_RUNTIME_NOTEBOOK)
print("Shared Pro runtime loaded.")
print("Required objects:", ", ".join(_RUNTIME_REQUIRED))

# %% [markdown] Cell 6
# ## Load Gou selected-feature data
# 
# The Gou input is the paper-selected miRNA/snRNA/snoRNA lung panel. It is loaded as features x samples and transposed to samples x features for Gou-specific preprocessing.

# %% Cell 7
lung_normal_data = pd.read_csv(DATA_PATH, index_col=0)
lung_normal_data.index = lung_normal_data.index.astype(str)
lung_normal_data.columns = lung_normal_data.columns.astype(str)

raw_data = lung_normal_data.T.copy()
raw_data.index.name = "sample_id"
raw_data.columns.name = "feature_id"
sample_ids = raw_data.index.copy()

orientation_audit = pd.DataFrame(
    [
        {"stage": "original CSV", "rows": lung_normal_data.shape[0], "columns": lung_normal_data.shape[1], "orientation": "features x samples"},
        {"stage": "after transpose", "rows": raw_data.shape[0], "columns": raw_data.shape[1], "orientation": "samples x features"},
    ]
)
display(orientation_audit)

# %% [markdown] Cell 8
# ## Construct LC/NOR and original batch labels

# %% Cell 9
class_label = pd.Series(index=sample_ids, dtype="object", name="class_label")
class_label.loc[class_label.index.str.contains(r"^LC_", regex=True)] = "LC"
class_label.loc[class_label.index.str.contains(r"^NOR_", regex=True)] = "NOR"
class_id = class_label.map({"LC": 0, "NOR": 1}).astype("Int64").rename("class")

original_batch = pd.Series(
    [re.sub(r"^([A-Z]+_[A-Z]+)[_0-9]*$", r"\1", sample_id) for sample_id in sample_ids],
    index=sample_ids,
    name="original_batch",
)

expected_classes = {"LC", "NOR"}
expected_batches = {"LC_SZDE", "LC_SZBU", "NOR_SZBA", "NOR_SZDW"}
if class_label.isna().any():
    raise ValueError(f"Unlabeled samples: {class_label[class_label.isna()].index.tolist()[:5]}")
if set(class_label.unique()) != expected_classes:
    raise ValueError(f"Unexpected class labels: {sorted(class_label.unique())}")
if set(original_batch.unique()) != expected_batches:
    raise ValueError(f"Unexpected original batches: {sorted(original_batch.unique())}")
if not raw_data.index.equals(class_id.index) or not raw_data.index.equals(original_batch.index):
    raise ValueError("Expression rows and labels are misaligned")

display(pd.DataFrame({"class_label": class_label, "class_id": class_id.astype(int)}).value_counts(["class_label", "class_id"]).rename("n_samples").reset_index())
display(pd.crosstab(original_batch, class_label))

# %% [markdown] Cell 10
# ## Feature QC

# %% Cell 11


raw_data_counts, filtering_audit = filter_gou_features(raw_data, min_present_prop=0.05)
display(filtering_audit)
print(f"Filtered selected-feature matrix: {raw_data_counts.shape[0]} samples x {raw_data_counts.shape[1]} features")

# %% [markdown] Cell 12
# ## TMM + log2 CPM transformation
# 
# Gou uses TMM normalization followed by log2 CPM before the shared non-RNA-seq-style SSV layer. No second log transform or RNA-seq count-normalization branch is applied later.

# %% Cell 13








normalized_data, normalization_audit = edgeR_style_logcpm_tmm(raw_data_counts, prior_count=1.0)
if normalized_data.shape != raw_data_counts.shape:
    raise ValueError("Normalization changed dimensions")
if not normalized_data.index.equals(raw_data_counts.index):
    raise ValueError("Normalization changed sample order")
if not normalized_data.columns.equals(raw_data_counts.columns):
    raise ValueError("Normalization changed feature order")
if not np.isfinite(normalized_data.to_numpy()).all():
    raise ValueError("Normalized matrix contains non-finite values")
display(normalization_audit.head())

# %% [markdown] Cell 14
# ## Construct early/late meta-batches

# %% Cell 15
EARLY_ORIGINAL_BATCHES = ("LC_SZDE", "NOR_SZBA")
TEMPORAL_BY_ORIGINAL = {
    "LC_SZDE": "early",
    "NOR_SZBA": "early",
    "LC_SZBU": "late",
    "NOR_SZDW": "late",
}
temporal_batch = original_batch.map(TEMPORAL_BY_ORIGINAL).rename("temporal_batch")
if temporal_batch.isna().any():
    raise ValueError("Temporal batch mapping failed")
if not normalized_data.index.equals(temporal_batch.index):
    raise ValueError("Temporal batch labels are not aligned")

temporal_mapping = pd.DataFrame({"original_batch": original_batch, "temporal_batch": temporal_batch}).drop_duplicates().sort_values(["temporal_batch", "original_batch"]).reset_index(drop=True)
display(temporal_mapping)

# %% [markdown] Cell 16
# ## Validate Gou Pro data contracts

# %% Cell 17
selected_expression = normalized_data.T.copy()
selected_expression.index.name = "feature_id"
selected_expression.columns.name = "sample_id"

temporal_pro_batch = temporal_batch.map({"early": "batch_1", "late": "batch_2"}).rename("batch")
if set(temporal_pro_batch.unique()) != {"batch_1", "batch_2"}:
    raise ValueError("Pro temporal batch labels must be binary batch_1/batch_2")

gou_pro_dataset = MolecularDataset(
    expression=selected_expression,
    class_labels=class_id.astype(int).rename("class"),
    batch_labels=temporal_pro_batch,
)
validate_molecular_dataset(gou_pro_dataset)

gou_original_batch_dataset = MolecularDataset(
    expression=selected_expression,
    class_labels=class_id.astype(int).rename("class"),
    batch_labels=original_batch.rename("batch"),
)
validate_molecular_dataset(gou_original_batch_dataset, min_batches=4)

gou_temporal_named_dataset = MolecularDataset(
    expression=selected_expression,
    class_labels=class_id.astype(int).rename("class"),
    batch_labels=temporal_batch.rename("batch"),
)
validate_molecular_dataset(gou_temporal_named_dataset)

print(f"Gou Pro dataset: {gou_pro_dataset.expression.shape[0]} selected features x {gou_pro_dataset.expression.shape[1]} samples")
print("Class counts:", gou_pro_dataset.class_labels.value_counts().sort_index().to_dict())
print("Temporal Pro batch counts:", gou_pro_dataset.batch_labels.value_counts().sort_index().to_dict())

# %% [markdown] Cell 18
# ## Gou batch/class QC
# 
# Diagnostics are descriptive. They expose label balance, batch balance, and label-batch association at both the original source-batch level and the early/late Pro level. Strong association is an identifiability/confounding warning, not an automatic exclusion rule.

# %% Cell 19

GOU_PRO_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
GOU_PRO_QC_DIR.mkdir(parents=True, exist_ok=True)

original_qc_summary, original_contingency = summarize_batch_structure(gou_original_batch_dataset, scenario="gou_original_batches")
temporal_qc_summary, temporal_contingency = summarize_batch_structure(gou_temporal_named_dataset, scenario="gou_temporal_early_late")
pro_qc_summary, pro_contingency = summarize_batch_structure(gou_pro_dataset, scenario="gou_pro_batch_1_batch_2")

qc_summary = pd.concat([original_qc_summary, temporal_qc_summary, pro_qc_summary], ignore_index=True)
qc_view = qc_summary[
    [
        "scenario",
        "domain",
        "metric",
        "ratio",
        "minority_proportion",
        "cramers_v",
        "abs_prevalence_difference",
        "chi_square_p_value",
        "status",
    ]
]

original_contingency_named = original_contingency.rename(columns={"0": "LC", "1": "NOR"})
temporal_contingency_named = temporal_contingency.rename(columns={"0": "LC", "1": "NOR"}).reindex(["early", "late"])
pro_contingency_named = pro_contingency.rename(columns={"0": "LC", "1": "NOR"}).reindex(["batch_1", "batch_2"])

sample_provenance = pd.DataFrame(
    {
        "sample_id": sample_ids,
        "class_label": class_label.loc[sample_ids].to_numpy(),
        "class_id": class_id.loc[sample_ids].to_numpy(),
        "original_batch": original_batch.loc[sample_ids].to_numpy(),
        "temporal_batch": temporal_batch.loc[sample_ids].to_numpy(),
        "pro_batch": temporal_pro_batch.loc[sample_ids].to_numpy(),
    }
)

display(qc_view)
display(original_contingency_named)
display(temporal_contingency_named)
display(pro_contingency_named)

qc_view.to_csv(GOU_PRO_QC_DIR / "gou_pro_qc_summary.csv", index=False)
original_contingency_named.to_csv(GOU_PRO_QC_DIR / "gou_pro_original_batch_class_contingency.csv")
temporal_contingency_named.to_csv(GOU_PRO_QC_DIR / "gou_pro_temporal_batch_class_contingency.csv")
pro_contingency_named.to_csv(GOU_PRO_QC_DIR / "gou_pro_batch1_batch2_class_contingency.csv")
sample_provenance.to_csv(GOU_PRO_QC_DIR / "gou_pro_sample_provenance.csv", index=False)

high_association = qc_view.loc[qc_view["domain"].eq("label_batch_association") & qc_view["status"].eq("HIGH")]
if not high_association.empty:
    warning_text = "Gou Pro warning: biological label and batch are strongly associated in at least one diagnostic level; interpret batch-aware performance as potentially confounded."
    if Markdown is not None:
        display(Markdown(f"**{warning_text}**"))
    else:
        print(warning_text)

# %% Cell 20
# ============================================================
# Gou Pro UMAP QC
# Diagnostic visualization only:
# UMAP coordinates are NOT used as classifier inputs.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import umap
except ImportError as exc:
    raise ImportError(
        "UMAP is required for this QC cell. "
        "Install it with: pip install umap-learn"
    ) from exc


# ------------------------------------------------------------
# 1. Output paths
# ------------------------------------------------------------

GOU_PRO_QC_DIR.mkdir(parents=True, exist_ok=True)

UMAP_CLASS_PATH = GOU_PRO_QC_DIR / "gou_pro_umap_by_class.png"
UMAP_ORIGINAL_BATCH_PATH = GOU_PRO_QC_DIR / "gou_pro_umap_by_original_batch.png"
UMAP_TEMPORAL_PATH = GOU_PRO_QC_DIR / "gou_pro_umap_by_temporal_batch.png"
UMAP_PRO_BATCH_PATH = GOU_PRO_QC_DIR / "gou_pro_umap_by_pro_batch.png"
UMAP_COORDINATES_PATH = GOU_PRO_QC_DIR / "gou_pro_umap_coordinates.csv"

for path in (
    UMAP_CLASS_PATH,
    UMAP_ORIGINAL_BATCH_PATH,
    UMAP_TEMPORAL_PATH,
    UMAP_PRO_BATCH_PATH,
    UMAP_COORDINATES_PATH,
):
    assert_inside_base(path)


# ------------------------------------------------------------
# 2. Validate sample alignment
# ------------------------------------------------------------

sample_order = selected_expression.columns.astype(str)

if not selected_expression.columns.equals(class_label.index):
    raise ValueError("selected_expression and class_label are not aligned")

if not selected_expression.columns.equals(original_batch.index):
    raise ValueError("selected_expression and original_batch are not aligned")

if not selected_expression.columns.equals(temporal_batch.index):
    raise ValueError("selected_expression and temporal_batch are not aligned")

if not selected_expression.columns.equals(temporal_pro_batch.index):
    raise ValueError("selected_expression and temporal_pro_batch are not aligned")


# ------------------------------------------------------------
# 3. Build samples × features UMAP input
# ------------------------------------------------------------

X_umap_input = selected_expression.T.astype(float).copy()

if X_umap_input.shape[0] != len(sample_order):
    raise ValueError("UMAP input sample count mismatch")

if not np.isfinite(X_umap_input.to_numpy()).all():
    raise ValueError("UMAP input contains NaN or infinite values")

print(
    f"UMAP input: "
    f"{X_umap_input.shape[0]} samples × "
    f"{X_umap_input.shape[1]} selected features"
)


# ------------------------------------------------------------
# 4. Fit ONE deterministic UMAP embedding
# ------------------------------------------------------------

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.2,
    metric="cosine",
    random_state=GOU_STRUCTURE_SEED,
)

embedding = umap_model.fit_transform(X_umap_input)

if embedding.shape != (X_umap_input.shape[0], 2):
    raise ValueError(
        f"Unexpected UMAP shape: {embedding.shape}; "
        f"expected {(X_umap_input.shape[0], 2)}"
    )

if not np.isfinite(embedding).all():
    raise ValueError("UMAP embedding contains NaN or infinite coordinates")


umap_df = pd.DataFrame(
    embedding,
    index=sample_order,
    columns=["UMAP1", "UMAP2"],
)

umap_df["class_label"] = class_label.loc[sample_order].astype(str).to_numpy()
umap_df["original_batch"] = original_batch.loc[sample_order].astype(str).to_numpy()
umap_df["temporal_batch"] = temporal_batch.loc[sample_order].astype(str).to_numpy()
umap_df["pro_batch"] = temporal_pro_batch.loc[sample_order].astype(str).to_numpy()

umap_df.index.name = "sample_id"
umap_df.to_csv(UMAP_COORDINATES_PATH)


# ------------------------------------------------------------
# 5. Shared plotting helper
# ------------------------------------------------------------



# ------------------------------------------------------------
# 6. UMAP by biological class
# ------------------------------------------------------------

fig_umap_class, ax_umap_class = plot_gou_umap(
    coordinates=umap_df,
    group_column="class_label",
    title="Gou Pro UMAP by biological class",
    output_path=UMAP_CLASS_PATH,
)

display(fig_umap_class)
plt.close(fig_umap_class)


# ------------------------------------------------------------
# 7. UMAP by original four-batch structure
# ------------------------------------------------------------

fig_umap_original, ax_umap_original = plot_gou_umap(
    coordinates=umap_df,
    group_column="original_batch",
    title="Gou Pro UMAP by original batch",
    output_path=UMAP_ORIGINAL_BATCH_PATH,
)

display(fig_umap_original)
plt.close(fig_umap_original)


# ------------------------------------------------------------
# 8. UMAP by early / late temporal batch
# ------------------------------------------------------------

fig_umap_temporal, ax_umap_temporal = plot_gou_umap(
    coordinates=umap_df,
    group_column="temporal_batch",
    title="Gou Pro UMAP by temporal batch",
    output_path=UMAP_TEMPORAL_PATH,
)

display(fig_umap_temporal)
plt.close(fig_umap_temporal)


# ------------------------------------------------------------
# 9. UMAP by Pro binary batch_1 / batch_2
# ------------------------------------------------------------

fig_umap_pro, ax_umap_pro = plot_gou_umap(
    coordinates=umap_df,
    group_column="pro_batch",
    title="Gou Pro UMAP by Pro batch",
    output_path=UMAP_PRO_BATCH_PATH,
)

display(fig_umap_pro)
plt.close(fig_umap_pro)


# ------------------------------------------------------------
# 10. Final integrity checks
# ------------------------------------------------------------

expected_umap_files = (
    UMAP_CLASS_PATH,
    UMAP_ORIGINAL_BATCH_PATH,
    UMAP_TEMPORAL_PATH,
    UMAP_PRO_BATCH_PATH,
    UMAP_COORDINATES_PATH,
)

for path in expected_umap_files:
    if not path.exists():
        raise FileNotFoundError(f"Missing UMAP artifact: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Empty UMAP artifact: {path}")


umap_audit = pd.DataFrame(
    {
        "artifact": [
            "coordinates",
            "by_class",
            "by_original_batch",
            "by_temporal_batch",
            "by_pro_batch",
        ],
        "path": [
            str(UMAP_COORDINATES_PATH),
            str(UMAP_CLASS_PATH),
            str(UMAP_ORIGINAL_BATCH_PATH),
            str(UMAP_TEMPORAL_PATH),
            str(UMAP_PRO_BATCH_PATH),
        ],
        "exists": [
            UMAP_COORDINATES_PATH.exists(),
            UMAP_CLASS_PATH.exists(),
            UMAP_ORIGINAL_BATCH_PATH.exists(),
            UMAP_TEMPORAL_PATH.exists(),
            UMAP_PRO_BATCH_PATH.exists(),
        ],
    }
)

display(umap_audit)

print("PASS — Gou Pro UMAP QC complete")
print(f"Saved under: {GOU_PRO_QC_DIR}")

# %% [markdown] Cell 21
# ## Sample-wise selected-feature expression boxplots
# 
# These plots use the already TMM-normalized log2 CPM selected-feature matrix. They are selected-feature diagnostics, not negative-control diagnostics.

# %% Cell 22



ORIGINAL_BOXPLOT_PATH = GOU_PRO_QC_DIR / "gou_pro_original_batch_samplewise_boxplot.png"
TEMPORAL_BOXPLOT_PATH = GOU_PRO_QC_DIR / "gou_pro_temporal_batch_samplewise_boxplot.png"

fig_original, ax_original, ordered_original_samples = plot_selected_feature_samplewise_boxplot(
    selected_expression,
    class_label.rename("class"),
    original_batch.rename("batch"),
    title="Gou selected-feature log2 CPM by original batch and class",
    output_path=ORIGINAL_BOXPLOT_PATH,
    max_samples_per_stratum=GOU_QC_BOXPLOT_MAX_SAMPLES_PER_STRATUM,
)
display(fig_original)

fig_temporal, ax_temporal, ordered_temporal_samples = plot_selected_feature_samplewise_boxplot(
    selected_expression,
    class_label.rename("class"),
    temporal_batch.rename("batch"),
    title="Gou selected-feature log2 CPM by early/late batch and class",
    output_path=TEMPORAL_BOXPLOT_PATH,
    max_samples_per_stratum=GOU_QC_BOXPLOT_MAX_SAMPLES_PER_STRATUM,
)
display(fig_temporal)

ordered_original_samples.to_csv(GOU_PRO_QC_DIR / "gou_pro_original_batch_boxplot_sample_order.csv", index=False)
ordered_temporal_samples.to_csv(GOU_PRO_QC_DIR / "gou_pro_temporal_boxplot_sample_order.csv", index=False)

display(
    pd.DataFrame(
        [
            {"plot": "original_batch", "retained_samples_for_plot": int(ordered_original_samples.shape[0]), "file": str(ORIGINAL_BOXPLOT_PATH)},
            {"plot": "temporal", "retained_samples_for_plot": int(ordered_temporal_samples.shape[0]), "file": str(TEMPORAL_BOXPLOT_PATH)},
        ]
    )
)

# %% [markdown] Cell 23
# ## Construct Gou Pro experiment registry

# %% Cell 24
GOU_PRO_EXPERIMENTS = {
    "gou_pro_selected": {
        "dataset": gou_pro_dataset,
        "scenario": "gou",
        "feature_set": "selected_feature_TMM_logCPM",
        "output_name": "gou_pro_selected",
    }
}

GOU_PRO_RUN_ORDER = ("gou_pro_selected",)
assert set(GOU_PRO_RUN_ORDER) == set(GOU_PRO_EXPERIMENTS)
assert all("gou_pro" in entry["output_name"] for entry in GOU_PRO_EXPERIMENTS.values())
for experiment_name, entry in GOU_PRO_EXPERIMENTS.items():
    validate_molecular_dataset(entry["dataset"])
    dataset = entry["dataset"]
    print(
        f"{experiment_name}: {dataset.expression.shape[0]} selected features x {dataset.expression.shape[1]} samples; "
        f"classes={dataset.class_labels.value_counts().sort_index().to_dict()}; "
        f"batches={dataset.batch_labels.value_counts().sort_index().to_dict()}; "
        f"output={entry['output_name']}"
    )

# %% [markdown] Cell 25
# ## Run bidirectional prevalence-mode Pro experiments
# 
# Full Gou Pro execution is disabled by default. When enabled, outputs are isolated under `results_pro_gou/`, run both directions, run both prevalence modes, and retain raw and balanced error.

# %% Cell 26
RUN_GOU_PRO = True
GOU_PRO_TEST_MODE = False

GOU_PRO_TIMES = 10
GOU_PRO_SPLITS = (50, 70, 80, 90, 100)
GOU_PRO_NORMALIZATIONS = ("non", "qn", "mn", "vsn")
GOU_PRO_CLASSIFIERS = ("knn", "lasso", "rf", "svm", "xgb")
GOU_PRO_SEED = 2024
GOU_PRO_PREVALENCE_MODES = ("natural", "class_stratified")

GOU_PRO_SMOKE_OUTPUT_ROOT = GOU_PRO_OUTPUT_ROOT / "smoke_test"
GOU_PRO_FULL_OUTPUT_ROOT = GOU_PRO_OUTPUT_ROOT / "full_run"

for output_path in (GOU_PRO_OUTPUT_ROOT, GOU_PRO_QC_DIR, GOU_PRO_SMOKE_OUTPUT_ROOT, GOU_PRO_FULL_OUTPUT_ROOT):
    assert_inside_base(output_path)
if GOU_PRO_OUTPUT_ROOT.name in {"results_r_parity_test", "results", "results_gou_python"}:
    raise ValueError("Gou Pro output root must not collide with original Gou outputs")

# Lightweight Pro smoke test: real engine, both directions, both prevalence modes.
smoke_start = perf_counter()
gou_pro_smoke_manifest = run_ssv_pro_experiment_batch(
    experiments=GOU_PRO_EXPERIMENTS,
    output_root=GOU_PRO_SMOKE_OUTPUT_ROOT,
    experiment_order=GOU_PRO_RUN_ORDER,
    n_repetitions=2,
    split_combination=(50, 100),
    normalization_methods=("non",),
    classifier_methods=("knn",),
    seed=GOU_PRO_SEED,
    prevalence_modes=GOU_PRO_PREVALENCE_MODES,
    data_type_rnaseq=DATA_TYPE_RNASEQ,
    progress=False,
)
assert not gou_pro_smoke_manifest.empty
assert gou_pro_smoke_manifest["status"].eq("complete").all()

smoke_raw_frames = []
for _, row in gou_pro_smoke_manifest.iterrows():
    output_dir = Path(row["output_directory"])
    raw_path = output_dir / "raw_results.csv"
    assert raw_path.exists() and raw_path.stat().st_size > 0
    raw = pd.read_csv(raw_path)
    raw.insert(0, "prevalence_output", output_dir.name)
    smoke_raw_frames.append(raw)

gou_pro_smoke_results = pd.concat(smoke_raw_frames, ignore_index=True)
assert set(gou_pro_smoke_results["direction"]) == {"batch_1_to_batch_2", "batch_2_to_batch_1"}
assert set(gou_pro_smoke_results["prevalence_mode"]) == set(GOU_PRO_PREVALENCE_MODES)
assert set(gou_pro_smoke_results["split_percentage"]) == {50, 100}
assert {"raw_error", "balanced_error"} <= set(gou_pro_smoke_results.columns)
assert np.isfinite(gou_pro_smoke_results[["raw_error", "balanced_error"]].to_numpy()).all()

smoke_elapsed_seconds = perf_counter() - smoke_start
print(f"Gou Pro smoke test completed in {smoke_elapsed_seconds:.1f} seconds.")
display(gou_pro_smoke_manifest)
display(gou_pro_smoke_results.head())

if RUN_GOU_PRO:
    pro_start = perf_counter()
    gou_pro_manifest = run_ssv_pro_experiment_batch(
        experiments=GOU_PRO_EXPERIMENTS,
        output_root=GOU_PRO_FULL_OUTPUT_ROOT,
        experiment_order=GOU_PRO_RUN_ORDER,
        n_repetitions=GOU_PRO_TIMES,
        split_combination=GOU_PRO_SPLITS,
        normalization_methods=GOU_PRO_NORMALIZATIONS,
        classifier_methods=GOU_PRO_CLASSIFIERS,
        seed=GOU_PRO_SEED,
        prevalence_modes=GOU_PRO_PREVALENCE_MODES,
        data_type_rnaseq=DATA_TYPE_RNASEQ,
        progress=True,
    )
    assert gou_pro_manifest["status"].eq("complete").all()
    elapsed_seconds = perf_counter() - pro_start
    print(f"Gou Pro full run elapsed time: {elapsed_seconds / 60:.2f} minutes")
    display(gou_pro_manifest)
else:
    gou_pro_manifest = pd.DataFrame()
    print("Gou Pro full run: NOT RUN")
    print("Full-run switch remains disabled by default: RUN_GOU_PRO = False.")
    print(f"When enabled, full outputs will be written under: {GOU_PRO_FULL_OUTPUT_ROOT}")

# %% [markdown] Cell 27
# ## Compact Pro summaries / robustness curves

# %% Cell 28

smoke_summary_tables = []
for _, row in gou_pro_smoke_manifest.iterrows():
    output_dir = Path(row["output_directory"])
    equal_weight_path = output_dir / "equal_weight_summary.csv"
    direction_gap_path = output_dir / "direction_gap.csv"
    assert equal_weight_path.exists() and equal_weight_path.stat().st_size > 0
    equal_weight = pd.read_csv(equal_weight_path)
    equal_weight.insert(0, "prevalence_output", output_dir.name)
    smoke_summary_tables.append(equal_weight)
    if direction_gap_path.exists():
        display(pd.read_csv(direction_gap_path).head())

gou_pro_smoke_summary = pd.concat(smoke_summary_tables, ignore_index=True)
display(gou_pro_smoke_summary)

if not gou_pro_manifest.empty:
    full_summary_tables = []
    for _, row in gou_pro_manifest.iterrows():
        output_dir = Path(row["output_directory"])
        equal_weight_path = output_dir / "equal_weight_summary.csv"
        if equal_weight_path.exists():
            equal_weight = pd.read_csv(equal_weight_path)
            equal_weight.insert(0, "prevalence_output", output_dir.name)
            full_summary_tables.append(equal_weight)
    if full_summary_tables:
        display(pd.concat(full_summary_tables, ignore_index=True).head())
else:
    print("No full-run Gou Pro modeling summaries because RUN_GOU_PRO is False.")

# %% [markdown] Cell 29
# ## Final integrity checks and artifact audit

# %% Cell 30

audit_rows = []


def audit(name, condition):
    audit_rows.append({"check": name, "status": "PASS" if bool(condition) else "FAIL"})


def nonempty(path):
    path = Path(path)
    return path.exists() and path.stat().st_size > 0


audit("new_pro_notebook_name", PRO_NOTEBOOK_PATH.name == "SSV_Gou_Python_pro.ipynb")
audit("baseline_source_not_overwritten", BASELINE_SOURCE_NOTEBOOK.resolve() != PRO_NOTEBOOK_PATH.resolve())
audit("shared_pro_runtime_loaded", all(name in globals() for name in _RUNTIME_REQUIRED))
audit("gou_specific_output_root", GOU_PRO_OUTPUT_ROOT.name == "results_pro_gou")
audit("output_root_inside_3Gou", BASE_DIR.resolve() in GOU_PRO_OUTPUT_ROOT.resolve().parents)
audit("selected_feature_panel_not_poor_marker_named", "poor" not in "selected_feature_TMM_logCPM".lower())

audit("qc_summary_saved_nonempty", nonempty(GOU_PRO_QC_DIR / "gou_pro_qc_summary.csv"))
audit("original_4x2_contingency_saved_nonempty", nonempty(GOU_PRO_QC_DIR / "gou_pro_original_batch_class_contingency.csv"))
audit("temporal_2x2_contingency_saved_nonempty", nonempty(GOU_PRO_QC_DIR / "gou_pro_temporal_batch_class_contingency.csv"))
audit("batch1_batch2_contingency_saved_nonempty", nonempty(GOU_PRO_QC_DIR / "gou_pro_batch1_batch2_class_contingency.csv"))
audit("sample_provenance_saved_nonempty", nonempty(GOU_PRO_QC_DIR / "gou_pro_sample_provenance.csv"))
audit("original_boxplot_saved_nonempty", nonempty(ORIGINAL_BOXPLOT_PATH))
audit("temporal_boxplot_saved_nonempty", nonempty(TEMPORAL_BOXPLOT_PATH))

all_sample_indices_aligned = (
    selected_expression.columns.equals(class_id.index)
    and selected_expression.columns.equals(original_batch.index)
    and selected_expression.columns.equals(temporal_batch.index)
    and selected_expression.columns.equals(temporal_pro_batch.index)
)
audit("sample_alignment", all_sample_indices_aligned)
audit("no_samples_lost_or_duplicated", len(set(sample_ids)) == len(sample_ids) == selected_expression.shape[1])
audit("temporal_pro_dataset_binary_batch", set(gou_pro_dataset.batch_labels.unique()) == {"batch_1", "batch_2"})
audit("batch_1_nonempty", (gou_pro_dataset.batch_labels == "batch_1").sum() > 0)
audit("batch_2_nonempty", (gou_pro_dataset.batch_labels == "batch_2").sum() > 0)
audit("early_late_mapping_correct", set(original_batch[temporal_batch.eq("early")]) == set(EARLY_ORIGINAL_BATCHES))
audit("nonfinite_expression_absent", np.isfinite(gou_pro_dataset.expression.to_numpy()).all())
audit("prevalence_modes_supported", set(GOU_PRO_PREVALENCE_MODES) == {"natural", "class_stratified"})
audit("bidirectional_pro_engine_available", "run_ssv_pro_bidirectional" in globals())
audit("raw_and_balanced_error_supported", {"raw_error", "balanced_error"} <= set(PRO_METRICS))
audit("full_splits_configured", GOU_PRO_SPLITS == (50, 70, 80, 90, 100))

audit("smoke_manifest_nonempty", not gou_pro_smoke_manifest.empty)
audit("smoke_results_nonempty", not gou_pro_smoke_results.empty)
audit("smoke_both_directions", set(gou_pro_smoke_results["direction"]) == {"batch_1_to_batch_2", "batch_2_to_batch_1"})
audit("smoke_both_prevalence_modes", set(gou_pro_smoke_results["prevalence_mode"]) == set(GOU_PRO_PREVALENCE_MODES))
audit("smoke_splits_50_100", set(gou_pro_smoke_results["split_percentage"]) == {50, 100})
audit("smoke_raw_and_balanced_error_columns", {"raw_error", "balanced_error"} <= set(gou_pro_smoke_results.columns))
audit("smoke_artifacts_written", all(nonempty(Path(row["output_directory"]) / "raw_results.csv") for _, row in gou_pro_smoke_manifest.iterrows()))
audit("full_run_disabled_by_default", RUN_GOU_PRO is False and GOU_PRO_TEST_MODE is False and gou_pro_manifest.empty)

audit_table = pd.DataFrame(audit_rows)
display(audit_table)
if not audit_table["status"].eq("PASS").all():
    display(audit_table.loc[~audit_table["status"].eq("PASS")])
    raise AssertionError("One or more Gou Pro audit checks failed")

print("Gou Pro structural/QC audit: PASS")
print("Gou Pro diagnostic outputs: PASS")
print("Gou Pro smoke test: PASS")
print("Gou Pro full run: NOT RUN")
