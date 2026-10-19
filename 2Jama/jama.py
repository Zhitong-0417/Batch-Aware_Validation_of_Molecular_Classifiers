"""JAMA-specific utilities for the batch-aware molecular classifier workflow.

Only dataset-specific logic lives here. Generic SSV splitting, normalization,
classification, evaluation, plotting, and artifact utilities remain in the
repository-level `src/` package.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from umap import UMAP


BASE_DIR = Path(__file__).resolve().parent

EXPRESSION_PATH = BASE_DIR / "expression_matrix.csv"
EXPRESSED_MIRNAS_PATH = BASE_DIR / "expressed_mirnas.txt"

UMAP_RANDOM_STATE = 1
KMEANS_RANDOM_STATE = 12345
KMEANS_N_INIT = 50
N_BATCHES = 3
EARLY_BATCH_ID = 2

def r_make_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z_.]", ".", text)
    if not re.match(r"^[A-Za-z.]", text) or re.match(r"^\.[0-9]", text):
        text = f"X{text}"
    reserved = {"if", "else", "repeat", "while", "function", "for", "in", "next", "break"}
    return f"{text}." if text in reserved else text

def sample_group(sample: str) -> str | None:
    if sample.startswith("Healthy.control."):
        return "HC"
    if sample.startswith("NTLD."):
        return "NTLD"
    if sample.startswith("LCa."):
        return "LCa"
    if sample.startswith("OD."):
        return "OD"
    return None

def sample_index(sample: str) -> int:
    match = re.search(r"\.(\d+)$", sample)
    if not match:
        raise ValueError(f"Cannot parse trailing sample index from {sample!r}")
    return int(match.group(1))

def load_expression_table(apply_expressed_filter: bool = False):
    data_all = pd.read_csv(EXPRESSION_PATH, sep="\t")
    if data_all.columns[0] != "miRNA":
        raise ValueError("expression_matrix.csv must have miRNA as the first column")
    data_all = data_all.rename(columns={name: r_make_name(name) for name in data_all.columns})
    if not data_all["miRNA"].is_unique:
        raise ValueError("miRNA feature names must be unique")
    if apply_expressed_filter:
        expressed = pd.read_csv(EXPRESSED_MIRNAS_PATH, header=None, sep=r"\s+", engine="python")[0].astype(str)
        data_all = data_all.loc[data_all["miRNA"].astype(str).isin(set(expressed))].copy()
        if data_all.empty:
            raise ValueError("expressed_mirnas.txt filter removed all features")
    return data_all

def numeric_expression_from_table(data_all):
    expr_raw = data_all.drop(columns=["miRNA"]).apply(pd.to_numeric, errors="raise")
    keep_samples = expr_raw.isna().sum(axis=0).eq(0)
    expr_clean = expr_raw.loc[:, keep_samples].copy()
    if expr_clean.empty:
        raise ValueError("No expression samples remain after NA-column removal")
    if not np.isfinite(expr_clean.to_numpy(dtype=float)).all():
        raise ValueError("Expression matrix contains non-finite values before log transform")
    return expr_clean

def make_expression_frame(values, feature_ids, sample_ids):
    expression = pd.DataFrame(values, index=pd.Index(feature_ids, name="miRNA"), columns=pd.Index(sample_ids, name="sample"))
    if not expression.index.is_unique:
        raise ValueError("Expression feature IDs are not unique")
    if not expression.columns.is_unique:
        raise ValueError("Expression sample IDs are not unique")
    return expression

def build_sample_info(sample_ids, clinical_meta=None):
    sample_info = pd.DataFrame({"Sample": list(sample_ids)})
    sample_info["ID"] = sample_info["Sample"].map(sample_group)
    sample_info["Sample_index"] = sample_info["Sample"].map(sample_index)
    if sample_info["ID"].isna().any():
        missing = sample_info.loc[sample_info["ID"].isna(), "Sample"].head().tolist()
        raise ValueError(f"Unexpected sample name pattern: {missing}")
    if clinical_meta is not None:
        group_col = clinical_meta.columns[6]
        stage_col = clinical_meta.columns[7]
        clinical = clinical_meta.copy()
        clinical["Group_clean"] = clinical[group_col].astype(str)
        clinical["Stage"] = pd.to_numeric(clinical[stage_col], errors="coerce")
        clinical["Group_index"] = clinical.groupby("Group_clean").cumcount() + 1
        sample_info = sample_info.merge(
            clinical,
            left_on=["ID", "Sample_index"],
            right_on=["Group_clean", "Group_index"],
            how="left",
            validate="one_to_one",
        )
        lca_missing_stage = sample_info.loc[sample_info["ID"].eq("LCa"), "Stage"].isna().sum()
        if lca_missing_stage == sample_info["ID"].eq("LCa").sum():
            raise ValueError("Clinical stage mapping failed for all LCa samples")
    return sample_info

def remap_three_kmeans_batches(raw_labels, kmeans_centers):
    centers = pd.DataFrame(kmeans_centers, columns=["UMAP1", "UMAP2"])
    centers["cluster"] = range(N_BATCHES)
    global_center = centers[["UMAP1", "UMAP2"]].mean(axis=0)
    centers["dist_to_center"] = (((centers[["UMAP1", "UMAP2"]] - global_center) ** 2).sum(axis=1)) ** 0.5
    middle_cluster = int(centers.loc[centers["dist_to_center"].idxmin(), "cluster"])
    outer_clusters = centers.loc[centers["cluster"].ne(middle_cluster)].sort_values("UMAP1")["cluster"].tolist()
    cluster_map = {outer_clusters[0]: 1, middle_cluster: 2, outer_clusters[1]: 3}
    batch_labels = [cluster_map[int(label)] for label in raw_labels]
    if set(batch_labels) != {1, 2, 3}:
        raise ValueError("KMeans batch remapping did not produce labels 1, 2, and 3")
    return batch_labels

def infer_jama_batches(expression, poor_data, sample_info):
    umap_input = poor_data.T.copy()
    if not umap_input.index.equals(expression.columns):
        raise ValueError("UMAP input sample IDs do not match expression sample IDs")
    if not np.isfinite(umap_input.to_numpy()).all():
        raise ValueError("UMAP input contains non-finite values")
    embedding = UMAP(n_components=4, random_state=UMAP_RANDOM_STATE).fit_transform(umap_input)
    umap_frame = pd.DataFrame(embedding, index=umap_input.index, columns=[f"UMAP{i}" for i in range(1, 5)])
    kmeans_input = umap_frame[["UMAP1", "UMAP2"]]
    kmeans = KMeans(n_clusters=N_BATCHES, n_init=KMEANS_N_INIT, random_state=KMEANS_RANDOM_STATE)
    raw_labels = kmeans.fit_predict(kmeans_input)
    batch_labels = pd.Series(
        remap_three_kmeans_batches(raw_labels, kmeans.cluster_centers_),
        index=umap_frame.index,
        name="batch",
    )
    layout = umap_frame.reset_index(names="Sample").merge(
        sample_info[["Sample", "ID"] + (["Stage"] if "Stage" in sample_info.columns else [])],
        on="Sample",
        how="left",
        validate="one_to_one",
    )
    layout["batch"] = batch_labels.loc[layout["Sample"]].to_numpy(dtype=int)
    if layout["ID"].isna().any():
        raise ValueError("UMAP layout lost biological group labels")
    return layout, batch_labels

def make_jama_meta_batches(inferred_batch_labels):
    """Map original JAMA inferred batches to the two Pro meta-batches.

    Source-of-truth JAMA/R-style temporal semantics:
    inferred batch 2 is early/training-side and maps to batch_1;
    inferred batches 1 and 3 are late/test-side and map to batch_2.
    """
    if not isinstance(inferred_batch_labels, pd.Series):
        raise TypeError("inferred_batch_labels must be a pandas Series")
    labels = inferred_batch_labels.copy()
    if labels.isna().any():
        raise ValueError("inferred batch labels must not contain missing values")
    numeric_labels = pd.to_numeric(labels, errors="raise").astype(int)
    observed = set(numeric_labels.unique())
    if not observed <= {1, 2, 3}:
        raise ValueError(f"Unexpected JAMA inferred batch labels: {sorted(observed)}")
    if observed != {1, 2, 3}:
        raise ValueError(f"Each JAMA scenario must retain inferred batches 1, 2, and 3 before Pro mapping; observed {sorted(observed)}")
    meta = numeric_labels.map({2: "batch_1", 1: "batch_2", 3: "batch_2"}).rename("batch")
    meta.index = inferred_batch_labels.index
    if set(meta.unique()) != {"batch_1", "batch_2"}:
        raise ValueError("JAMA meta-batch mapping must produce exactly batch_1 and batch_2")
    if not meta.index.equals(inferred_batch_labels.index):
        raise ValueError("JAMA meta-batch mapping changed sample order")
    return meta

def validate_jama_meta_batch_mapping(inferred_batch_labels, meta_batch_labels, class_labels, expression, scenario):
    if not expression.columns.equals(inferred_batch_labels.index):
        raise ValueError(f"{scenario}: expression and inferred batch labels are not aligned")
    if not expression.columns.equals(meta_batch_labels.index):
        raise ValueError(f"{scenario}: expression and meta-batch labels are not aligned")
    if not expression.columns.equals(class_labels.index):
        raise ValueError(f"{scenario}: expression and class labels are not aligned")
    if len(meta_batch_labels) != len(inferred_batch_labels) or meta_batch_labels.index.has_duplicates:
        raise ValueError(f"{scenario}: meta-batch mapping lost or duplicated samples")
    if set(meta_batch_labels.unique()) != {"batch_1", "batch_2"}:
        raise ValueError(f"{scenario}: Pro dataset must contain exactly batch_1 and batch_2")
    if not meta_batch_labels.loc[inferred_batch_labels.astype(int).eq(EARLY_BATCH_ID)].eq("batch_1").all():
        raise ValueError(f"{scenario}: inferred batch {EARLY_BATCH_ID} must map to batch_1")
    if not meta_batch_labels.loc[inferred_batch_labels.astype(int).isin([1, 3])].eq("batch_2").all():
        raise ValueError(f"{scenario}: inferred batches 1 and 3 must map to batch_2")
    contingency = pd.crosstab(meta_batch_labels.rename("meta_batch"), class_labels.rename("class"))
    if (contingency == 0).any().any():
        print(f"WARNING: {scenario} has at least one empty meta-batch x class cell; interpret transfer estimates with identifiability caution.")
    return contingency

def plot_umap_by_series(
    umap_frame,
    color_series,
    title,
    legend_title,
    save_path,
):
    """
    Yokoi-style UMAP plot.

    The plot is:
    1. displayed inline;
    2. saved to save_path.

    Existing UMAP1 / UMAP2 coordinates are reused.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    color_series = pd.Series(
        color_series,
        index=umap_frame.index,
    )

    plot_df = pd.DataFrame(
        {
            "UMAP1": umap_frame["UMAP1"].to_numpy(),
            "UMAP2": umap_frame["UMAP2"].to_numpy(),
            "group": color_series.astype(str).to_numpy(),
        },
        index=umap_frame.index,
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    for group in pd.unique(plot_df["group"]):
        mask = plot_df["group"].eq(group)

        ax.scatter(
            plot_df.loc[mask, "UMAP1"],
            plot_df.loc[mask, "UMAP2"],
            s=10,
            alpha=0.72,
            label=str(group),
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)

    ax.legend(
        title=legend_title,
        frameon=False,
        markerscale=1.6,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()

    # Save before plt.show()
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Saved: {save_path}")

