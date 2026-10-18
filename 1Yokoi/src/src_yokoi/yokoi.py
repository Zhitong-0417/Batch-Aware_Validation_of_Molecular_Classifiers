"""Yokoi-specific helpers."""



from .common import *
from .data import *



def make_unique_like_r(values):
    """Return names with R make.unique-style numeric suffixes for duplicates."""
    seen = {}
    unique_values = []
    for value in map(str, values):
        count = seen.get(value, 0)
        unique_values.append(value if count == 0 else f"{value}.{count}")
        seen[value] = count + 1
    return unique_values


def load_yokoi_combined_csv(path):
    """Load the Yokoi combined metadata/expression CSV using Block 1 semantics."""
    combined = pd.read_csv(path, dtype=str)
    if combined.shape[0] <= 8:
        raise ValueError("combined data must contain 8 metadata rows plus expression rows")
    if combined.shape[1] <= 1:
        raise ValueError("combined data must contain one ID column and at least one sample column")

    sample_ids = pd.Index(combined.columns[1:].astype(str), name="sample_id")
    if not sample_ids.is_unique:
        raise ValueError("sample IDs in the combined file are not unique")

    metadata_rows = combined.iloc[:8, :]
    metadata_field_names = make_unique_like_r(metadata_rows.iloc[:, 0])
    metadata = metadata_rows.iloc[:, 1:].T.copy(deep=True)
    metadata.index = sample_ids
    metadata.index.name = "sample_id"
    metadata.columns = metadata_field_names

    class_labels_raw = metadata_rows.iloc[4, 1:].copy(deep=True)
    class_labels_raw.index = sample_ids
    class_labels_raw.name = "class_label_raw"

    expression_rows = combined.iloc[8:, :]
    feature_ids = pd.Index(make_unique_like_r(expression_rows.iloc[:, 0]), name="feature_id")
    if feature_ids.empty:
        raise ValueError("expression matrix has no feature rows")
    if not feature_ids.is_unique:
        raise ValueError("feature IDs are not unique after make_unique-style repair")

    expression_values = expression_rows.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
    expression_all = pd.DataFrame(
        expression_values.to_numpy(copy=True),
        index=feature_ids,
        columns=sample_ids,
    )

    return expression_all, metadata, class_labels_raw


def load_yokoi_meta_crosscheck(path, expected_sample_ids):
    """Read Yokoi_meta.csv only as a sample-order cross-check."""
    meta_crosscheck = pd.read_csv(path, dtype=str)
    if "sample_id" not in meta_crosscheck.columns:
        raise ValueError("Yokoi_meta.csv must contain a sample_id column for cross-checking")

    meta_sample_ids = pd.Index(meta_crosscheck["sample_id"].astype(str), name="sample_id")
    if not meta_sample_ids.is_unique:
        raise ValueError("sample_id values in Yokoi_meta.csv are not unique")
    if not expected_sample_ids.equals(meta_sample_ids):
        raise ValueError("Yokoi_meta.csv sample_id values do not match combined-file sample order")

    return meta_crosscheck.set_index("sample_id", drop=False)


def validate_yokoi_block1(expression_all, metadata, class_labels_raw):
    """Validate Block 1 sample alignment without altering the source data."""
    if expression_all.empty:
        raise ValueError("expression matrix is empty")
    if metadata.empty:
        raise ValueError("metadata table is empty")
    if class_labels_raw.empty:
        raise ValueError("class label vector is empty")
    if not expression_all.columns.equals(metadata.index):
        raise ValueError("expression columns and metadata sample IDs are not identically aligned")
    if not expression_all.columns.equals(class_labels_raw.index):
        raise ValueError("expression columns and class-label sample IDs are not identically aligned")
    if not expression_all.columns.is_unique:
        raise ValueError("expression sample IDs are not unique")
    if not expression_all.index.is_unique:
        raise ValueError("expression feature IDs are not unique")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in expression_all.dtypes):
        raise TypeError("all expression columns must be numeric")

    return {
        "n_features": expression_all.shape[0],
        "n_samples": expression_all.shape[1],
        "n_metadata_fields": metadata.shape[1],
        "n_classes": class_labels_raw.nunique(dropna=False),
    }


def load_frozen_batches(scenario_name, expected_sample_ids):
    path = FROZEN_BATCH_DIR / f"{scenario_name}_inferred_batches.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen batch file: {path}. Create it once from validated R-reproduction batch labels. "
            "The Pro notebook intentionally does not rerun UMAP/KMeans."
        )
    frame = pd.read_csv(path)
    required = {"sample_id", "inferred_batch"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    labels = pd.Series(frame["inferred_batch"].to_numpy(), index=frame["sample_id"].astype(str), name="batch").loc[list(expected_sample_ids)]
    if not labels.index.equals(pd.Index(expected_sample_ids)):
        raise ValueError(f"{scenario_name}: frozen batch labels do not align with samples")
    return labels


def plot_feature_mean_thresholds(feature_means, poor_lower, poor_upper, positive_cutoff, title, sample_label):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(feature_means, bins=60, color="#7a8699", edgecolor="white", linewidth=0.5)
    ax.axvline(poor_lower, color="#d97706", linestyle="--", linewidth=1.5, label="poor lower")
    ax.axvline(poor_upper, color="#f59e0b", linestyle="--", linewidth=1.5, label="poor upper")
    ax.axvline(positive_cutoff, color="#0f766e", linestyle="-", linewidth=1.8, label="positive cutoff")
    ax.set_xlabel(f"Mean expression across {sample_label}")
    ax.set_ylabel("Number of features")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_selected_feature_means(feature_means, poor_mask, positive_mask, title="Scenario 1 selected feature positions"):
    ordered_means = feature_means.sort_values()
    ordered_poor = poor_mask.loc[ordered_means.index]
    ordered_positive = positive_mask.loc[ordered_means.index]
    ranks = pd.Series(range(1, len(ordered_means) + 1), index=ordered_means.index)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(ranks[~ordered_poor & ~ordered_positive], ordered_means[~ordered_poor & ~ordered_positive], s=8, color="#c6cbd3", label="other")
    ax.scatter(ranks[ordered_poor], ordered_means[ordered_poor], s=14, color="#d97706", label="poor")
    ax.scatter(ranks[ordered_positive], ordered_means[ordered_positive], s=14, color="#0f766e", label="positive")
    ax.set_xlabel("Feature rank by mean expression")
    ax.set_ylabel("Mean expression")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax
