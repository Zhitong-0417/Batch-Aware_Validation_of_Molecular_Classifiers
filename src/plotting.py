"""SSV workflow utilities."""



from .common import *
from .evaluation import *



def plot_error_by_split(summary, classifier=None, normalization=None, split_levels=None):
    """Plot mean outer-test error by equally spaced split-percentage levels."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for plotting but is not installed.")
    if classifier is None and normalization is None:
        raise ValueError("provide classifier or normalization to fix one comparison dimension")

    plot_data = summary.copy()
    if classifier is not None:
        plot_data = plot_data[plot_data["classifier_method"] == classifier]
    if normalization is not None:
        plot_data = plot_data[plot_data["normalization_method"] == normalization]
    if plot_data.empty:
        raise ValueError("no rows remain after applying plot filters")

    split_levels = _resolve_split_levels(plot_data["split_percentage"], split_levels)
    split_to_x = {split: position for position, split in enumerate(split_levels)}
    plot_data["_x"] = plot_data["split_percentage"].map(split_to_x)

    varying = "normalization_method" if classifier is not None else "classifier_method"
    if classifier is not None and normalization is not None:
        varying = "classifier_method"

    fig, ax = plt.subplots(figsize=(5, 3))
    for label, group in plot_data.groupby(varying, sort=False):
        group = group.sort_values("_x")
        yerr = group["se_outer_error"].where(np.isfinite(group["se_outer_error"]), np.nan)
        ax.errorbar(
            group["_x"],
            group["mean_outer_error"],
            yerr=yerr,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=label,
        )
    ax.set_xticks(np.arange(len(split_levels)))
    ax.set_xticklabels([str(split) for split in split_levels])
    ax.set_xlabel("Split percentage")
    ax.set_ylabel("Mean outer test error")
    if plot_data[varying].nunique() > 1:
        ax.legend(title=varying.replace("_", " "))
    fig.tight_layout()
    return fig, ax


def _resolve_classifier_colors(classifier_methods, classifier_colors=None):
    palette = dict(DEFAULT_CLASSIFIER_COLORS)
    if classifier_colors is not None:
        palette.update(classifier_colors)
    fallback = plt.get_cmap("tab10").colors if MATPLOTLIB_AVAILABLE else []
    colors = {}
    for idx, classifier in enumerate(classifier_methods):
        if classifier in palette:
            colors[classifier] = palette[classifier]
        elif fallback:
            colors[classifier] = fallback[idx % len(fallback)]
        else:
            colors[classifier] = "#333333"
    return colors


def _resolve_normalization_linestyles(normalization_methods, normalization_linestyles=None):
    styles = dict(DEFAULT_NORMALIZATION_LINESTYLES)
    if normalization_linestyles is not None:
        styles.update(normalization_linestyles)
    fallback = ["-", "--", ":", "-."]
    return {
        normalization: styles.get(normalization, fallback[idx % len(fallback)])
        for idx, normalization in enumerate(normalization_methods)
    }


def _plot_semantic_error_curve(
    summary,
    *,
    classifier_methods=None,
    normalization_methods=None,
    split_levels=None,
    title,
    ylabel,
    classifier_colors=None,
    normalization_linestyles=None,
    linewidth=DEFAULT_ERROR_CURVE_LINEWIDTH,
    marker="o",
    figsize=(7, 4.5),
):
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for plotting but is not installed.")
    required = {"split_percentage", "normalization_method", "classifier_method", "mean_outer_error", "se_outer_error"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"summary is missing required columns: {sorted(missing)}")

    plot_data = summary[np.isfinite(pd.to_numeric(summary["mean_outer_error"], errors="coerce"))].copy()
    if plot_data.empty:
        raise ValueError("no finite mean_outer_error rows are available for plotting")

    classifier_methods = _ordered_values(plot_data["classifier_method"], classifier_methods)
    normalization_methods = _ordered_values(plot_data["normalization_method"], normalization_methods)
    split_levels = _resolve_split_levels(plot_data["split_percentage"], split_levels)
    split_to_x = {split: position for position, split in enumerate(split_levels)}
    plot_data["_x"] = plot_data["split_percentage"].map(split_to_x)
    colors = _resolve_classifier_colors(classifier_methods, classifier_colors)
    linestyles = _resolve_normalization_linestyles(normalization_methods, normalization_linestyles)

    fig, ax = plt.subplots(figsize=figsize)
    plotted = 0
    for classifier in classifier_methods:
        for normalization in normalization_methods:
            subset = plot_data[
                plot_data["classifier_method"].eq(classifier)
                & plot_data["normalization_method"].eq(normalization)
            ].copy()
            if subset.empty:
                continue
            subset = subset.sort_values("_x")
            yerr = pd.to_numeric(subset["se_outer_error"], errors="coerce").to_numpy(dtype=float)
            ax.errorbar(
                subset["_x"],
                subset["mean_outer_error"],
                yerr=yerr,
                color=colors[classifier],
                linestyle=linestyles[normalization],
                marker=marker,
                linewidth=linewidth,
                markersize=3,
                capsize=2,
                label="_nolegend_",
            )
            plotted += 1
    if plotted == 0:
        raise ValueError("no curves were plotted")

    from matplotlib.lines import Line2D

    classifier_handles = [
        Line2D([0], [0], color=colors[classifier], linestyle="-", linewidth=linewidth, label=classifier)
        for classifier in classifier_methods
    ]
    normalization_handles = [
        Line2D([0], [0], color="#444444", linestyle=linestyles[normalization], linewidth=linewidth, label=normalization)
        for normalization in normalization_methods
    ]

    classifier_legend = ax.legend(
        handles=classifier_handles,
        title="Classifier",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
    ax.add_artist(classifier_legend)
    ax.legend(
        handles=normalization_handles,
        title="Normalization",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0,
    )
    ax.set_xticks(np.arange(len(split_levels)))
    ax.set_xticklabels([str(split) for split in split_levels])
    ax.set_title(title)
    ax.set_xlabel("Split percentage")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    return fig, ax


def plot_direction_error_curve(
    direction_summary,
    *,
    direction,
    direction_column="direction",
    classifier_methods=None,
    normalization_methods=None,
    split_levels=None,
    title=None,
    classifier_colors=None,
    normalization_linestyles=None,
    linewidth=DEFAULT_ERROR_CURVE_LINEWIDTH,
):
    """Plot one transfer direction as a self-contained error curve."""
    if direction_column not in direction_summary.columns:
        raise ValueError(f"direction_summary is missing '{direction_column}'")
    plot_data = direction_summary[direction_summary[direction_column].eq(direction)].copy()
    if plot_data.empty:
        raise ValueError(f"no rows found for direction: {direction}")
    return _plot_semantic_error_curve(
        plot_data,
        classifier_methods=classifier_methods,
        normalization_methods=normalization_methods,
        split_levels=split_levels,
        title=title or f"{direction} error curve",
        ylabel="Mean outer test error",
        classifier_colors=classifier_colors,
        normalization_linestyles=normalization_linestyles,
        linewidth=linewidth,
    )


def plot_bidirectional_error_curve(
    bidirectional_summary,
    *,
    classifier_methods=None,
    normalization_methods=None,
    split_levels=None,
    title="Final bidirectional error curve",
    classifier_colors=None,
    normalization_linestyles=None,
    linewidth=DEFAULT_ERROR_CURVE_LINEWIDTH,
):
    """Plot equal-weight paired bidirectional summary errors."""
    return _plot_semantic_error_curve(
        bidirectional_summary,
        classifier_methods=classifier_methods,
        normalization_methods=normalization_methods,
        split_levels=split_levels,
        title=title,
        ylabel="Mean bidirectional outer test error",
        classifier_colors=classifier_colors,
        normalization_linestyles=normalization_linestyles,
        linewidth=linewidth,
    )


def plot_ssv_error_curve(
    summary,
    *,
    classifier_methods=None,
    normalization_methods=None,
    split_levels=None,
    title="R-style SSV reproduction",
    classifier_colors=None,
    normalization_linestyles=None,
    linewidth=DEFAULT_ERROR_CURVE_LINEWIDTH,
):
    """Plot R-style SSV mean outer-test error by split percentage."""
    return _plot_semantic_error_curve(
        summary,
        classifier_methods=classifier_methods,
        normalization_methods=normalization_methods,
        split_levels=split_levels,
        title=title,
        ylabel="Mean outer test error",
        classifier_colors=classifier_colors,
        normalization_linestyles=normalization_linestyles,
        linewidth=linewidth,
    )


def _plot_pro_equal_weight(equal_weight_summary, *, metric, title, output_path):
    metric_col, se_col = f"mean_{metric}", f"se_{metric}"
    summary = equal_weight_summary.rename(columns={metric_col: "mean_outer_error", se_col: "se_outer_error"})
    fig, ax = plot_ssv_error_curve(summary, classifier_methods=ACTIVE_CLASSIFIER_METHODS, normalization_methods=("non", "qn", "mn", "vsn"), split_levels=(50, 70, 80, 90, 100), title=title)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_poor_marker_samplewise_boxplot(
    poor_expression,
    class_labels,
    batch_labels,
    *,
    title=None,
    max_samples_per_stratum=None,
    seed=12345,
):
    """Draw one poor-marker expression boxplot per sample.

    Each sample contributes exactly one box from its vector of poor-marker
    values. Values are used on the already transformed analysis scale; this
    helper intentionally performs no log transform.
    """
    if not isinstance(poor_expression, pd.DataFrame):
        raise TypeError("poor_expression must be a pandas DataFrame")
    class_labels = pd.Series(class_labels).copy()
    batch_labels = pd.Series(batch_labels).copy()
    if poor_expression.empty:
        raise ValueError("poor_expression is empty")
    if not poor_expression.columns.equals(class_labels.index):
        raise ValueError("poor_expression columns must exactly match class_labels.index")
    if not poor_expression.columns.equals(batch_labels.index):
        raise ValueError("poor_expression columns must exactly match batch_labels.index")

    metadata = pd.DataFrame(
        {
            "sample_id": poor_expression.columns.astype(str),
            "batch": batch_labels.astype(str).to_numpy(),
            "class": class_labels.astype(str).to_numpy(),
            "original_order": np.arange(poor_expression.shape[1]),
        },
        index=poor_expression.columns,
    )

    if max_samples_per_stratum is not None:
        max_samples_per_stratum = int(max_samples_per_stratum)
        if max_samples_per_stratum < 1:
            raise ValueError("max_samples_per_stratum must be >= 1")
        rng = np.random.default_rng(seed)
        retained = []
        for _, group in metadata.groupby(["batch", "class"], sort=True):
            ordered_group = group.sort_values("original_order")
            if ordered_group.shape[0] <= max_samples_per_stratum:
                retained.extend(ordered_group.index.tolist())
            else:
                chosen = set(rng.choice(ordered_group.index.to_numpy(), size=max_samples_per_stratum, replace=False))
                retained.extend([sample_id for sample_id in ordered_group.index if sample_id in chosen])
        metadata = metadata.loc[retained]

    metadata = metadata.sort_values(["batch", "class", "original_order"], kind="mergesort")
    sample_ids = metadata.index.tolist()
    vectors = [poor_expression.loc[:, sample_id].dropna().to_numpy(dtype=float) for sample_id in sample_ids]
    if len(vectors) != metadata.shape[0]:
        raise AssertionError("box count does not match retained sample count")
    if any(vector.size == 0 for vector in vectors):
        raise ValueError("at least one retained sample has no finite poor-marker values")

    classes = sorted(metadata["class"].unique(), key=str)
    palette = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color",
        ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"],
    )
    class_colors = {class_label: palette[i % len(palette)] for i, class_label in enumerate(classes)}
    positions = np.arange(1, len(vectors) + 1)
    fig_width = min(24.0, max(9.0, 0.055 * len(vectors) + 6.0))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    bp = ax.boxplot(
        vectors,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": "black", "linewidth": 0.6},
        boxprops={"linewidth": 0.35},
        whiskerprops={"linewidth": 0.35},
        capprops={"linewidth": 0.35},
    )
    for box, class_label in zip(bp["boxes"], metadata["class"]):
        box.set_facecolor(class_colors[class_label])
        box.set_alpha(0.68)
        box.set_edgecolor("#30343b")

    block_rows = []
    start = 1
    for (batch, class_label), group in metadata.groupby(["batch", "class"], sort=False):
        end = start + group.shape[0] - 1
        block_rows.append({"batch": batch, "class": class_label, "start": start, "end": end})
        start = end + 1
    blocks = pd.DataFrame(block_rows)

    # Thin class separators within batch; stronger separators between batches.
    for _, row in blocks.iloc[:-1].iterrows():
        next_row = blocks.iloc[int(row.name) + 1]
        is_batch_boundary = row["batch"] != next_row["batch"]
        ax.axvline(
            row["end"] + 0.5,
            color="#111827" if is_batch_boundary else "#9ca3af",
            linewidth=1.2 if is_batch_boundary else 0.55,
            alpha=0.85 if is_batch_boundary else 0.7,
        )

    batch_blocks = metadata.groupby("batch", sort=False).agg(start=("original_order", "size"))
    current = 1
    y0, y1 = ax.get_ylim()
    y_text = y0 - 0.08 * (y1 - y0)
    for batch, group in metadata.groupby("batch", sort=False):
        batch_start = current
        batch_end = current + group.shape[0] - 1
        center = (batch_start + batch_end) / 2
        ax.text(center, y_text, str(batch), ha="center", va="top", fontsize=9, clip_on=False)
        current = batch_end + 1

    if len(sample_ids) <= 40:
        ax.set_xticks(positions)
        ax.set_xticklabels(metadata["sample_id"], rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    ax.set_xlim(0.5, len(vectors) + 0.5)
    ax.set_xlabel("Samples ordered by batch, then disease group")
    ax.set_ylabel("Poor-marker expression")
    ax.set_title(title or "Poor-marker expression by sample, batch, and class")
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=6, alpha=0.68, label=str(class_label))
        for class_label, color in class_colors.items()
    ]
    ax.legend(handles=handles, title="Disease group", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", color="#d0d5dd", linewidth=0.5, alpha=0.65)
    fig.subplots_adjust(bottom=0.18, right=0.82)
    return fig, ax, metadata.reset_index(drop=True)


def plot_poor_marker_boxplot(
    poor_expression,
    class_labels,
    batch_labels,
    *,
    title=None,
    max_samples=None,
    seed=12345,
):
    """Backward-compatible wrapper for the sample-wise poor-marker boxplot."""
    return plot_poor_marker_samplewise_boxplot(
        poor_expression,
        class_labels,
        batch_labels,
        title=title,
        max_samples_per_stratum=max_samples,
        seed=seed,
    )[:2]
