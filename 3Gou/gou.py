"""Gou-specific utilities for the batch-aware molecular classifier workflow."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def filter_gou_features(samples_by_features, min_present_prop=0.05):
    frame = samples_by_features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    audit = [{"stage": "numeric + NA to zero", "n_features": frame.shape[1]}]

    frame = frame.loc[:, frame.abs().sum(axis=0) > 0]
    audit.append({"stage": "remove all-zero features", "n_features": frame.shape[1]})

    frame = frame.loc[:, frame.var(axis=0, ddof=1) > 0]
    audit.append({"stage": "remove zero-variance features", "n_features": frame.shape[1]})

    frame = frame.loc[:, (frame > 0).mean(axis=0) >= min_present_prop]
    audit.append({"stage": "detected in >=5% samples", "n_features": frame.shape[1]})

    if not frame.index.equals(samples_by_features.index):
        raise ValueError("Feature filtering changed sample order")
    if frame.empty:
        raise ValueError("Feature filtering removed every feature")
    return frame, pd.DataFrame(audit)

def _positive_series(values, name):
    series = pd.Series(values, dtype=float)
    if not np.isfinite(series.to_numpy()).all() or (series <= 0).any():
        raise ValueError(f"{name} must be finite and positive")
    return series

def _select_tmm_reference(counts_features_by_samples):
    raw = counts_features_by_samples.astype(float)
    lib_sizes = _positive_series(raw.sum(axis=0), "library sizes")
    uq = raw.apply(lambda col: np.quantile(col[col > 0], 0.75), axis=0) / lib_sizes
    return int(np.argmin(np.abs(uq - uq.mean())))

def tmm_normalization_factors(counts_features_by_samples, ref_col=None, logratio_trim=0.30, sum_trim=0.05):
    raw = counts_features_by_samples.astype(float)
    lib_sizes = _positive_series(raw.sum(axis=0), "library sizes")
    ref_idx = _select_tmm_reference(raw) if ref_col is None else int(ref_col)
    ref = raw.iloc[:, ref_idx]
    ref_lib = lib_sizes.iloc[ref_idx]
    factors = []

    for col_idx in range(raw.shape[1]):
        obs = raw.iloc[:, col_idx]
        obs_lib = lib_sizes.iloc[col_idx]
        keep = (obs > 0) & (ref > 0)
        if col_idx == ref_idx or keep.sum() < 2:
            factors.append(1.0)
            continue
        obs_p = obs[keep] / obs_lib
        ref_p = ref[keep] / ref_lib
        m_values = np.log2(obs_p / ref_p)
        a_values = 0.5 * np.log2(obs_p * ref_p)
        weights = 1 / ((1 / obs[keep]) - (1 / obs_lib) + (1 / ref[keep]) - (1 / ref_lib))
        finite = np.isfinite(m_values) & np.isfinite(a_values) & np.isfinite(weights) & (weights > 0)
        m_values = m_values[finite]
        a_values = a_values[finite]
        weights = weights[finite]
        n = len(m_values)
        if n < 2:
            factors.append(1.0)
            continue
        m_rank = pd.Series(m_values).rank(method="first").to_numpy()
        a_rank = pd.Series(a_values).rank(method="first").to_numpy()
        lo_m, hi_m = np.floor(n * logratio_trim) + 1, n - np.floor(n * logratio_trim)
        lo_a, hi_a = np.floor(n * sum_trim) + 1, n - np.floor(n * sum_trim)
        trim_keep = (m_rank >= lo_m) & (m_rank <= hi_m) & (a_rank >= lo_a) & (a_rank <= hi_a)
        if not trim_keep.any():
            factors.append(1.0)
            continue
        factors.append(2 ** np.average(m_values[trim_keep], weights=weights[trim_keep]))

    factors = pd.Series(factors, index=raw.columns, dtype=float, name="normalization_factor")
    factors = factors / np.exp(np.mean(np.log(factors)))
    return _positive_series(factors, "TMM normalization factors")

def edgeR_style_logcpm_tmm(samples_by_features, prior_count=1.0):
    counts = samples_by_features.T.astype(float)
    library_size = _positive_series(counts.sum(axis=0), "raw library sizes")
    norm_factor = tmm_normalization_factors(counts)
    effective_library_size = library_size * norm_factor
    scaled_prior = prior_count * effective_library_size / effective_library_size.mean()
    augmented_library = effective_library_size + 2 * scaled_prior
    cpm = counts.add(scaled_prior, axis=1).divide(augmented_library, axis=1) * 1_000_000
    logcpm = np.log2(cpm).T
    logcpm.index = samples_by_features.index.copy()
    logcpm.columns = samples_by_features.columns.copy()
    audit = pd.DataFrame({"raw_library_size": library_size, "normalization_factor": norm_factor, "effective_library_size": effective_library_size})
    audit.index.name = "sample_id"
    return logcpm, audit

def plot_gou_umap(
    coordinates,
    group_column,
    title,
    output_path,
):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    groups = coordinates[group_column].astype(str)
    categories = list(pd.unique(groups))

    for category in categories:
        mask = groups.eq(category)

        ax.scatter(
            coordinates.loc[mask, "UMAP1"],
            coordinates.loc[mask, "UMAP2"],
            s=36,
            alpha=0.78,
            label=category,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    ax.legend(
        title=group_column.replace("_", " ").title(),
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    return fig, ax

def plot_selected_feature_samplewise_boxplot(expression, class_labels, batch_labels, *, title, output_path, max_samples_per_stratum=40):
    fig, ax, ordered_samples = plot_poor_marker_samplewise_boxplot(
        expression,
        class_labels,
        batch_labels,
        title=title,
        max_samples_per_stratum=max_samples_per_stratum,
        seed=GOU_STRUCTURE_SEED,
    )
    ax.set_ylabel("Selected-feature log2 CPM")
    ax.set_title(title)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig, ax, ordered_samples
