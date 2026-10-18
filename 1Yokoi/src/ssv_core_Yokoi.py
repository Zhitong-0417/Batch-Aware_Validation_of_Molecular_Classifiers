"""Core utilities for the Batch-Aware Validation of Molecular Classifiers project.

This module consolidates reusable code from:
- SSV_Python_Reproduction_pro.ipynb
- SSV_Yokoi_Python_pro.ipynb

The notebooks can remain as analysis entry points, while reusable data handling,
splitting, normalization, classification, QC, and Yokoi helper functions live here.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False


# Shared classifier / plotting defaults
ACTIVE_CLASSIFIER_METHODS = ("knn", "lasso", "rf", "svm", "xgb")
DEFAULT_CLASSIFIER_COLORS = {
    "knn": "#1f77b4",
    "lasso": "#d62728",
    "rf": "#2ca02c",
    "svm": "#8c564b",
    "xgb": "#ff7f0e",
}
DEFAULT_NORMALIZATION_LINESTYLES = {
    "non": "-",
    "qn": ":",
    "mn": "-.",
    "vsn": "--",
}
DEFAULT_ERROR_CURVE_LINEWIDTH = 1.2

# Pro workflow direction definitions
PRO_DIRECTIONS = ("batch_1_to_batch_2", "batch_2_to_batch_1")
PRO_DIRECTION_BATCHES = {
    "batch_1_to_batch_2": ("batch_1", "batch_2"),
    "batch_2_to_batch_1": ("batch_2", "batch_1"),
}
PRO_METRICS = ("raw_error", "balanced_error")


# -----------------------------------------------------------------------------
# Generic SSV core
# -----------------------------------------------------------------------------


class MolecularDataset:
    """Expression data with rows as features and columns as samples."""

    expression: pd.DataFrame
    class_labels: pd.Series
    batch_labels: pd.Series

    def __post_init__(self):
        if not isinstance(self.expression, pd.DataFrame):
            raise TypeError("expression must be a pandas DataFrame")
        if not isinstance(self.class_labels, pd.Series):
            raise TypeError("class_labels must be a pandas Series")
        if not isinstance(self.batch_labels, pd.Series):
            raise TypeError("batch_labels must be a pandas Series")
        if self.expression.empty:
            raise ValueError("expression must be non-empty")
        if not self.expression.columns.is_unique:
            raise ValueError("expression column sample IDs must be unique")
        if not self.class_labels.index.is_unique:
            raise ValueError("class label sample IDs must be unique")
        if not self.batch_labels.index.is_unique:
            raise ValueError("batch label sample IDs must be unique")
        if not self.expression.columns.equals(self.class_labels.index):
            raise ValueError("expression columns must exactly match class_labels.index")
        if not self.expression.columns.equals(self.batch_labels.index):
            raise ValueError("expression columns must exactly match batch_labels.index")

        self.expression = self.expression.copy()
        self.class_labels = self.class_labels.copy()
        self.batch_labels = self.batch_labels.copy()

def _ordered_members(sample_ids, selected):
    selected = set(selected)
    return [sample_id for sample_id in sample_ids if sample_id in selected]

def _ordered_unique(values):
    return list(pd.unique(pd.Series(values)))

def construct_meta_batches(dataset, mode="random", early_batch_ids=None, rng=None):
    """Construct two repetition-level meta-batches from raw batch labels."""
    sample_ids = list(dataset.expression.columns)
    batches = dataset.batch_labels
    raw_batches = pd.unique(batches.to_numpy())

    if mode == "temporal":
        if early_batch_ids is None:
            raise ValueError("early_batch_ids is required for temporal mode")
        early_batch_ids = set(early_batch_ids)
        group1_batches = [batch for batch in raw_batches if batch in early_batch_ids]
        group2_batches = [batch for batch in raw_batches if batch not in early_batch_ids]
    elif mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        # R batch_time=FALSE shuffles raw batch labels, then assigns whole labels to meta-batches.
        shuffled = rng.permutation(raw_batches)
        half_length = int(np.ceil(len(shuffled) / 2))
        group1_batches = list(shuffled[:half_length])
        group2_batches = list(shuffled[half_length:])
    else:
        raise ValueError("mode must be 'random' or 'temporal'")

    meta_batch_1 = [sample_id for sample_id in sample_ids if batches.loc[sample_id] in group1_batches]
    meta_batch_2 = [sample_id for sample_id in sample_ids if batches.loc[sample_id] in group2_batches]
    if not meta_batch_1 or not meta_batch_2:
        raise ValueError("both meta-batches must contain at least one sample")
    return meta_batch_1, meta_batch_2

def balance_meta_batches(meta_batch_1, meta_batch_2, rng):
    """Downsample the larger meta-batch to match the smaller one."""
    meta_batch_1 = list(meta_batch_1)
    meta_batch_2 = list(meta_batch_2)
    length_diff = abs(len(meta_batch_1) - len(meta_batch_2))
    if length_diff == 0:
        return meta_batch_1, meta_batch_2

    if len(meta_batch_1) > len(meta_batch_2):
        remove = set(rng.choice(meta_batch_1, size=length_diff, replace=False))
        return [sample_id for sample_id in meta_batch_1 if sample_id not in remove], meta_batch_2

    remove = set(rng.choice(meta_batch_2, size=length_diff, replace=False))
    return meta_batch_1, [sample_id for sample_id in meta_batch_2 if sample_id not in remove]

def r_sample_size(count, percentage):
    """Reproduce base R sample(size=...) coercion used by ml_sum().

    R accepts non-integer positive numeric size values in sample() and coerces
    them to integer by truncating toward zero, e.g. 984.9 -> 984.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if percentage < 0 or percentage > 100:
        raise ValueError("percentage must be between 0 and 100")
    return int(np.trunc(count * percentage / 100))

def split_meta_batches(meta_batch_1, meta_batch_2, split, rng, sample_order=None):
    """Split split-specific balanced meta-batches using the SSV percentage convention."""
    if split < 0 or split > 100:
        raise ValueError("split must be between 0 and 100")

    meta_batch_1 = list(meta_batch_1)
    meta_batch_2 = list(meta_batch_2)
    n_train_1 = r_sample_size(len(meta_batch_1), split)
    n_train_2 = r_sample_size(len(meta_batch_2), 100 - split)
    train_1 = set(rng.choice(meta_batch_1, size=n_train_1, replace=False)) if n_train_1 else set()
    train_2 = set(rng.choice(meta_batch_2, size=n_train_2, replace=False)) if n_train_2 else set()
    train_set = train_1 | train_2

    retained_set = set(meta_batch_1) | set(meta_batch_2)
    if sample_order is None:
        sample_order = meta_batch_1 + meta_batch_2
    train_ids = [sample_id for sample_id in sample_order if sample_id in train_set]
    test_ids = [sample_id for sample_id in sample_order if sample_id in retained_set and sample_id not in train_set]
    return train_ids, test_ids

def split_manifest_from_experiments(experiments):
    """Build a compact one-row-per-repetition/split audit manifest."""
    rows = []
    for experiment in experiments:
        for split_result in experiment["splits"]:
            train_ids = split_result["train_ids"]
            test_ids = split_result["test_ids"]
            rows.append(
                {
                    "repetition": experiment["repetition"],
                    "split_percentage": split_result["split_percentage"],
                    "meta_batch_1_labels": "|".join(map(str, experiment["meta_batch_1_labels"])),
                    "meta_batch_2_labels": "|".join(map(str, experiment["meta_batch_2_labels"])),
                    "meta_batch_1_n_before_balance": len(experiment["meta_batch_1_ids"]),
                    "meta_batch_2_n_before_balance": len(experiment["meta_batch_2_ids"]),
                    "meta_batch_1_n_after_balance": len(split_result["balanced_meta_batch_1_ids"]),
                    "meta_batch_2_n_after_balance": len(split_result["balanced_meta_batch_2_ids"]),
                    "train_1_n": len(set(train_ids) & set(split_result["balanced_meta_batch_1_ids"])),
                    "train_2_n": len(set(train_ids) & set(split_result["balanced_meta_batch_2_ids"])),
                    "train_n": len(train_ids),
                    "test_n": len(test_ids),
                    "overlap_n": len(set(train_ids) & set(test_ids)),
                }
            )
    return pd.DataFrame(rows)

def generate_split_experiments(
    dataset,
    split_percentages,
    times=1,
    mode="random",
    early_batch_ids=None,
    batch_balance=False,
    seed=2024,
):
    """Generate R ml_sum-style repetition meta-batches and split-specific train/test IDs."""
    sample_order = list(dataset.expression.columns)
    experiments = []
    for repetition in range(1, times + 1):
        # R uses set.seed(k + 2024) once per repetition.
        # NumPy uses a different RNG engine, so exact sampled IDs are not expected
        # to match R, but the seed hierarchy and RNG consumption order are preserved.
        rng = np.random.default_rng(seed + repetition)
        meta_batch_1, meta_batch_2 = construct_meta_batches(
            dataset,
            mode=mode,
            early_batch_ids=early_batch_ids,
            rng=rng,
        )

        meta_batch_1_labels = _ordered_unique(dataset.batch_labels.loc[meta_batch_1].to_numpy())
        meta_batch_2_labels = _ordered_unique(dataset.batch_labels.loc[meta_batch_2].to_numpy())

        splits = []
        for split in split_percentages:
            if batch_balance:
                balanced_meta_batch_1, balanced_meta_batch_2 = balance_meta_batches(meta_batch_1, meta_batch_2, rng)
            else:
                balanced_meta_batch_1, balanced_meta_batch_2 = list(meta_batch_1), list(meta_batch_2)

            train_ids, test_ids = split_meta_batches(
                balanced_meta_batch_1,
                balanced_meta_batch_2,
                split,
                rng,
                sample_order=sample_order,
            )
            splits.append(
                {
                    "split_percentage": split,
                    "train_ids": train_ids,
                    "test_ids": test_ids,
                    "balanced_meta_batch_1_ids": balanced_meta_batch_1,
                    "balanced_meta_batch_2_ids": balanced_meta_batch_2,
                }
            )

        experiments.append(
            {
                "repetition": repetition,
                "meta_batch_1_ids": meta_batch_1,
                "meta_batch_2_ids": meta_batch_2,
                "meta_batch_1_labels": meta_batch_1_labels,
                "meta_batch_2_labels": meta_batch_2_labels,
                "splits": splits,
            }
        )
    return experiments

def _copy_numeric_frame(frame, name="matrix"):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{name} must be non-empty")
    return frame.astype(float).copy()

def _restore_frame(values, index, columns):
    return pd.DataFrame(values, index=index.copy(), columns=columns.copy())

def identity_normalize(matrix):
    """Return an unchanged copy."""
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix must be a pandas DataFrame")
    if matrix.empty:
        raise ValueError("matrix must be non-empty")
    return matrix.copy(deep=True)

def quantile_normalize(matrix):
    """Normalize columns to a shared empirical rank distribution."""
    frame = _copy_numeric_frame(matrix)
    values = frame.to_numpy()
    sorted_values = np.sort(values, axis=0)
    rank_means = sorted_values.mean(axis=1)
    ranks = frame.rank(axis=0, method="average", ascending=True).to_numpy() - 1

    normalized = np.empty_like(values, dtype=float)
    rank_positions = np.arange(len(rank_means), dtype=float)
    for col_idx in range(values.shape[1]):
        normalized[:, col_idx] = np.interp(ranks[:, col_idx], rank_positions, rank_means)
    return _restore_frame(normalized, frame.index, frame.columns)

def fit_quantile_reference(train):
    """Fit the sorted QN reference distribution from training data."""
    train_qn = quantile_normalize(train)
    return np.sort(train_qn.iloc[:, 0].to_numpy())

def apply_quantile_reference(test, reference):
    """Map each test column to a training-derived rank reference."""
    frame = _copy_numeric_frame(test)
    reference = np.asarray(reference, dtype=float)
    if len(reference) != frame.shape[0]:
        raise ValueError("reference length must equal the number of features")
    ranks = frame.rank(axis=0, method="average", ascending=True).to_numpy() - 1
    normalized = np.empty(frame.shape, dtype=float)
    rank_positions = np.arange(len(reference), dtype=float)
    for col_idx in range(frame.shape[1]):
        normalized[:, col_idx] = np.interp(ranks[:, col_idx], rank_positions, reference)
    return _restore_frame(normalized, frame.index, frame.columns)

def median_normalize(train, test):
    """Median-shift train and test using the scalar training median."""
    train_frame = _copy_numeric_frame(train, "train")
    test_frame = _copy_numeric_frame(test, "test")
    reference = np.nanmedian(train_frame.to_numpy())
    train_norm = train_frame.subtract(train_frame.median(axis=0) - reference, axis=1)
    test_norm = test_frame.subtract(test_frame.median(axis=0) - reference, axis=1)
    return train_norm, test_norm

def normalize_transformed_qn(train, test):
    """R quant.norm semantics: QN train, then frozen-reference test mapping."""
    train_qn = quantile_normalize(train)
    reference = np.sort(train_qn.iloc[:, 0].to_numpy())
    test_qn = apply_quantile_reference(test, reference)
    return train_qn, test_qn

def fit_vsn_approximation(train):
    """Fit a lightweight training-only VSN approximation."""
    train_frame = _copy_numeric_frame(train, "train")
    location = train_frame.median(axis=1)
    mad = train_frame.subtract(location, axis=0).abs().median(axis=1)
    scale = (mad * 1.4826).astype(float)
    positive_scale = scale[np.isfinite(scale) & (scale > 0)]
    fallback = float(np.nanmedian(positive_scale)) if not positive_scale.empty else 1.0
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    scale = scale.where(np.isfinite(scale) & (scale > 0), fallback)
    return {"location": location.astype(float), "scale": scale.astype(float)}

def apply_vsn_approximation(matrix, state):
    """Apply training-fitted VSN approximation parameters without refitting."""
    frame = _copy_numeric_frame(matrix)
    location = state["location"]
    scale = state["scale"]
    if not frame.index.equals(location.index) or not frame.index.equals(scale.index):
        raise ValueError("VSN state feature IDs must match matrix feature IDs")
    values = np.arcsinh(frame.subtract(location, axis=0).divide(scale, axis=0).to_numpy(dtype=float))
    return _restore_frame(values, frame.index, frame.columns)

def normalize_vsn(train, test):
    """Leakage-safe Python approximation to transformed-expression VSN."""
    state = fit_vsn_approximation(train)
    return apply_vsn_approximation(train, state), apply_vsn_approximation(test, state)

def _check_positive_factors(factors, name="scaling factors"):
    factors = pd.Series(factors, dtype=float)
    if not np.isfinite(factors.to_numpy()).all() or (factors <= 0).any():
        raise ValueError(f"{name} must be finite and positive")
    return factors

def _scale_columns(matrix, factors):
    frame = _copy_numeric_frame(matrix)
    factors = _check_positive_factors(pd.Series(factors, index=frame.columns), "column scaling factors")
    return frame.divide(factors, axis=1)

def normalize_tc(raw_counts):
    """Total-count CPM normalization."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    factors = raw.sum(axis=0) / 1e6
    return _scale_columns(raw, factors), factors

def normalize_uq(raw_counts):
    """Upper-quartile scaling using strictly nonzero counts."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    factors = raw.apply(lambda col: np.quantile(col[col != 0], 0.75), axis=0) / 1e6
    return _scale_columns(raw, factors), factors

def normalize_rna_median(raw_counts):
    """RNA median scaling using strictly nonzero counts per sample."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    factors = raw.apply(lambda col: np.median(col[col != 0]), axis=0) / 1e6
    return _scale_columns(raw, factors), factors

def deseq_size_factors(raw_counts):
    """Estimate DESeq-style median-of-ratios size factors."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    positive_rows = (raw > 0).all(axis=1)
    if not positive_rows.any():
        raise ValueError("DESeq size factors require at least one feature positive in every sample")
    log_geo_means = np.log(raw.loc[positive_rows]).mean(axis=1)
    geo_means = np.exp(log_geo_means)
    ratios = raw.loc[positive_rows].divide(geo_means, axis=0)
    factors = ratios.median(axis=0)
    return _check_positive_factors(factors, "DESeq size factors")

def normalize_deseq(raw_counts):
    """DESeq size-factor normalization only."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    factors = deseq_size_factors(raw)
    return _scale_columns(raw, factors), factors

def _select_tmm_reference(raw):
    lib_sizes = raw.sum(axis=0)
    upper_quartiles = raw.apply(lambda col: np.quantile(col[col > 0], 0.75), axis=0) / lib_sizes
    return int(np.argmin(np.abs(upper_quartiles - upper_quartiles.mean())))

def tmm_factors(raw_counts, ref_col=None, logratio_trim=0.30, sum_trim=0.05):
    """Estimate core edgeR-style TMM normalization factors."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    lib_sizes = raw.sum(axis=0).astype(float)
    _check_positive_factors(lib_sizes, "library sizes")
    ref_idx = _select_tmm_reference(raw) if ref_col is None else ref_col
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
        weighted_m = np.average(m_values[trim_keep], weights=weights[trim_keep])
        factors.append(2 ** weighted_m)

    factors = pd.Series(factors, index=raw.columns, dtype=float)
    factors = factors / np.exp(np.mean(np.log(factors)))
    return _check_positive_factors(factors, "TMM factors")

def normalize_tmm(raw_counts, ref_col=None):
    """TMM factor estimation followed by CPM with effective library sizes."""
    raw = _copy_numeric_frame(raw_counts, "raw_counts")
    lib_sizes = raw.sum(axis=0)
    factors = tmm_factors(raw, ref_col=ref_col)
    scaling = factors * lib_sizes / 1e6
    return _scale_columns(raw, scaling), scaling

def normalize_tmm_frozen(train_counts, test_counts):
    """Normalize first train sample plus test cohort, then return test columns only."""
    train = _copy_numeric_frame(train_counts, "train_counts")
    test = _copy_numeric_frame(test_counts, "test_counts")
    combined = pd.concat([train.iloc[:, [0]], test], axis=1)
    combined_norm, scaling = normalize_tmm(combined, ref_col=0)
    return combined_norm.iloc[:, 1:].copy(), scaling.iloc[1:].copy()

def normalize_rna_qn_frozen(train_counts, test_counts):
    train_qn = quantile_normalize(train_counts)
    reference = np.sort(train_qn.iloc[:, 0].to_numpy())
    test_qn = apply_quantile_reference(test_counts, reference)
    return train_qn, test_qn

def preprocess_train_test(train, test, method, expression_source):
    """Normalize train and test matrices using the R workflow method names."""
    if expression_source == "transformed_expression":
        if method == "non":
            return identity_normalize(train), identity_normalize(test)
        if method == "qn":
            return normalize_transformed_qn(train, test)
        if method == "mn":
            return median_normalize(train, test)
        if method == "vsn":
            return normalize_vsn(train, test)
        raise ValueError("unknown transformed_expression method")

    if expression_source == "raw_counts":
        if method == "non":
            return identity_normalize(train), identity_normalize(test)
        if method == "TC":
            return normalize_tc(train)[0], normalize_tc(test)[0]
        if method == "UQ":
            return normalize_uq(train)[0], normalize_uq(test)[0]
        if method == "med":
            return normalize_rna_median(train)[0], normalize_rna_median(test)[0]
        if method == "TMM":
            return normalize_tmm(train)[0], normalize_tmm(test)[0]
        if method == "TMM.frozen":
            train_norm = normalize_tmm(train)[0]
            test_norm = normalize_tmm_frozen(train, test)[0]
            return train_norm, test_norm
        if method == "DESeq":
            return normalize_deseq(train)[0], normalize_deseq(test)[0]
        if method == "QN":
            return quantile_normalize(train), quantile_normalize(test)
        if method == "QN.frozen":
            return normalize_rna_qn_frozen(train, test)
        raise ValueError("unknown raw_counts method")

    raise ValueError("expression_source must be 'transformed_expression' or 'raw_counts'")

def _assert_same_axes(original, normalized):
    assert original.shape == normalized.shape
    assert original.index.equals(normalized.index)
    assert original.columns.equals(normalized.columns)
    assert np.isfinite(normalized.to_numpy()).all()

class ClassifierResult:
    predicted_labels: pd.Series
    probabilities: pd.Series | None
    inner_cv_error: float | None
    outer_test_error: float | None
    selected_params: dict

def _require_sklearn():
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for this classifier but is not installed.")

def _safe_stratified_cv(labels, requested_folds=5, random_state=2024):
    _require_sklearn()
    counts = pd.Series(labels).value_counts()
    if counts.min() < 2:
        raise ValueError("at least two samples per class are required for stratified CV")
    n_splits = int(min(requested_folds, counts.min()))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def _validate_classifier_inputs(train_expression, train_labels, test_expression, test_labels=None):
    if not isinstance(train_expression, pd.DataFrame) or not isinstance(test_expression, pd.DataFrame):
        raise TypeError("train_expression and test_expression must be pandas DataFrames")
    if not isinstance(train_labels, pd.Series):
        raise TypeError("train_labels must be a pandas Series")
    if not train_expression.columns.equals(train_labels.index):
        raise ValueError("train_expression columns must exactly match train_labels.index")
    if not train_expression.index.equals(test_expression.index):
        raise ValueError("train and test feature IDs must match exactly")
    if test_labels is not None:
        if not isinstance(test_labels, pd.Series):
            raise TypeError("test_labels must be a pandas Series")
        if not test_expression.columns.equals(test_labels.index):
            raise ValueError("test_expression columns must exactly match test_labels.index")

    classes = pd.Index(pd.unique(train_labels))
    if len(classes) != 2:
        raise ValueError("classifier helpers currently require exactly two training classes")

    train_x = train_expression.T.copy()
    test_x = test_expression.T.copy()
    return train_x, train_labels.copy(), test_x, None if test_labels is None else test_labels.copy(), classes

def _finish_classifier_result(predictions, probabilities, test_index, test_labels, inner_cv_error, selected_params):
    predicted_labels = pd.Series(predictions, index=test_index, name="predicted_label")
    prob_series = None
    if probabilities is not None:
        prob_series = pd.Series(probabilities, index=test_index, name="positive_class_probability")
    outer_error = None
    if test_labels is not None:
        outer_error = float(np.mean(predicted_labels.to_numpy() != test_labels.to_numpy()))
    return ClassifierResult(predicted_labels, prob_series, inner_cv_error, outer_error, selected_params)

def _positive_class_index(estimator, positive_class):
    return int(np.where(estimator.classes_ == positive_class)[0][0])

def _fit_grid_classifier(train_x, y, test_x, estimator, param_grid, cv, positive_class=None):
    grid = GridSearchCV(estimator, param_grid=param_grid, cv=cv, scoring="accuracy")
    grid.fit(train_x, y)
    predictions = grid.predict(test_x)
    probabilities = None
    if positive_class is not None and hasattr(grid.best_estimator_, "predict_proba"):
        class_idx = _positive_class_index(grid.best_estimator_, positive_class)
        probabilities = grid.best_estimator_.predict_proba(test_x)[:, class_idx]
    return predictions, probabilities, float(1 - grid.best_score_), dict(grid.best_params_)

def _fit_rf(train_x, y, test_x, cv, positive_class, random_state=2024):
    _require_sklearn()
    candidates = [
        {"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None},
        {"n_estimators": 150, "max_features": "sqrt", "min_samples_leaf": 2, "max_depth": None},
        {"n_estimators": 150, "max_features": 0.5, "min_samples_leaf": 1, "max_depth": 4},
        {"n_estimators": 200, "max_features": "log2", "min_samples_leaf": 1, "max_depth": None},
        {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 3, "max_depth": 6},
    ]
    best_score = -np.inf
    best_params = candidates[0]
    for params in candidates:
        estimator = Pipeline(
            [("scale", StandardScaler()), ("model", RandomForestClassifier(random_state=random_state, **params))]
        )
        score = cross_val_score(estimator, train_x, y, cv=cv, scoring="accuracy").mean()
        if score > best_score:
            best_score = float(score)
            best_params = params
    final_model = Pipeline(
        [("scale", StandardScaler()), ("model", RandomForestClassifier(random_state=random_state, **best_params))]
    )
    final_model.fit(train_x, y)
    predictions = final_model.predict(test_x)
    rf_model = final_model.named_steps["model"]
    class_idx = _positive_class_index(rf_model, positive_class)
    probabilities = final_model.predict_proba(test_x)[:, class_idx]
    return predictions, probabilities, float(1 - best_score), best_params.copy()

def _fit_xgb(train_x, y, test_x, classes, cv, random_state=2024):
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost is unavailable because `xgboost` is not installed.")
    class_to_int = {cls: idx for idx, cls in enumerate(classes)}
    y_int = y.map(class_to_int).astype(int).to_numpy()
    dtrain = xgb.DMatrix(train_x.to_numpy(dtype=float), label=y_int)
    best_error = np.inf
    best_params = None
    best_rounds = 1
    for min_child_weight in [1, 5, 9]:
        params = {
            "min_child_weight": min_child_weight,
            "objective": "binary:logistic",
            "eval_metric": "error",
            "seed": random_state,
        }
        cvout = xgb.cv(
            params=params,
            dtrain=dtrain,
            nfold=cv.get_n_splits(),
            num_boost_round=100,
            early_stopping_rounds=20,
            stratified=True,
            shuffle=True,
            seed=random_state,
            verbose_eval=False,
        )
        errors = cvout["test-error-mean"].to_numpy()
        candidate_error = float(errors.min())
        if candidate_error < best_error:
            best_error = candidate_error
            best_params = params
            best_rounds = int(errors.argmin() + 1)
    model = xgb.train(best_params, dtrain, num_boost_round=best_rounds, verbose_eval=False)
    probabilities = model.predict(xgb.DMatrix(test_x.to_numpy(dtype=float)))
    predictions = classes.to_numpy()[(probabilities > 0.5).astype(int)]
    return predictions, probabilities, best_error, {"min_child_weight": best_params["min_child_weight"], "n_rounds": best_rounds}

def fit_classifier(train_expression, train_labels, test_expression, test_labels=None, method="svm", config=None, random_state=2024):
    """Fit/tune one SSV classifier and predict test samples."""
    config = {} if config is None else dict(config)
    train_x, y, test_x, test_y, classes = _validate_classifier_inputs(
        train_expression, train_labels, test_expression, test_labels
    )
    positive_class = config.get("positive_class", classes[-1])
    cv = _safe_stratified_cv(y, requested_folds=config.get("kfold", 5), random_state=random_state)

    if method == "svm":
        _require_sklearn()
        estimator = Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf"))])
        grid = {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.1, 1]}
        predictions, probabilities, inner_error, selected = _fit_grid_classifier(train_x, y, test_x, estimator, grid, cv)
    elif method == "lasso":
        _require_sklearn()
        estimator = Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000))]
        )
        grid = {"model__C": [0.01, 0.1, 1, 10]}
        predictions, probabilities, inner_error, selected = _fit_grid_classifier(
            train_x, y, test_x, estimator, grid, cv, positive_class=positive_class
        )
        selected["selection_note"] = "R reports lambda.min CV error and predicts with lambda.1se; sklearn uses the selected C."
    elif method == "xgb":
        predictions, probabilities, inner_error, selected = _fit_xgb(train_x, y, test_x, classes, cv, random_state)
    elif method == "knn":
        _require_sklearn()
        max_valid_k = len(y) - int(np.ceil(len(y) / cv.get_n_splits()))
        k_values = [k for k in [1, 3, 5, 7, 9] if k <= max_valid_k]
        estimator = Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())])
        predictions, probabilities, inner_error, selected = _fit_grid_classifier(
            train_x, y, test_x, estimator, {"model__n_neighbors": k_values}, cv, positive_class=positive_class
        )
    elif method == "rf":
        predictions, probabilities, inner_error, selected = _fit_rf(train_x, y, test_x, cv, positive_class, random_state)
    else:
        raise ValueError("method must be one of: svm, lasso, xgb, knn, rf")

    return _finish_classifier_result(predictions, probabilities, test_x.index, test_y, inner_error, selected)

def _short_error_message(exc):
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    return message[:200]

def _binary_count_labels(labels):
    observed = pd.Index(pd.unique(labels))
    observed_set = set(observed)
    if {0, 1} <= observed_set:
        return 0, 1
    if {"class_0", "class_1"} <= observed_set:
        return "class_0", "class_1"
    ordered = sorted(observed, key=lambda value: str(value))
    if len(ordered) != 2:
        raise ValueError("outer-test diagnostics require exactly two classes")
    return ordered[0], ordered[1]

def _outer_test_diagnostics(test_labels, predicted_labels=None):
    test_labels = pd.Series(test_labels)
    class_0, class_1 = _binary_count_labels(test_labels)
    test_n = int(test_labels.shape[0])
    test_class_0 = int((test_labels == class_0).sum())
    test_class_1 = int((test_labels == class_1).sum())
    balanced_accuracy = np.nan
    if predicted_labels is not None:
        balanced_accuracy = float(balanced_accuracy_score(test_labels, pd.Series(predicted_labels, index=test_labels.index)))
    return {
        "test_n": test_n,
        "test_class_0": test_class_0,
        "test_class_1": test_class_1,
        "test_positive_rate": test_class_1 / test_n if test_n else np.nan,
        "majority_baseline_error": min(test_class_0, test_class_1) / test_n if test_n else np.nan,
        "balanced_accuracy": balanced_accuracy,
        "balanced_error": 1 - balanced_accuracy if np.isfinite(balanced_accuracy) else np.nan,
    }

def _failure_row(repetition, split, normalization_method, classifier_method, message, test_labels=None):
    diagnostics = (
        _outer_test_diagnostics(test_labels)
        if test_labels is not None
        else {
            "test_n": np.nan,
            "test_class_0": np.nan,
            "test_class_1": np.nan,
            "test_positive_rate": np.nan,
            "majority_baseline_error": np.nan,
            "balanced_accuracy": np.nan,
            "balanced_error": np.nan,
        }
    )
    return {
        "repetition": repetition,
        "split_percentage": split,
        "normalization_method": normalization_method,
        "classifier_method": classifier_method,
        "inner_cv_error": np.nan,
        "outer_test_error": np.nan,
        **diagnostics,
        "status": "failed",
        "selected_params": {},
        "message": message,
    }

def _format_elapsed_minutes(seconds):
    return f"{seconds / 60:.2f} min"

def _progress_checkpoints(total, steps=10):
    if total <= 0:
        return set()
    return {
        min(total, max(1, int(np.ceil(total * fraction / steps))))
        for fraction in range(1, steps + 1)
    }

def run_ssv_summary(
    dataset,
    split_combination=(50, 70, 80, 90, 100),
    n_repetitions=1,
    batch_mode="temporal",
    early_batch_ids=None,
    balance=False,
    expression_source="transformed_expression",
    normalization_methods=("non",),
    classifier_methods=("knn",),
    seed=2024,
    classifier_config=None,
    progress=False,
    progress_label=None,
):
    """Run the SSV repetition/split/normalization/classifier workflow."""
    if not isinstance(dataset, MolecularDataset):
        raise TypeError("dataset must be a MolecularDataset")
    if n_repetitions < 1:
        raise ValueError("n_repetitions must be at least 1")
    split_combination = tuple(split_combination)
    normalization_methods = tuple(normalization_methods)
    classifier_methods = tuple(classifier_methods)
    if not split_combination:
        raise ValueError("split_combination must be non-empty")
    if not normalization_methods:
        raise ValueError("normalization_methods must be non-empty")
    if not classifier_methods:
        raise ValueError("classifier_methods must be non-empty")
    if any(split < 0 or split > 100 for split in split_combination):
        raise ValueError("all split percentages must be between 0 and 100")

    rows = []
    total_units = n_repetitions * len(split_combination) * len(normalization_methods) * len(classifier_methods)
    completed_units = 0
    progress_points = _progress_checkpoints(total_units)
    if progress:
        import time

        progress_start = time.perf_counter()
        label = f" | {progress_label}" if progress_label else ""
        print(f"Evaluation units: {total_units}{label}")

    def report_progress():
        if not progress or completed_units not in progress_points:
            return
        elapsed = time.perf_counter() - progress_start
        percent = 100 * completed_units / total_units if total_units else 100
        print(
            f"Progress: {completed_units}/{total_units} ({percent:.1f}%) | "
            f"elapsed: {_format_elapsed_minutes(elapsed)}"
        )

    split_experiments = generate_split_experiments(
        dataset,
        split_percentages=split_combination,
        times=n_repetitions,
        mode=batch_mode,
        early_batch_ids=early_batch_ids,
        batch_balance=balance,
        seed=seed,
    )

    for experiment in split_experiments:
        repetition = experiment["repetition"]
        # Meta-batch assignment is fixed within a repetition; balancing is repeated inside each split.
        for split_result in experiment["splits"]:
            split = split_result["split_percentage"]
            train_ids = _ordered_members(dataset.expression.columns, split_result["train_ids"])
            test_ids = _ordered_members(dataset.expression.columns, split_result["test_ids"])

            train_expression = dataset.expression.loc[:, train_ids]
            test_expression = dataset.expression.loc[:, test_ids]
            train_labels = dataset.class_labels.loc[train_ids]
            test_labels = dataset.class_labels.loc[test_ids]

            for normalization_method in normalization_methods:
                # Normalization is performed after splitting to preserve train/test separation.
                try:
                    train_norm, test_norm = preprocess_train_test(
                        train_expression,
                        test_expression,
                        method=normalization_method,
                        expression_source=expression_source,
                    )
                except Exception as exc:
                    message = _short_error_message(exc)
                    for classifier_method in classifier_methods:
                        rows.append(
                            _failure_row(
                                repetition,
                                split,
                                normalization_method,
                                classifier_method,
                                message,
                                test_labels=test_labels,
                            )
                        )
                        completed_units += 1
                        report_progress()
                    continue

                for classifier_method in classifier_methods:
                    try:
                        result = fit_classifier(
                            train_norm,
                            train_labels,
                            test_norm,
                            test_labels=test_labels,
                            method=classifier_method,
                            config=classifier_config,
                            random_state=seed + repetition,
                        )
                        diagnostics = _outer_test_diagnostics(test_labels, result.predicted_labels)
                        rows.append(
                            {
                                "repetition": repetition,
                                "split_percentage": split,
                                "normalization_method": normalization_method,
                                "classifier_method": classifier_method,
                                "inner_cv_error": result.inner_cv_error,
                                "outer_test_error": result.outer_test_error,
                                **diagnostics,
                                "status": "success",
                                "selected_params": result.selected_params,
                                "message": "",
                            }
                        )
                        completed_units += 1
                        report_progress()
                    except Exception as exc:
                        rows.append(
                            _failure_row(
                                repetition,
                                split,
                                normalization_method,
                                classifier_method,
                                _short_error_message(exc),
                                test_labels=test_labels,
                            )
                        )
                        completed_units += 1
                        report_progress()

    result = pd.DataFrame(rows)
    result.attrs["split_manifest"] = split_manifest_from_experiments(split_experiments)
    return result

def aggregate_results(results):
    """Summarize successful outer-test errors while retaining failure counts."""
    required = {
        "split_percentage",
        "normalization_method",
        "classifier_method",
        "outer_test_error",
        "status",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")

    rows = []
    group_columns = ["split_percentage", "normalization_method", "classifier_method"]
    for keys, group in results.groupby(group_columns, sort=False, dropna=False):
        outer_errors = pd.to_numeric(group["outer_test_error"], errors="coerce")
        success_mask = (group["status"] == "success") & np.isfinite(outer_errors)
        successful_errors = outer_errors[success_mask]
        n_requested = int(len(group))
        n_success = int(len(successful_errors))
        n_failed = n_requested - n_success
        mean_error = float(successful_errors.mean()) if n_success else np.nan
        sd_error = float(successful_errors.std(ddof=1)) if n_success >= 2 else np.nan
        se_error = sd_error / np.sqrt(n_success) if n_success >= 2 else np.nan
        rows.append(
            {
                "split_percentage": keys[0],
                "normalization_method": keys[1],
                "classifier_method": keys[2],
                "n_requested": n_requested,
                "n_success": n_success,
                "n_failed": n_failed,
                "failure_rate": n_failed / n_requested,
                "mean_outer_error": mean_error,
                "sd_outer_error": sd_error,
                "se_outer_error": se_error,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "split_percentage",
            "normalization_method",
            "classifier_method",
            "n_requested",
            "n_success",
            "n_failed",
            "failure_rate",
            "mean_outer_error",
            "sd_outer_error",
            "se_outer_error",
        ],
    )

def _resolve_split_levels(split_values, split_levels=None):
    observed = _ordered_values(split_values)
    if split_levels is None:
        return observed
    split_levels = list(split_levels)
    return [split for split in split_levels if split in observed] + [
        split for split in observed if split not in split_levels
    ]

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

def validate_molecular_dataset(dataset, *, require_binary=True, require_batches=True, min_batches=2):
    """Validate generic SSV molecular dataset invariants."""
    if not isinstance(dataset, MolecularDataset):
        raise TypeError("dataset must be a MolecularDataset")
    if not isinstance(dataset.expression, pd.DataFrame):
        raise TypeError("dataset.expression must be a pandas DataFrame")
    if not isinstance(dataset.class_labels, pd.Series):
        raise TypeError("dataset.class_labels must be a pandas Series")
    if not isinstance(dataset.batch_labels, pd.Series):
        raise TypeError("dataset.batch_labels must be a pandas Series")
    if dataset.expression.empty:
        raise ValueError("expression must be non-empty")
    if dataset.expression.shape[1] == 0:
        raise ValueError("expression must contain samples")
    if not dataset.expression.columns.is_unique:
        raise ValueError("expression sample IDs must be unique")
    if not dataset.expression.index.is_unique:
        raise ValueError("expression feature IDs must be unique")
    if not dataset.class_labels.index.is_unique:
        raise ValueError("class label sample IDs must be unique")
    if not dataset.batch_labels.index.is_unique:
        raise ValueError("batch label sample IDs must be unique")
    if not dataset.expression.columns.equals(dataset.class_labels.index):
        raise ValueError("expression columns must exactly match class_labels.index")
    if not dataset.expression.columns.equals(dataset.batch_labels.index):
        raise ValueError("expression columns must exactly match batch_labels.index")
    if dataset.class_labels.isna().any():
        raise ValueError("class labels must not contain missing values")
    if dataset.batch_labels.isna().any():
        raise ValueError("batch labels must not contain missing values")
    if require_binary and dataset.class_labels.nunique(dropna=False) != 2:
        raise ValueError("binary SSV workflow requires exactly two classes")
    if require_batches and dataset.batch_labels.nunique(dropna=False) < min_batches:
        raise ValueError(f"SSV workflow requires at least {min_batches} batch/group labels")
    return dataset

def run_ssv_smoke_test(
    dataset,
    *,
    split_combination=(50, 100),
    n_repetitions=1,
    batch_mode="temporal",
    early_batch_ids=None,
    balance=False,
    expression_source="transformed_expression",
    normalization_methods=("non",),
    classifier_methods=("knn",),
    seed=2024,
    progress=False,
):
    """Run a small generic SSV smoke test and validate the returned result table."""
    validate_molecular_dataset(dataset)
    result = run_ssv_summary(
        dataset,
        split_combination=split_combination,
        n_repetitions=n_repetitions,
        batch_mode=batch_mode,
        early_batch_ids=early_batch_ids,
        balance=balance,
        expression_source=expression_source,
        normalization_methods=normalization_methods,
        classifier_methods=classifier_methods,
        seed=seed,
        progress=progress,
        progress_label="smoke test",
    )
    required_columns = {
        "repetition",
        "split_percentage",
        "normalization_method",
        "classifier_method",
        "inner_cv_error",
        "outer_test_error",
        "status",
    }
    missing = required_columns - set(result.columns)
    if missing:
        raise ValueError(f"smoke result is missing required columns: {sorted(missing)}")
    expected_rows = n_repetitions * len(tuple(split_combination)) * len(tuple(normalization_methods)) * len(tuple(classifier_methods))
    if result.shape[0] != expected_rows:
        raise AssertionError(f"expected {expected_rows} smoke rows, found {result.shape[0]}")
    if set(result["split_percentage"]) != set(split_combination):
        raise AssertionError("smoke result does not contain every requested split")
    if set(result["normalization_method"]) != set(normalization_methods):
        raise AssertionError("smoke result does not contain every requested normalization")
    if set(result["classifier_method"]) != set(classifier_methods):
        raise AssertionError("smoke result does not contain every requested classifier")
    if not result["status"].eq("success").all():
        raise AssertionError("smoke test expected all rows to succeed")
    if not result["outer_test_error"].between(0, 1).all():
        raise AssertionError("outer_test_error must be between 0 and 1")
    if not result["inner_cv_error"].between(0, 1).all():
        raise AssertionError("inner_cv_error must be between 0 and 1")
    return result

def _safe_artifact_stem(value):
    """Return a deterministic filesystem-safe stem for a short label."""
    safe = str(value)
    for character in ('/', '\\', ':', '*', '?', '"', '<', '>', '|'):
        safe = safe.replace(character, "_")
    return safe

def run_directional_ssv(
    dataset,
    *,
    directions,
    direction_batches,
    n_repetitions,
    split_combination,
    normalization_methods,
    classifier_methods,
    seed,
    batch_balance=True,
    data_type_rnaseq=False,
    direction_column="direction",
    checkpoint_dir=None,
    progress=False,
):
    """Run SSV for one or more supplied transfer directions."""
    validate_molecular_dataset(dataset)
    directions = tuple(directions)
    if not directions:
        raise ValueError("directions must be non-empty")
    missing_directions = [direction for direction in directions if direction not in direction_batches]
    if missing_directions:
        raise ValueError(f"direction_batches is missing directions: {missing_directions}")

    expression_source = "raw_counts" if data_type_rnaseq else "transformed_expression"
    if checkpoint_dir is not None:
        from pathlib import Path

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        import time

    results = []
    n_directions = len(directions)
    for direction_index, direction in enumerate(directions, start=1):
        if progress:
            direction_start = time.perf_counter()
            print(f"[Direction {direction_index}/{n_directions}] START - {direction}")
        early_batch_ids = tuple(direction_batches[direction])
        result = run_ssv_summary(
            dataset,
            split_combination=split_combination,
            n_repetitions=n_repetitions,
            batch_mode="temporal",
            early_batch_ids=early_batch_ids,
            balance=batch_balance,
            expression_source=expression_source,
            normalization_methods=normalization_methods,
            classifier_methods=classifier_methods,
            seed=seed,
            progress=progress,
            progress_label=str(direction),
        ).copy()
        result.insert(0, direction_column, direction)
        checkpoint_path = None
        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"{_safe_artifact_stem(direction)}_raw.csv"
            result.to_csv(checkpoint_path, index=False)
        if progress:
            elapsed = time.perf_counter() - direction_start
            print(f"[Direction {direction_index}/{n_directions}] DONE - {direction}")
            print(f"elapsed: {_format_elapsed_minutes(elapsed)}")
            if checkpoint_path is not None:
                print(f"checkpoint saved: {checkpoint_path}")
            print()
        results.append(result)
    return pd.concat(results, ignore_index=True)

def _ordered_values(values, preferred_order=None):
    observed = list(pd.unique(pd.Series(values).dropna()))
    if preferred_order is None:
        return observed
    preferred_order = list(preferred_order)
    return [value for value in preferred_order if value in observed] + [
        value for value in observed if value not in preferred_order
    ]

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

def aggregate_direction_results(results, direction_column="direction"):
    """Aggregate SSV results within each transfer direction."""
    required = {direction_column, "split_percentage", "normalization_method", "classifier_method", "outer_test_error", "status"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")

    summaries = []
    for direction, group in results.groupby(direction_column, sort=False, dropna=False):
        summary = aggregate_results(group).copy()
        summary.insert(0, direction_column, direction)
        summaries.append(summary)
    if not summaries:
        raise ValueError("no direction groups are available to aggregate")
    return pd.concat(summaries, ignore_index=True)

def _paired_bidirectional_errors(
    results,
    *,
    directions=None,
    direction_column="direction",
    error_column="outer_test_error",
):
    required = {
        direction_column,
        "repetition",
        "split_percentage",
        "normalization_method",
        "classifier_method",
        error_column,
        "status",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")

    directions = tuple(pd.unique(results[direction_column].dropna())) if directions is None else tuple(directions)
    if len(directions) != 2:
        raise ValueError("bidirectional aggregation requires exactly two directions")
    observed_directions = set(pd.unique(results[direction_column].dropna()))
    missing_directions = set(directions) - observed_directions
    if missing_directions:
        raise ValueError(f"missing required directions: {sorted(missing_directions)}")

    key_columns = ["repetition", "split_percentage", "normalization_method", "classifier_method"]
    paired_source = results.loc[results[direction_column].isin(directions), key_columns + [direction_column, error_column, "status"]].copy()
    paired_source[error_column] = pd.to_numeric(paired_source[error_column], errors="coerce")
    duplicate_mask = paired_source.duplicated(key_columns + [direction_column], keep=False)
    if duplicate_mask.any():
        duplicate_rows = paired_source.loc[duplicate_mask, key_columns + [direction_column]]
        raise ValueError(f"duplicate direction rows found for bidirectional pairing: {duplicate_rows.head().to_dict('records')}")

    direction_counts = paired_source.groupby(key_columns, dropna=False)[direction_column].nunique()
    incomplete_keys = direction_counts[direction_counts.ne(len(directions))]
    if not incomplete_keys.empty:
        raise ValueError("every repetition/configuration must contain both directions before pooling")

    status_pivot = paired_source.pivot(index=key_columns, columns=direction_column, values="status")
    error_pivot = paired_source.pivot(index=key_columns, columns=direction_column, values=error_column)
    if status_pivot.loc[:, list(directions)].ne("success").any().any():
        raise ValueError("bidirectional aggregation requires successful rows for both directions")
    if ~np.isfinite(error_pivot.loc[:, list(directions)].to_numpy(dtype=float)).all():
        raise ValueError("bidirectional aggregation requires finite outer-test errors for both directions")

    paired = error_pivot.loc[:, list(directions)].reset_index()
    paired["bidirectional_outer_error"] = paired.loc[:, list(directions)].mean(axis=1)
    expected = 0.5 * (paired[directions[0]] + paired[directions[1]])
    if not np.allclose(paired["bidirectional_outer_error"], expected, rtol=0, atol=1e-12):
        raise AssertionError("bidirectional errors must be the equal-weight average of the two paired directions")
    return paired

def aggregate_bidirectional_results(
    results,
    *,
    directions=None,
    direction_column="direction",
    error_column="outer_test_error",
):
    """Pair two directions within repetition, then summarize equal-weight bidirectional errors."""
    paired = _paired_bidirectional_errors(
        results,
        directions=directions,
        direction_column=direction_column,
        error_column=error_column,
    )
    group_columns = ["split_percentage", "normalization_method", "classifier_method"]
    rows = []
    for keys, group in paired.groupby(group_columns, sort=False, dropna=False):
        errors = group["bidirectional_outer_error"].astype(float)
        n_repetitions = int(errors.shape[0])
        sd_error = float(errors.std(ddof=1)) if n_repetitions >= 2 else np.nan
        rows.append(
            {
                "split_percentage": keys[0],
                "normalization_method": keys[1],
                "classifier_method": keys[2],
                "n_repetitions": n_repetitions,
                "mean_outer_error": float(errors.mean()) if n_repetitions else np.nan,
                "sd_outer_error": sd_error,
                "se_outer_error": sd_error / np.sqrt(n_repetitions) if n_repetitions >= 2 else np.nan,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "split_percentage",
            "normalization_method",
            "classifier_method",
            "n_repetitions",
            "mean_outer_error",
            "sd_outer_error",
            "se_outer_error",
        ],
    )

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

def save_ssv_run_artifacts(
    raw_results,
    *,
    output_dir,
    directions,
    classifier_methods=None,
    normalization_methods=None,
    split_levels=None,
    title_prefix="SSV",
    direction_column="direction",
    classifier_colors=None,
    normalization_linestyles=None,
    linewidth=DEFAULT_ERROR_CURVE_LINEWIDTH,
):
    """Save raw SSV results, summaries, and direction-specific/final error-curve figures."""
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    directions = tuple(directions)
    if len(directions) != 2:
        raise ValueError("save_ssv_run_artifacts requires exactly two directions")

    direction_summary = aggregate_direction_results(raw_results, direction_column=direction_column)
    bidirectional_summary = aggregate_bidirectional_results(
        raw_results,
        directions=directions,
        direction_column=direction_column,
    )
    paired_errors = _paired_bidirectional_errors(
        raw_results,
        directions=directions,
        direction_column=direction_column,
    )

    expected_pair_average = 0.5 * (paired_errors[directions[0]] + paired_errors[directions[1]])
    if not np.allclose(paired_errors["bidirectional_outer_error"], expected_pair_average, rtol=0, atol=1e-12):
        raise AssertionError("final bidirectional values are not exact paired equal-weight averages")

    paths = {
        "raw_results_csv": output_dir / "raw_results.csv",
        "direction_summary_csv": output_dir / "direction_summary.csv",
        "bidirectional_summary_csv": output_dir / "bidirectional_summary.csv",
    }
    raw_results.to_csv(paths["raw_results_csv"], index=False)
    direction_summary.to_csv(paths["direction_summary_csv"], index=False)
    bidirectional_summary.to_csv(paths["bidirectional_summary_csv"], index=False)

    figures = {}
    for direction in directions:
        fig, ax = plot_direction_error_curve(
            direction_summary,
            direction=direction,
            direction_column=direction_column,
            classifier_methods=classifier_methods,
            normalization_methods=normalization_methods,
            split_levels=split_levels,
            title=f"{title_prefix} - {direction}",
            classifier_colors=classifier_colors,
            normalization_linestyles=normalization_linestyles,
            linewidth=linewidth,
        )
        png_path = output_dir / f"{direction}_error_curve.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        paths[f"{direction}_png"] = png_path
        figures[direction] = (fig, ax)

    final_fig, final_ax = plot_bidirectional_error_curve(
        bidirectional_summary,
        classifier_methods=classifier_methods,
        normalization_methods=normalization_methods,
        split_levels=split_levels,
        title=f"{title_prefix} - equal-weight bidirectional",
        classifier_colors=classifier_colors,
        normalization_linestyles=normalization_linestyles,
        linewidth=linewidth,
    )
    paths["final_bidirectional_png"] = output_dir / "final_bidirectional_error_curve.png"
    final_fig.savefig(paths["final_bidirectional_png"], dpi=300, bbox_inches="tight")
    figures["final_bidirectional"] = (final_fig, final_ax)

    for key, path in paths.items():
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"expected artifact was not written: {key} -> {path}")

    return {
        "raw_results": raw_results,
        "direction_summary": direction_summary,
        "bidirectional_summary": bidirectional_summary,
        "paired_bidirectional_errors": paired_errors,
        "paths": paths,
        "figures": figures,
    }

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

def save_ssv_summary_artifacts(
    raw_results,
    *,
    output_dir,
    split_manifest=None,
    classifier_methods=None,
    normalization_methods=None,
    split_levels=None,
    title_prefix="SSV",
    classifier_colors=None,
    normalization_linestyles=None,
    linewidth=DEFAULT_ERROR_CURVE_LINEWIDTH,
):
    """Save R-style SSV raw results, summary, split manifest, and error curve."""
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "direction_summary.csv",
        "bidirectional_summary.csv",
        "final_bidirectional_error_curve.png",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    for stale_path in output_dir.glob("*_to_*_error_curve.png"):
        stale_path.unlink()
    if split_manifest is None:
        split_manifest = raw_results.attrs.get("split_manifest", pd.DataFrame())

    summary = aggregate_results(raw_results)
    paths = {
        "raw_results_csv": output_dir / "raw_results.csv",
        "summary_csv": output_dir / "summary.csv",
        "split_manifest_csv": output_dir / "split_manifest.csv",
        "error_curve_png": output_dir / "error_curve.png",
    }
    raw_results.to_csv(paths["raw_results_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    split_manifest.to_csv(paths["split_manifest_csv"], index=False)

    fig, ax = plot_ssv_error_curve(
        summary,
        classifier_methods=classifier_methods,
        normalization_methods=normalization_methods,
        split_levels=split_levels,
        title=f"{title_prefix} - R-style SSV reproduction",
        classifier_colors=classifier_colors,
        normalization_linestyles=normalization_linestyles,
        linewidth=linewidth,
    )
    fig.savefig(paths["error_curve_png"], dpi=300, bbox_inches="tight")

    for key, path in paths.items():
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"expected artifact was not written: {key} -> {path}")

    return {
        "raw_results": raw_results,
        "summary": summary,
        "split_manifest": split_manifest,
        "paths": paths,
        "figures": {"error_curve": (fig, ax)},
    }

def run_ssv_experiment_batch(
    experiments,
    *,
    output_root,
    n_repetitions,
    split_combination,
    normalization_methods,
    classifier_methods,
    seed,
    batch_mode="random",
    batch_balance=True,
    data_type_rnaseq=False,
    directions=None,
    direction_batches=None,
    early_batch_ids=None,
    run_smoke_test=False,
    overwrite=True,
    experiment_order=None,
    progress=True,
):
    """Run SSV experiments and save a manifest.

    When directions is None, this follows the R ml_sum-style single workflow.
    Passing directions keeps the older bidirectional extended-analysis path available.
    """
    from pathlib import Path
    import time

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if experiment_order is None:
        experiment_names = list(experiments)
    else:
        experiment_names = list(experiment_order)
        missing = [name for name in experiment_names if name not in experiments]
        if missing:
            raise ValueError(f"experiment_order contains names missing from experiments: {missing}")
    experiment_items = [(name, experiments[name]) for name in experiment_names]
    directional_mode = directions is not None
    if directional_mode and direction_batches is None:
        raise ValueError("direction_batches is required when directions are provided")

    manifest_path = output_root / "ssv_batch_manifest.csv"
    manifest_columns = [
        "experiment",
        "status",
        "n_raw_rows",
        "output_directory",
        "raw_results_file",
        "summary_file",
        "split_manifest_file",
        "error_curve_file",
        "direction_summary_file",
        "bidirectional_summary_file",
        "n_direction_plots",
        "final_plot_file",
        "message",
    ]

    total_experiments = len(experiment_items)
    batch_start_time = time.perf_counter()
    if progress:
        print("=" * 64)
        print("SSV BATCH RUN")
        print(f"Experiments: {total_experiments}")
        print("=" * 64)
        print()

    manifest_rows = []
    for experiment_index, (experiment_name, experiment) in enumerate(experiment_items, start=1):
        output_name = experiment.get("output_name", experiment_name)
        output_dir = output_root / output_name
        status = "complete"
        message = ""
        raw_results = pd.DataFrame()
        artifact_paths = {}
        if progress:
            experiment_start = time.perf_counter()
            print(f"[{experiment_index}/{total_experiments}] START - {experiment_name}")
            print()
        try:
            dataset = experiment["dataset"]
            validate_molecular_dataset(dataset)
            if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
                raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            expression_source = "raw_counts" if data_type_rnaseq else "transformed_expression"
            if run_smoke_test:
                smoke_kwargs = {
                    "split_combination": (tuple(split_combination)[0],),
                    "n_repetitions": 1,
                    "batch_mode": batch_mode,
                    "early_batch_ids": early_batch_ids,
                    "balance": batch_balance,
                    "expression_source": expression_source,
                    "normalization_methods": (tuple(normalization_methods)[0],),
                    "classifier_methods": (tuple(classifier_methods)[0],),
                    "seed": seed,
                    "progress": False,
                }
                if directional_mode:
                    smoke_kwargs["batch_mode"] = "temporal"
                    smoke_kwargs["early_batch_ids"] = tuple(direction_batches[tuple(directions)[0]])
                run_ssv_smoke_test(dataset, **smoke_kwargs)

            if directional_mode:
                raw_results = run_directional_ssv(
                    dataset,
                    directions=directions,
                    direction_batches=direction_batches,
                    n_repetitions=n_repetitions,
                    split_combination=split_combination,
                    normalization_methods=normalization_methods,
                    classifier_methods=classifier_methods,
                    seed=seed,
                    batch_balance=batch_balance,
                    data_type_rnaseq=data_type_rnaseq,
                    checkpoint_dir=output_dir / "checkpoints",
                    progress=progress,
                )
                if progress:
                    print("Saving final artifacts...")
                    print()
                artifacts = save_ssv_run_artifacts(
                    raw_results,
                    output_dir=output_dir,
                    directions=directions,
                    classifier_methods=classifier_methods,
                    normalization_methods=normalization_methods,
                    split_levels=split_combination,
                    title_prefix=experiment_name,
                )
                artifact_paths = artifacts["paths"]
                artifact_order = [
                    ("raw_results.csv", artifact_paths.get("raw_results_csv")),
                    ("direction_summary.csv", artifact_paths.get("direction_summary_csv")),
                    ("bidirectional_summary.csv", artifact_paths.get("bidirectional_summary_csv")),
                ]
                for direction in directions:
                    artifact_order.append(
                        (f"{_safe_artifact_stem(direction)}_error_curve.png", artifact_paths.get(f"{direction}_png"))
                    )
                artifact_order.append(
                    ("final_bidirectional_error_curve.png", artifact_paths.get("final_bidirectional_png"))
                )
            else:
                raw_results = run_ssv_summary(
                    dataset,
                    split_combination=split_combination,
                    n_repetitions=n_repetitions,
                    batch_mode=batch_mode,
                    early_batch_ids=early_batch_ids,
                    balance=batch_balance,
                    expression_source=expression_source,
                    normalization_methods=normalization_methods,
                    classifier_methods=classifier_methods,
                    seed=seed,
                    progress=progress,
                    progress_label=str(experiment_name),
                )
                split_manifest = raw_results.attrs.get("split_manifest", pd.DataFrame())
                if progress:
                    print("Saving final artifacts...")
                    print()
                artifacts = save_ssv_summary_artifacts(
                    raw_results,
                    output_dir=output_dir,
                    split_manifest=split_manifest,
                    classifier_methods=classifier_methods,
                    normalization_methods=normalization_methods,
                    split_levels=split_combination,
                    title_prefix=experiment_name,
                )
                artifact_paths = artifacts["paths"]
                artifact_order = [
                    ("raw_results.csv", artifact_paths.get("raw_results_csv")),
                    ("summary.csv", artifact_paths.get("summary_csv")),
                    ("split_manifest.csv", artifact_paths.get("split_manifest_csv")),
                    ("error_curve.png", artifact_paths.get("error_curve_png")),
                ]

            if progress:
                for label, path in artifact_order:
                    result = "PASS" if path is not None and Path(path).exists() else "MISSING"
                    print(f"{result} - {label}")
                print()
        except Exception as exc:
            status = "failed"
            message = f"{type(exc).__name__}: {exc}"
            if progress:
                print(f"FAIL - {experiment_name} | {message}")
                print()

        if progress:
            elapsed = time.perf_counter() - experiment_start
            done_label = "DONE" if status == "complete" else "FAILED"
            print(
                f"[{experiment_index}/{total_experiments}] {done_label} - {experiment_name} | "
                f"elapsed: {_format_elapsed_minutes(elapsed)}"
            )
            print()

        manifest_rows.append(
            {
                "experiment": experiment_name,
                "status": status,
                "n_raw_rows": int(raw_results.shape[0]) if isinstance(raw_results, pd.DataFrame) else 0,
                "output_directory": str(output_dir),
                "raw_results_file": str(artifact_paths.get("raw_results_csv", "")),
                "summary_file": str(artifact_paths.get("summary_csv", "")),
                "split_manifest_file": str(artifact_paths.get("split_manifest_csv", "")),
                "error_curve_file": str(artifact_paths.get("error_curve_png", "")),
                "direction_summary_file": str(artifact_paths.get("direction_summary_csv", "")),
                "bidirectional_summary_file": str(artifact_paths.get("bidirectional_summary_csv", "")),
                "n_direction_plots": int(sum(str(key).endswith("_png") and key != "final_bidirectional_png" for key in artifact_paths)) if directional_mode else 0,
                "final_plot_file": str(artifact_paths.get("final_bidirectional_png", "")),
                "message": message,
            }
        )
        pd.DataFrame(manifest_rows, columns=manifest_columns).to_csv(manifest_path, index=False)

    manifest = pd.DataFrame(manifest_rows, columns=manifest_columns)
    manifest.to_csv(manifest_path, index=False)
    if progress:
        total_elapsed = time.perf_counter() - batch_start_time
        print("=" * 64)
        print("SSV BATCH RUN COMPLETE")
        print("=" * 64)
        print()
        for row in manifest_rows:
            label = "PASS" if row["status"] == "complete" else "FAIL"
            print(f"{label} - {row['experiment']}")
        print()
        print(f"Total elapsed: {_format_elapsed_minutes(total_elapsed)}")
        print(f"Manifest: {manifest_path}")
        print("=" * 64)
    return manifest

def _finite_ratio(max_count, min_count):
    if min_count == 0:
        return np.inf if max_count > 0 else np.nan
    return float(max_count / min_count)

def _imbalance_status(ratio, *, moderate=2.0, high=4.0):
    if not np.isfinite(ratio):
        return "HIGH"
    if ratio >= high:
        return "HIGH"
    if ratio >= moderate:
        return "MODERATE"
    return "LOW"

def _association_status(cramers_v, abs_prevalence_difference):
    if (np.isfinite(cramers_v) and cramers_v >= 0.50) or (
        np.isfinite(abs_prevalence_difference) and abs_prevalence_difference >= 0.20
    ):
        return "HIGH"
    if (np.isfinite(cramers_v) and cramers_v >= 0.30) or (
        np.isfinite(abs_prevalence_difference) and abs_prevalence_difference >= 0.10
    ):
        return "MODERATE"
    return "LOW"

def compute_batch_class_qc(dataset):
    """Return batch x class composition and one-row Pro batch/class QC summary."""
    qc_summary, contingency = summarize_batch_structure(dataset)
    composition = contingency.reset_index().rename(columns={"index": "batch"})
    association = qc_summary.set_index("domain").loc["label_batch_association"]
    class_counts = dataset.class_labels.value_counts(dropna=False)
    batch_counts = dataset.batch_labels.value_counts(dropna=False)
    summary = pd.DataFrame(
        [
            {
                "n_classes": int(class_counts.shape[0]),
                "n_batches": int(batch_counts.shape[0]),
                "class_imbalance_ratio": float(qc_summary.set_index("domain").loc["label_balance", "ratio"]),
                "batch_imbalance_ratio": float(qc_summary.set_index("domain").loc["batch_balance", "ratio"]),
                "abs_prevalence_difference": float(association["abs_prevalence_difference"]),
                "prevalence_ratio": np.nan,
                "cramers_v": float(association["cramers_v"]),
                "chi_square_p_value": float(association["chi_square_p_value"]),
                "confounding_status": association["status"],
            }
        ]
    )
    return composition, summary

def summarize_batch_structure(dataset, scenario=None):
    """Summarize class balance, batch balance, and label-batch association.

    Returns
    -------
    qc_summary : pandas.DataFrame
        One tidy row for label balance, batch balance, and label-batch association.
    contingency : pandas.DataFrame
        Batch x class count table used for the association diagnostics.
    """
    validate_molecular_dataset(dataset, require_binary=False)
    frame = pd.DataFrame(
        {
            "batch": dataset.batch_labels.astype(str),
            "class": dataset.class_labels.astype(str),
        },
        index=dataset.expression.columns,
    )

    class_counts = frame["class"].value_counts(dropna=False).sort_index()
    batch_counts = frame["batch"].value_counts(dropna=False).sort_index()
    contingency = pd.crosstab(frame["batch"], frame["class"]).sort_index(axis=0).sort_index(axis=1)

    class_ratio = _finite_ratio(int(class_counts.max()), int(class_counts.min()))
    batch_ratio = _finite_ratio(int(batch_counts.max()), int(batch_counts.min()))
    class_minority = float(class_counts.min() / class_counts.sum()) if class_counts.sum() else np.nan
    batch_minority = float(batch_counts.min() / batch_counts.sum()) if batch_counts.sum() else np.nan

    class_props = contingency.div(contingency.sum(axis=1).replace(0, np.nan), axis=0)
    class_ranges = class_props.max(axis=0) - class_props.min(axis=0)
    abs_prevalence_difference = float(class_ranges.max()) if not class_ranges.empty else np.nan

    cramers_v = np.nan
    chi_square_p_value = np.nan
    table = contingency.to_numpy(dtype=float)
    if table.shape[0] >= 2 and table.shape[1] >= 2 and table.sum() > 0:
        try:
            from scipy.stats import chi2_contingency

            chi2, chi_square_p_value, _, _ = chi2_contingency(table, correction=False)
            denom = table.sum() * (min(table.shape) - 1)
            cramers_v = float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan
        except Exception:
            cramers_v = np.nan
            chi_square_p_value = np.nan

    rows = [
        {
            "scenario": scenario,
            "domain": "label_balance",
            "metric": "max_class_count / min_class_count",
            "value": class_counts.to_dict(),
            "ratio": float(class_ratio),
            "minority_proportion": class_minority,
            "status": _imbalance_status(class_ratio),
            "cramers_v": np.nan,
            "abs_prevalence_difference": np.nan,
            "chi_square_p_value": np.nan,
        },
        {
            "scenario": scenario,
            "domain": "batch_balance",
            "metric": "max_batch_count / min_batch_count",
            "value": batch_counts.to_dict(),
            "ratio": float(batch_ratio),
            "minority_proportion": batch_minority,
            "status": _imbalance_status(batch_ratio),
            "cramers_v": np.nan,
            "abs_prevalence_difference": np.nan,
            "chi_square_p_value": np.nan,
        },
        {
            "scenario": scenario,
            "domain": "label_batch_association",
            "metric": "Cramer's V and max class-prevalence range across batches",
            "value": contingency.to_dict(),
            "ratio": np.nan,
            "minority_proportion": np.nan,
            "status": _association_status(cramers_v, abs_prevalence_difference),
            "cramers_v": cramers_v,
            "abs_prevalence_difference": abs_prevalence_difference,
            "chi_square_p_value": float(chi_square_p_value) if np.isfinite(chi_square_p_value) else np.nan,
        },
    ]
    return pd.DataFrame(rows), contingency

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

def _ordered_dataset_members(dataset, selected):
    return _ordered_members(dataset.expression.columns, selected)

def balance_batches_once(dataset, train_batch, test_batch, rng):
    train_ids = list(dataset.batch_labels.index[dataset.batch_labels == train_batch])
    test_ids = list(dataset.batch_labels.index[dataset.batch_labels == test_batch])
    return balance_meta_batches(train_ids, test_ids, rng)

def balance_batches_by_class(dataset, train_batch, test_batch, rng):
    train_retained, test_retained = [], []
    for class_label in sorted(pd.unique(dataset.class_labels), key=lambda value: str(value)):
        train_ids = [sid for sid in dataset.expression.columns if dataset.batch_labels.loc[sid] == train_batch and dataset.class_labels.loc[sid] == class_label]
        test_ids = [sid for sid in dataset.expression.columns if dataset.batch_labels.loc[sid] == test_batch and dataset.class_labels.loc[sid] == class_label]
        keep = min(len(train_ids), len(test_ids))
        if keep:
            train_retained.extend(_ordered_dataset_members(dataset, set(rng.choice(train_ids, size=keep, replace=False))))
            test_retained.extend(_ordered_dataset_members(dataset, set(rng.choice(test_ids, size=keep, replace=False))))
    return _ordered_dataset_members(dataset, train_retained), _ordered_dataset_members(dataset, test_retained)

def construct_fixed_retained_population(dataset, train_batch, test_batch, rng, prevalence_mode="natural"):
    if prevalence_mode == "natural":
        return balance_batches_once(dataset, train_batch, test_batch, rng)
    if prevalence_mode == "class_stratified":
        return balance_batches_by_class(dataset, train_batch, test_batch, rng)
    raise ValueError("prevalence_mode must be 'natural' or 'class_stratified'")

def generate_directional_splits_pro(dataset, *, direction, split_percentages, n_repetitions, seed, prevalence_mode="natural"):
    if direction not in PRO_DIRECTION_BATCHES:
        raise ValueError(f"unknown Pro direction: {direction}")
    train_batch, test_batch = PRO_DIRECTION_BATCHES[direction]
    repetitions = []
    for repetition in range(1, n_repetitions + 1):
        rng = np.random.default_rng(seed + repetition)
        retained_train, retained_test = construct_fixed_retained_population(dataset, train_batch, test_batch, rng, prevalence_mode=prevalence_mode)
        retained_train = _ordered_dataset_members(dataset, retained_train)
        retained_test = _ordered_dataset_members(dataset, retained_test)
        splits = []
        for split in tuple(split_percentages):
            n_train_from_train = r_sample_size(len(retained_train), split)
            n_train_from_test = r_sample_size(len(retained_test), 100 - split)
            train_from_train = set(rng.choice(retained_train, size=n_train_from_train, replace=False)) if n_train_from_train else set()
            train_from_test = set(rng.choice(retained_test, size=n_train_from_test, replace=False)) if n_train_from_test else set()
            train_set = train_from_train | train_from_test
            retained_set = set(retained_train) | set(retained_test)
            train_ids = _ordered_dataset_members(dataset, train_set)
            test_ids = [sid for sid in dataset.expression.columns if sid in retained_set and sid not in train_set]
            splits.append({"split_percentage": split, "train_ids": train_ids, "test_ids": test_ids, "retained_train_batch_ids": retained_train, "retained_test_batch_ids": retained_test})
        repetitions.append({"repetition": repetition, "direction": direction, "train_batch": train_batch, "test_batch": test_batch, "prevalence_mode": prevalence_mode, "retained_train_batch_ids": retained_train, "retained_test_batch_ids": retained_test, "splits": splits})
    return repetitions

def pro_split_manifest(split_experiments):
    rows = []
    for experiment in split_experiments:
        retained_train = set(experiment["retained_train_batch_ids"])
        retained_test = set(experiment["retained_test_batch_ids"])
        for split_result in experiment["splits"]:
            train_ids = set(split_result["train_ids"])
            test_ids = set(split_result["test_ids"])
            rows.append({"repetition": experiment["repetition"], "direction": experiment["direction"], "prevalence_mode": experiment["prevalence_mode"], "split_percentage": split_result["split_percentage"], "train_batch": experiment["train_batch"], "test_batch": experiment["test_batch"], "retained_train_batch_n": len(retained_train), "retained_test_batch_n": len(retained_test), "train_from_train_batch_n": len(train_ids & retained_train), "train_from_test_batch_n": len(train_ids & retained_test), "train_n": len(train_ids), "test_n": len(test_ids), "overlap_n": len(train_ids & test_ids)})
    return pd.DataFrame(rows)

def run_ssv_pro_direction(dataset, *, direction, split_combination, n_repetitions, normalization_methods, classifier_methods, seed, prevalence_mode="natural", data_type_rnaseq=False, progress=True):
    expression_source = "raw_counts" if data_type_rnaseq else "transformed_expression"
    split_experiments = generate_directional_splits_pro(dataset, direction=direction, split_percentages=split_combination, n_repetitions=n_repetitions, seed=seed, prevalence_mode=prevalence_mode)
    rows = []
    for experiment in split_experiments:
        repetition = experiment["repetition"]
        for split_result in experiment["splits"]:
            split = split_result["split_percentage"]
            train_ids, test_ids = split_result["train_ids"], split_result["test_ids"]
            train_expression = dataset.expression.loc[:, train_ids]
            test_expression = dataset.expression.loc[:, test_ids]
            train_labels = dataset.class_labels.loc[train_ids]
            test_labels = dataset.class_labels.loc[test_ids]
            for normalization_method in normalization_methods:
                try:
                    train_norm, test_norm = preprocess_train_test(train_expression, test_expression, method=normalization_method, expression_source=expression_source)
                except Exception as exc:
                    for classifier_method in classifier_methods:
                        failure = _failure_row(repetition, split, normalization_method, classifier_method, _short_error_message(exc), test_labels=test_labels)
                        failure.update({"direction": direction, "prevalence_mode": prevalence_mode, "raw_error": np.nan, "balanced_error": np.nan})
                        rows.append(failure)
                    continue
                for classifier_method in classifier_methods:
                    try:
                        result = fit_classifier(train_norm, train_labels, test_norm, test_labels=test_labels, method=classifier_method, random_state=seed + repetition)
                        diagnostics = _outer_test_diagnostics(test_labels, result.predicted_labels)
                        balanced_error = 1 - diagnostics["balanced_accuracy"] if np.isfinite(diagnostics["balanced_accuracy"]) else np.nan
                        rows.append({"direction": direction, "prevalence_mode": prevalence_mode, "repetition": repetition, "split_percentage": split, "normalization_method": normalization_method, "classifier_method": classifier_method, "inner_cv_error": result.inner_cv_error, "outer_test_error": result.outer_test_error, "raw_error": result.outer_test_error, **diagnostics, "balanced_error": balanced_error, "status": "success", "selected_params": result.selected_params, "message": ""})
                    except Exception as exc:
                        failure = _failure_row(repetition, split, normalization_method, classifier_method, _short_error_message(exc), test_labels=test_labels)
                        failure.update({"direction": direction, "prevalence_mode": prevalence_mode, "raw_error": np.nan, "balanced_error": np.nan})
                        rows.append(failure)
    result = pd.DataFrame(rows)
    result.attrs["split_manifest"] = pro_split_manifest(split_experiments)
    return result

def run_ssv_pro_bidirectional(dataset, **kwargs):
    frames, manifests = [], []

    for direction in PRO_DIRECTIONS:
        result = run_ssv_pro_direction(
            dataset,
            direction=direction,
            **kwargs,
        )

        # Preserve split_manifest separately before concat.
        manifest = result.attrs.get(
            "split_manifest",
            pd.DataFrame(),
        )

        if isinstance(manifest, pd.DataFrame):
            manifest = manifest.copy()

            if not manifest.empty:
                manifests.append(manifest)

        # IMPORTANT:
        # DataFrame.copy() preserves attrs, so clear them explicitly.
        result_clean = result.copy()
        result_clean.attrs = {}

        frames.append(result_clean)

    if not frames:
        raw = pd.DataFrame()
    else:
        raw = pd.concat(
            frames,
            ignore_index=True,
        )

    if manifests:
        combined_manifest = pd.concat(
            manifests,
            ignore_index=True,
        )
    else:
        combined_manifest = pd.DataFrame()

    raw.attrs["split_manifest"] = combined_manifest

    return raw

def aggregate_pro_direction_results(raw_results):
    rows = []
    group_columns = ["prevalence_mode", "direction", "split_percentage", "normalization_method", "classifier_method"]
    for keys, group in raw_results[raw_results["status"].eq("success")].groupby(group_columns, sort=False):
        row = dict(zip(group_columns, keys))
        row["n"] = int(group.shape[0])
        for metric in PRO_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"mean_{metric}"] = float(values.mean())
            row[f"sd_{metric}"] = float(values.std(ddof=1)) if values.shape[0] >= 2 else np.nan
            row[f"se_{metric}"] = row[f"sd_{metric}"] / np.sqrt(values.shape[0]) if values.shape[0] >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_equal_weight_directions(raw_results):
    success = raw_results[raw_results["status"].eq("success")].copy()
    index_cols = ["prevalence_mode", "repetition", "split_percentage", "normalization_method", "classifier_method"]
    summaries = []
    for metric in PRO_METRICS:
        wide = success.pivot_table(index=index_cols, columns="direction", values=metric, aggfunc="mean").reset_index()
        for direction in PRO_DIRECTIONS:
            if direction not in wide:
                raise ValueError(f"missing direction for equal-weight aggregation: {direction}")
        wide[f"equal_weight_{metric}"] = 0.5 * (wide[PRO_DIRECTIONS[0]] + wide[PRO_DIRECTIONS[1]])
        group_columns = ["prevalence_mode", "split_percentage", "normalization_method", "classifier_method"]
        metric_summary = wide.groupby(group_columns, as_index=False).agg(n_repetitions=(f"equal_weight_{metric}", "size"), **{f"mean_{metric}": (f"equal_weight_{metric}", "mean"), f"sd_{metric}": (f"equal_weight_{metric}", "std")})
        metric_summary[f"se_{metric}"] = metric_summary[f"sd_{metric}"] / np.sqrt(metric_summary["n_repetitions"])
        summaries.append(metric_summary)
    output = summaries[0]
    for summary in summaries[1:]:
        output = output.merge(summary, on=["prevalence_mode", "split_percentage", "normalization_method", "classifier_method", "n_repetitions"], how="outer")
    return output

def compute_direction_gap(raw_results):
    direction_summary = aggregate_pro_direction_results(raw_results)
    split100 = direction_summary[direction_summary["split_percentage"].eq(100)]
    rows = []
    for metric in PRO_METRICS:
        wide = split100.pivot_table(index=["prevalence_mode", "normalization_method", "classifier_method"], columns="direction", values=f"mean_{metric}", aggfunc="mean").reset_index()
        for direction in PRO_DIRECTIONS:
            if direction not in wide:
                wide[direction] = np.nan
        wide[f"signed_direction_gap_100_{metric}"] = wide[PRO_DIRECTIONS[0]] - wide[PRO_DIRECTIONS[1]]
        wide[f"absolute_direction_gap_100_{metric}"] = wide[f"signed_direction_gap_100_{metric}"].abs()
        wide["metric"] = metric
        rows.append(wide)
    return pd.concat(rows, ignore_index=True)

def _plot_pro_equal_weight(equal_weight_summary, *, metric, title, output_path):
    metric_col, se_col = f"mean_{metric}", f"se_{metric}"
    summary = equal_weight_summary.rename(columns={metric_col: "mean_outer_error", se_col: "se_outer_error"})
    fig, ax = plot_ssv_error_curve(summary, classifier_methods=ACTIVE_CLASSIFIER_METHODS, normalization_methods=("non", "qn", "mn", "vsn"), split_levels=(50, 70, 80, 90, 100), title=title)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

def save_ssv_pro_artifacts(raw_results, *, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_results.to_csv(output_dir / "raw_results.csv", index=False)
    raw_results.attrs.get("split_manifest", pd.DataFrame()).to_csv(output_dir / "split_manifest.csv", index=False)
    direction_summary = aggregate_pro_direction_results(raw_results)
    equal_weight_summary = aggregate_equal_weight_directions(raw_results)
    direction_gap = compute_direction_gap(raw_results)
    direction_summary.to_csv(output_dir / "direction_summary.csv", index=False)
    equal_weight_summary.to_csv(output_dir / "equal_weight_summary.csv", index=False)
    direction_gap.to_csv(output_dir / "direction_gap.csv", index=False)
    _plot_pro_equal_weight(equal_weight_summary, metric="raw_error", title="Equal-weight raw error", output_path=output_dir / "equal_weight_raw_error_curve.png")
    _plot_pro_equal_weight(equal_weight_summary, metric="balanced_error", title="Equal-weight balanced error", output_path=output_dir / "equal_weight_balanced_error_curve.png")
    return {"direction_summary": direction_summary, "equal_weight_summary": equal_weight_summary, "direction_gap": direction_gap}

def run_ssv_pro_experiment_batch(experiments, *, output_root, experiment_order, n_repetitions, split_combination, normalization_methods, classifier_methods, seed, prevalence_modes=("natural", "class_stratified"), data_type_rnaseq=False, progress=True):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment_name in experiment_order:
        experiment = experiments[experiment_name]
        dataset = experiment["dataset"]
        output_name = experiment.get("output_name", experiment_name)
        composition, qc = compute_batch_class_qc(dataset)
        qc_dir = output_root / output_name / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        composition.to_csv(qc_dir / "batch_class_composition.csv", index=False)
        qc.to_csv(qc_dir / "batch_class_qc.csv", index=False)
        for prevalence_mode in prevalence_modes:
            if progress:
                print(f"{experiment_name} | {prevalence_mode}")
            raw = run_ssv_pro_bidirectional(dataset, split_combination=split_combination, n_repetitions=n_repetitions, normalization_methods=normalization_methods, classifier_methods=classifier_methods, seed=seed, prevalence_mode=prevalence_mode, data_type_rnaseq=data_type_rnaseq, progress=progress)
            protocol_dir = output_root / output_name / prevalence_mode
            save_ssv_pro_artifacts(raw, output_dir=protocol_dir)
            rows.append({"experiment": experiment_name, "output_name": output_name, "prevalence_mode": prevalence_mode, "status": "complete", "n_raw_rows": int(raw.shape[0]), "output_directory": str(protocol_dir)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_root / "ssv_pro_manifest.csv", index=False)
    return manifest

# -----------------------------------------------------------------------------
# Yokoi-specific helpers
# -----------------------------------------------------------------------------


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
