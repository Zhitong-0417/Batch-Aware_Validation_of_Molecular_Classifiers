"""SSV workflow utilities."""



from .common import *



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
