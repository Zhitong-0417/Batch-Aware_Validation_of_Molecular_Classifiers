"""SSV workflow utilities."""



from .common import *
from .data import *
from .splitting import *
from .normalization import *
from .models import *



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
