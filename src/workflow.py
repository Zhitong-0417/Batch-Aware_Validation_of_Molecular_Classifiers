"""SSV workflow utilities."""



from .common import *
from .data import *
from .evaluation import *
from .qc import *
from .artifacts import *



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
