"""SSV workflow utilities."""



from .common import *
from .evaluation import *
from .plotting import *



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
