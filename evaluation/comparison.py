"""Create the report-ready comparison across completed methods."""

import json
from pathlib import Path

import pandas as pd


class ComparisonError(ValueError):
    """Raised when evaluated model results cannot be compared fairly."""


def load_evaluation_results(evaluation_root):
    """Load standardized per-model result directories."""
    evaluation_root = Path(evaluation_root)
    summary_rows = []
    per_class_frames = []

    directories = evaluation_root.iterdir() if evaluation_root.exists() else []
    for method_dir in sorted(directories):
        if not method_dir.is_dir():
            continue
        metrics_path = method_dir / "metrics.json"
        per_class_path = method_dir / "per_class_metrics.csv"
        if not metrics_path.is_file() or not per_class_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.setdefault("method_key", method_dir.name)
        summary_rows.append(metrics)
        per_class = pd.read_csv(per_class_path)
        per_class["method_name"] = metrics["method_name"]
        per_class["method_key"] = metrics["method_key"]
        per_class_frames.append(per_class)

    if not summary_rows:
        raise ComparisonError(
            f"No evaluated model directories found under {evaluation_root}"
        )
    summary = pd.DataFrame(summary_rows).sort_values("method_name").reset_index(
        drop=True
    )
    if summary["method_name"].duplicated().any():
        raise ComparisonError("Duplicate method_name values were found")
    if summary["split_id"].nunique() != 1:
        raise ComparisonError("Models use different split_id values")
    if summary["num_classes"].nunique() != 1:
        raise ComparisonError("Models use different numbers of classes")

    per_class = pd.concat(per_class_frames, ignore_index=True)
    return summary, per_class


def write_comparison_outputs(evaluation_root, output_dir):
    """Write generic comparison tables and figures."""
    from .plots import (
        plot_metric_comparison,
        plot_per_class_f1_distribution,
        plot_runtime_vs_performance,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, per_class = load_evaluation_results(evaluation_root)

    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    runtime_columns = [
        "method_name",
        "training_time_seconds",
        "inference_time_seconds",
        "inference_num_images",
        "inference_images_per_second",
        "device",
    ]
    summary[runtime_columns].to_csv(
        output_dir / "runtime_comparison.csv",
        index=False,
    )
    plot_metric_comparison(summary, output_dir / "model_comparison.png")
    plot_runtime_vs_performance(
        summary,
        output_dir / "runtime_vs_performance.png",
    )
    plot_per_class_f1_distribution(
        per_class,
        output_dir / "per_class_f1_distribution.png",
    )
    return summary, per_class
