"""Refresh primary comparison tables and figures from tracked method outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT

from evaluation.ablation import write_ablation_outputs
from evaluation.class_scaling import write_class_scaling_outputs
from evaluation.export_results import (
    load_exported_comparison_inputs,
    write_comparison_outputs,
    write_manifest,
)
from evaluation.plots import plot_confusion_matrix_subset
from utils.serialization import save_rows_csv


def refresh_deep_ablation_outputs(output_dir: Path) -> int:
    ablation_root = output_dir / "ablations" / "deep_learning"
    refreshed = 0
    for study_path in sorted(ablation_root.glob("*/study.json")):
        study_dir = study_path.parent
        write_ablation_outputs(study_dir, study_dir)
        runs_path = study_dir / "ablation_runs.csv"
        with runs_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        for row in rows:
            row["run_dir"] = (
                Path(row["run_dir"])
                .resolve()
                .relative_to(study_dir.resolve())
                .as_posix()
            )
        save_rows_csv(runs_path, rows, fieldnames)
        refreshed += 1
    return refreshed


def refresh_class_scaling_outputs(output_dir: Path) -> bool:
    config_path = PROJECT_ROOT / "configs" / "class_scaling.json"
    scaling_root = output_dir / "advanced" / "class_scaling"
    if not config_path.exists() or not scaling_root.exists():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    write_class_scaling_outputs(
        scaling_root / "runs",
        scaling_root / "reproducibility",
        scaling_root,
        config["split_specs"],
    )
    return True


def refresh_compact_confusion_figures(output_dir: Path) -> int:
    refreshed = 0
    for figure_path in output_dir.rglob("test_confusion_matrix.png"):
        prediction_path = figure_path.with_name("test_predictions_top1.csv")
        metrics_path = figure_path.with_name("metrics.csv")
        if not prediction_path.exists() or not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            metrics = list(csv.DictReader(file))
        if len(metrics) != 1:
            raise ValueError(f"Expected one metrics row in {metrics_path}")
        num_classes = int(metrics[0]["num_classes"])
        with prediction_path.open("r", encoding="utf-8", newline="") as file:
            predictions = list(csv.DictReader(file))
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for row in predictions:
            matrix[
                int(row["true_class_index"]),
                int(row["pred_class_index"]),
            ] += 1
        plot_confusion_matrix_subset(figure_path, matrix)
        refreshed += 1
    return refreshed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh final comparison CSVs and figures from report-ready "
            "per-method metrics without loading datasets or checkpoints."
        )
    )
    parser.add_argument(
        "--final-results-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "final_results" / "inat500",
    )
    args = parser.parse_args()
    output_dir = args.final_results_root.resolve()
    summaries, per_class = load_exported_comparison_inputs(output_dir)
    write_comparison_outputs(output_dir, summaries, per_class)
    ablation_studies = refresh_deep_ablation_outputs(output_dir)
    class_scaling = refresh_class_scaling_outputs(output_dir)
    confusion_figures = refresh_compact_confusion_figures(output_dir)
    write_manifest(output_dir)
    print(
        f"Refreshed {len(summaries)} methods, {ablation_studies} deep "
        f"ablation studies, {int(class_scaling)} class-scaling study, and "
        f"{confusion_figures} compact confusion figures: {output_dir}"
    )


if __name__ == "__main__":
    main()
