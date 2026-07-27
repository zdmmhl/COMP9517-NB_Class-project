"""Validate and visualize the Advanced class-scaling experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("top1_accuracy", "Top-1 accuracy"),
    ("top5_accuracy", "Top-5 accuracy"),
    ("macro_f1", "Macro-F1"),
]


def load_single_row(path: Path) -> dict:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"{path} must contain exactly one result row")
    return frame.iloc[0].to_dict()


def validate_training_controls(
    configurations: dict[str, dict[str, str]],
) -> list[str]:
    ignored = {
        "method_key",
        "method_name",
        "split_id",
        "max_classes",
        "max_train_per_class",
        "max_eval_per_class",
    }
    baseline_key = "classes_500"
    baseline = configurations[baseline_key]
    errors = []
    for run_key, candidate in configurations.items():
        changed = {
            key
            for key in set(baseline) | set(candidate)
            if baseline.get(key, "") != candidate.get(key, "")
            and key not in ignored
        }
        if changed:
            errors.append(f"{run_key}: uncontrolled parameters {sorted(changed)}")
    if errors:
        raise ValueError("; ".join(errors))
    return sorted(set(baseline) - ignored)


def plot_main_scaling(summary: pd.DataFrame, output_path: Path) -> None:
    main = summary[summary["kind"] == "main"].sort_values("num_classes")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for key, label in METRICS:
        axes[0].plot(
            main["num_classes"],
            main[key],
            marker="o",
            linewidth=2,
            label=label,
        )
    axes[0].set_xlabel("Number of classes")
    axes[0].set_ylabel("Test score")
    axes[0].set_title("Performance as Recognition Scales Up")
    axes[0].set_xticks(main["num_classes"])
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        main["num_classes"],
        main["train_per_class"],
        marker="o",
        color="#b54832",
        linewidth=2,
    )
    axes[1].set_xlabel("Number of classes")
    axes[1].set_ylabel("Training images per class")
    axes[1].set_title("Fixed Total Training Budget")
    axes[1].set_xticks(main["num_classes"])
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sample_controls(summary: pd.DataFrame, output_path: Path) -> None:
    controls = summary[
        summary["run_key"].isin(
            ["classes_500", "control_500x20", "control_500x8"]
        )
    ].sort_values("train_per_class")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for key, label in METRICS:
        ax.plot(
            controls["train_per_class"],
            controls[key],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.set_xlabel("Training images per class (500 classes fixed)")
    ax.set_ylabel("Test score")
    ax.set_title("Effect of Per-Class Sample Size")
    ax.set_xticks(controls["train_per_class"])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values(["kind", "num_classes", "train_per_class"])
    labels = ordered["run_key"].tolist()
    values = ordered["training_time_seconds"].astype(float).to_numpy() / 60
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(labels, values, color="#386b8c")
    ax.bar_label(bars, fmt="%.1f min", padding=3, fontsize=8)
    ax.set_ylabel("Recorded training time (minutes)")
    ax.set_title("Class-Scaling Experiment Runtime")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_class_scaling_outputs(
    run_root: str | Path,
    split_root: str | Path,
    output_dir: str | Path,
    split_specs: dict,
) -> pd.DataFrame:
    run_root = Path(run_root)
    split_root = Path(split_root)
    output_dir = Path(output_dir)
    rows = []
    configurations = {}
    validation = json.loads(
        (split_root / "validation_report.json").read_text(encoding="utf-8")
    )
    if not validation.get("valid"):
        raise ValueError("Scaling split validation report is not valid")

    for run_key, spec in split_specs.items():
        run_dir = run_root / run_key
        metric = load_single_row(run_dir / "metrics.csv")
        configuration_frame = pd.read_csv(
            run_dir / "configuration.csv",
            dtype=str,
            keep_default_na=False,
        )
        configurations[run_key] = dict(
            zip(configuration_frame["parameter"], configuration_frame["value"])
        )
        split_summary = json.loads(
            (split_root / run_key / "split_summary.json").read_text(
                encoding="utf-8"
            )
        )
        prediction_check = json.loads(
            (run_dir / "prediction_validation.json").read_text(encoding="utf-8")
        )
        if not prediction_check.get("metrics_match_predictions"):
            raise ValueError(f"{run_key}: prediction metrics did not validate")
        rows.append(
            {
                "run_key": run_key,
                "kind": spec["kind"],
                "num_classes": int(spec["num_classes"]),
                "train_per_class": int(spec["train_per_class"]),
                "val_per_class": int(spec["val_per_class"]),
                "test_per_class": int(spec["test_per_class"]),
                "total_train_images": int(split_summary["rows"]["train"]),
                "total_val_images": int(split_summary["rows"]["val"]),
                "total_test_images": int(split_summary["rows"]["test"]),
                "best_epoch": metric["best_epoch"],
                "completed_epochs": metric["completed_epochs"],
                "top1_accuracy": metric["top1_accuracy"],
                "top5_accuracy": metric["top5_accuracy"],
                "macro_f1": metric["macro_f1"],
                "training_time_seconds": metric["training_time_seconds"],
                "inference_time_seconds": metric["inference_time_seconds"],
                "peak_gpu_memory_mb": metric["peak_gpu_memory_mb"],
            }
        )

    checked_parameters = validate_training_controls(configurations)
    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        output_dir / "scaling_summary.csv",
        index=False,
        lineterminator="\n",
    )
    plot_main_scaling(summary, output_dir / "class_scaling_metrics.png")
    plot_sample_controls(summary, output_dir / "sample_size_control.png")
    plot_runtime(summary, output_dir / "runtime_comparison.png")
    (output_dir / "validation_report.json").write_text(
        json.dumps(
            {
                "valid": True,
                "nested_classes": {
                    "base_500_in_1000": validation["base_500_nested_in_1000"],
                    "classes_1000_in_2500": validation[
                        "classes_1000_nested_in_2500"
                    ],
                },
                "nested_samples": validation["sample_nesting"],
                "prediction_metrics_recomputed": True,
                "controlled_training_parameters": checked_parameters,
                "num_runs": len(summary),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary
