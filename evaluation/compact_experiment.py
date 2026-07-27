"""Export one raw deep-learning run into a compact, auditable result folder."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from evaluation.plots import plot_confusion_matrix_subset
from evaluation.top5_evidence import write_top5_evidence
from utils.serialization import save_rows_csv


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_sanitized_training_state(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    state = _read_json(source)
    for key in ("last_checkpoint", "best_checkpoint"):
        if state.get(key):
            state[key] = Path(state[key]).name
    destination.write_bytes(
        (json.dumps(state, indent=2) + "\n").encode("utf-8")
    )


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_deep_run(
    raw_dir: str | Path,
    output_dir: str | Path,
    split_dir: str | Path,
    method_key: str,
    method_name: str,
    split_id: str,
    include_confusion_csv: bool = True,
) -> dict:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    split_dir = Path(split_dir)
    metrics = _read_json(raw_dir / "metrics.json")
    prediction_rows = _read_csv(raw_dir / "test_predictions.csv")
    labels = np.asarray(
        [int(row["true_class_index"]) for row in prediction_rows],
        dtype=np.int64,
    )
    predictions = np.asarray(
        [int(row["pred_class_index"]) for row in prediction_rows],
        dtype=np.int64,
    )
    num_classes = int(metrics["num_classes"])
    class_rows = _read_csv(split_dir / "selected_classes.csv")
    classes = {int(row["class_index"]): row for row in class_rows}
    test = metrics["test"]
    history = (
        _read_json(raw_dir / "history.json")
        if (raw_dir / "history.json").is_file()
        else []
    )
    checkpoint = Path(metrics.get("evaluated_checkpoint", raw_dir / "best_model.pt"))
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint

    summary = {
        "method_key": method_key,
        "method_name": method_name,
        "split_id": split_id,
        "initialization": metrics["initialization"],
        "device": metrics["device"],
        "random_seed": metrics["params"]["seed"],
        "num_classes": num_classes,
        "num_test_samples": len(labels),
        "top1_accuracy": float(test["top1_accuracy"]),
        "top5_accuracy": float(test["top5_accuracy"]),
        "overall_accuracy": float(test["overall_accuracy"]),
        "macro_precision": float(test["macro_precision"]),
        "macro_recall": float(test["macro_recall"]),
        "macro_f1": float(test["macro_f1"]),
        "balanced_accuracy": float(test["macro_recall"]),
        "test_loss": float(test["loss"]),
        "best_epoch": int(metrics["best_epoch"]),
        "completed_epochs": int(
            metrics.get("training_state", {}).get(
                "completed_epochs",
                len(history),
            )
        ),
        "training_time_seconds": float(metrics["training_seconds"]),
        "inference_time_seconds": float(test["seconds"]),
        "inference_images_per_second": float(test["images_per_second"]),
        "peak_gpu_memory_mb": max(
            (float(row.get("peak_gpu_memory_mb", 0.0)) for row in history),
            default=0.0,
        ),
        "checkpoint_sha256": (
            _checkpoint_sha256(checkpoint) if checkpoint.is_file() else ""
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(output_dir / "metrics.csv", [summary])

    configuration = [
        {"parameter": "method_key", "value": method_key},
        {"parameter": "method_name", "value": method_name},
        {"parameter": "model", "value": metrics["method"]},
        {"parameter": "initialization", "value": metrics["initialization"]},
        {"parameter": "split_id", "value": split_id},
    ]
    configuration.extend(
        {"parameter": key, "value": value}
        for key, value in metrics["params"].items()
        if key not in {"resume_checkpoint"}
    )
    save_rows_csv(
        output_dir / "configuration.csv",
        configuration,
        ["parameter", "value"],
    )

    if history:
        save_rows_csv(output_dir / "history.csv", history)
    save_rows_csv(
        output_dir / "test_predictions_top1.csv",
        prediction_rows,
        ["file_name", "true_class_index", "pred_class_index", "correct"],
    )
    top5_validation = write_top5_evidence(
        output_dir,
        prediction_rows,
        summary["top1_accuracy"],
        summary["top5_accuracy"],
        classes,
        metrics["data_root"],
    )

    class_indices = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_indices,
        zero_division=0,
    )
    recomputed_top1 = float(np.mean(labels == predictions))
    recomputed_macro_f1 = float(np.mean(f1))
    if not np.isclose(recomputed_top1, summary["top1_accuracy"], atol=1e-10):
        raise ValueError(
            f"{raw_dir}: Top-1 mismatch between predictions "
            f"({recomputed_top1}) and metrics ({summary['top1_accuracy']})"
        )
    if not np.isclose(recomputed_macro_f1, summary["macro_f1"], atol=1e-10):
        raise ValueError(
            f"{raw_dir}: Macro-F1 mismatch between predictions "
            f"({recomputed_macro_f1}) and metrics ({summary['macro_f1']})"
        )
    (output_dir / "prediction_validation.json").write_text(
        json.dumps(
            {
                "num_predictions": len(prediction_rows),
                "recomputed_top1_accuracy": recomputed_top1,
                "recorded_top1_accuracy": summary["top1_accuracy"],
                "recomputed_top5_accuracy": top5_validation[
                    "recomputed_top5_accuracy"
                ],
                "recorded_top5_accuracy": summary["top5_accuracy"],
                "top1_wrong_top5_correct_count": top5_validation[
                    "top1_wrong_top5_correct_count"
                ],
                "recomputed_macro_f1": recomputed_macro_f1,
                "recorded_macro_f1": summary["macro_f1"],
                "metrics_match_predictions": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    per_class_rows = []
    for class_index in class_indices:
        item = classes[class_index]
        per_class_rows.append(
            {
                "class_index": class_index,
                "category_id": item["category_id"],
                "species_name": item["name"],
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
            }
        )
    save_rows_csv(output_dir / "per_class_metrics.csv", per_class_rows)

    if include_confusion_csv:
        matrix = confusion_matrix(labels, predictions, labels=class_indices)
        matrix_rows = []
        for row_index, values in enumerate(matrix):
            row = {"true_class_index": row_index}
            row.update({str(index): int(value) for index, value in enumerate(values)})
            matrix_rows.append(row)
        save_rows_csv(
            output_dir / "confusion_matrix.csv",
            matrix_rows,
            ["true_class_index", *[str(index) for index in class_indices]],
        )
        plot_confusion_matrix_subset(
            output_dir / "test_confusion_matrix.png",
            matrix,
        )

    for name in [
        "training_curves.png",
        "learning_rate.png",
        "test_prediction_examples.png",
        "test_classification_report.json",
    ]:
        _copy_if_exists(raw_dir / name, output_dir / name)
    _copy_sanitized_training_state(
        raw_dir / "training_state.json",
        output_dir / "training_state.json",
    )
    return summary
