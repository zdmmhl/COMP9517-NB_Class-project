"""Re-evaluate saved checkpoints and add auditable per-image Top-5 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import SplitImageDataset
from data.splits import read_rows
from data.transforms import build_transforms
from evaluation.top5_evidence import read_prediction_rows, write_top5_evidence
from evaluation.plots import plot_confusion_matrix
from models.factory import build_model
from training.engine import evaluate
from utils.serialization import save_json, save_predictions


BASE_RUNS = [
    "simple_cnn_converged_full",
    "resnet18_pretrained_converged_full",
    "resnet50_pretrained_optimized_full",
    "convnext_tiny_mixup_full",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional target keys. By default all main, ablation, scaling and ensemble runs are processed.",
    )
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_classes(split_dir: Path) -> dict[int, dict[str, str]]:
    return {
        int(row["class_index"]): row
        for row in read_rows(split_dir / "selected_classes.csv")
    }


def checkpoint_path(raw_dir: Path, metrics: dict) -> Path:
    candidates = [
        metrics.get("evaluated_checkpoint"),
        metrics.get("training_state", {}).get("best_checkpoint"),
        raw_dir / "best_model.pt",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            return path
    raise FileNotFoundError(f"No checkpoint found for {raw_dir}")


def refresh_top1_artifacts(raw_dir: Path, num_classes: int) -> None:
    rows = read_prediction_rows(raw_dir / "test_predictions.csv")
    labels = [int(row["true_class_index"]) for row in rows]
    predictions = [int(row["pred_class_index"]) for row in rows]
    save_json(
        raw_dir / "test_classification_report.json",
        classification_report(
            labels,
            predictions,
            labels=list(range(num_classes)),
            zero_division=0,
            output_dict=True,
        ),
    )
    plot_confusion_matrix(
        raw_dir / "test_confusion_matrix.png",
        labels,
        predictions,
        num_classes,
    )


def backfill_checkpoint(
    key: str,
    raw_dir: Path,
    split_dir: Path,
    device: torch.device,
    batch_size_override: int,
) -> dict:
    metrics = load_json(raw_dir / "metrics.json")
    checkpoint = torch.load(
        checkpoint_path(raw_dir, metrics),
        map_location=device,
        weights_only=False,
    )
    num_classes = int(metrics["num_classes"])
    model, _ = build_model(metrics["method"], num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    channels_last = bool(metrics.get("params", {}).get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    params = metrics.get("params", {})
    image_size = int(params.get("image_size", 224))
    _, transform = build_transforms(
        image_size,
        params.get("augmentation", "basic"),
    )
    rows = read_rows(split_dir / "test.csv")
    dataset = SplitImageDataset(rows, metrics["data_root"], transform)
    batch_size = batch_size_override or int(
        params.get("eval_batch_size") or params.get("batch_size", 64)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(params.get("label_smoothing", 0.0))
    )
    evaluated = evaluate(
        model,
        loader,
        criterion,
        device,
        num_classes,
        use_amp=device.type == "cuda",
        tta=bool(params.get("tta", False)),
        channels_last=channels_last,
    )
    previous_top1 = float(metrics["test"]["top1_accuracy"])
    previous_top5 = float(metrics["test"]["top5_accuracy"])
    existing_evidence = metrics.get("top5_evidence", {})
    original_top1 = float(
        existing_evidence.get(
            "original_recorded_top1_accuracy",
            existing_evidence.get("previous_recorded_top1_accuracy", previous_top1),
        )
    )
    original_top5 = float(
        existing_evidence.get(
            "original_recorded_top5_accuracy",
            existing_evidence.get("previous_recorded_top5_accuracy", previous_top5),
        )
    )
    maximum_count_difference = 2
    maximum_accuracy_difference = maximum_count_difference / len(rows) + 1e-9
    for name, previous, refreshed in [
        ("Top-1", previous_top1, evaluated["top1_accuracy"]),
        ("Top-5", previous_top5, evaluated["top5_accuracy"]),
    ]:
        if abs(previous - refreshed) > maximum_accuracy_difference:
            raise ValueError(
                f"{key}: refreshed {name} differs from the historical summary by "
                f"more than {maximum_count_difference} images "
                f"({previous} vs {refreshed})"
            )
    save_predictions(
        raw_dir / "test_predictions.csv",
        evaluated["paths"],
        evaluated["labels"],
        evaluated["preds"],
        evaluated["top5_preds"],
        evaluated["top5_scores"],
    )
    refresh_top1_artifacts(raw_dir, num_classes)
    class_rows = load_classes(split_dir)
    validation = write_top5_evidence(
        raw_dir,
        read_prediction_rows(raw_dir / "test_predictions.csv"),
        evaluated["top1_accuracy"],
        evaluated["top5_accuracy"],
        class_rows,
        metrics["data_root"],
    )
    metrics["test"].update(
        {
            key: evaluated[key]
            for key in [
                "loss",
                "top1_accuracy",
                "overall_accuracy",
                "top5_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "seconds",
                "images_per_second",
            ]
        }
    )
    metrics["top5_evidence"] = {
        "predictions_file": "test_predictions_top5.csv",
        "validation_file": "top5_prediction_validation.json",
        "rescued_cases_file": "top1_wrong_top5_correct.csv",
        "rescued_cases_figure": "top1_wrong_top5_correct.png",
        "original_recorded_top1_accuracy": original_top1,
        "original_recorded_top5_accuracy": original_top5,
        "previous_recorded_top1_accuracy": previous_top1,
        "previous_recorded_top5_accuracy": previous_top5,
        "top1_refresh_delta": evaluated["top1_accuracy"] - previous_top1,
        "top5_refresh_delta": evaluated["top5_accuracy"] - previous_top5,
        **validation,
    }
    save_json(raw_dir / "metrics.json", metrics)
    print(
        f"{key}: Top-1={validation['recomputed_top1_accuracy']:.4f}, "
        f"Top-5={validation['recomputed_top5_accuracy']:.4f}, "
        f"rescued={validation['top1_wrong_top5_correct_count']}"
    )
    return validation


def backfill_ensemble(raw_dir: Path, split_dir: Path) -> dict:
    metrics = load_json(raw_dir / "metrics.json")
    weights = metrics["weights"]
    probabilities = (
        float(weights["resnet50"])
        * np.load(raw_dir / "resnet50_test_probabilities.npy")
        + float(weights["convnext"])
        * np.load(raw_dir / "convnext_test_probabilities.npy")
    )
    rows = read_rows(split_dir / "test.csv")
    labels = np.asarray([int(row["class_index"]) for row in rows])
    top5_indices = np.argsort(probabilities, axis=1)[:, -5:][:, ::-1]
    top5_scores = np.take_along_axis(probabilities, top5_indices, axis=1)
    save_predictions(
        raw_dir / "test_predictions.csv",
        [row["file_name"] for row in rows],
        labels.tolist(),
        top5_indices[:, 0].tolist(),
        top5_indices.tolist(),
        top5_scores.tolist(),
    )
    refresh_top1_artifacts(raw_dir, probabilities.shape[1])
    validation = write_top5_evidence(
        raw_dir,
        read_prediction_rows(raw_dir / "test_predictions.csv"),
        metrics["test"]["top1_accuracy"],
        metrics["test"]["top5_accuracy"],
        load_classes(split_dir),
        load_json(
            PROJECT_ROOT / "results" / "resnet50_pretrained_optimized_full" / "metrics.json"
        )["data_root"],
    )
    metrics["top5_evidence"] = {
        "predictions_file": "test_predictions_top5.csv",
        "validation_file": "top5_prediction_validation.json",
        "rescued_cases_file": "top1_wrong_top5_correct.csv",
        "rescued_cases_figure": "top1_wrong_top5_correct.png",
        **validation,
    }
    save_json(raw_dir / "metrics.json", metrics)
    print(
        f"deep_ensemble: Top-1={validation['recomputed_top1_accuracy']:.4f}, "
        f"Top-5={validation['recomputed_top5_accuracy']:.4f}, "
        f"rescued={validation['top1_wrong_top5_correct_count']}"
    )
    return validation


def targets() -> list[tuple[str, Path, Path]]:
    items = [
        (key, PROJECT_ROOT / "results" / key, PROJECT_ROOT / "data_splits")
        for key in BASE_RUNS
    ]
    deep_config = load_json(PROJECT_ROOT / "configs" / "deep_ablations.json")
    for key in [
        *deep_config["training_runs"],
        *deep_config.get("evaluation_runs", {}),
    ]:
        items.append(
            (
                f"ablation:{key}",
                PROJECT_ROOT / deep_config["results_root"] / key,
                PROJECT_ROOT / deep_config["split_dir"],
            )
        )
    scaling_config = load_json(PROJECT_ROOT / "configs" / "class_scaling.json")
    for key in scaling_config["split_specs"]:
        if key == "classes_500":
            continue
        items.append(
            (
                f"scaling:{key}",
                PROJECT_ROOT / scaling_config["results_root"] / key,
                PROJECT_ROOT / scaling_config["split_root"] / key,
            )
        )
    return items


def main() -> None:
    args = parse_args()
    selected = set(args.only or [])
    device = torch.device(args.device)
    for key, raw_dir, split_dir in targets():
        if selected and key not in selected:
            continue
        backfill_checkpoint(key, raw_dir, split_dir, device, args.batch_size)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not selected or "deep_ensemble" in selected:
        backfill_ensemble(
            PROJECT_ROOT / "results" / "deep_ensemble",
            PROJECT_ROOT / "data_splits",
        )


if __name__ == "__main__":
    main()
