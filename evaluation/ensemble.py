"""Validation-selected probability ensemble evaluation."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import SplitImageDataset
from data.splits import read_rows
from data.transforms import build_transforms
from evaluation.metrics import metrics_from_probabilities
from evaluation.plots import plot_confusion_matrix
from models.factory import build_model
from utils.reproducibility import seed_everything
from utils.serialization import save_json, save_predictions


MODEL_SPECS = [
    (
        "resnet50-pretrained",
        Path("results/resnet50_pretrained_optimized_full/best_model.pt"),
        "resnet50",
    ),
    (
        "convnext-tiny-pretrained",
        Path("results/convnext_tiny_mixup_full/best_model.pt"),
        "convnext",
    ),
]


@torch.inference_mode()
def predict_probabilities(model_name, checkpoint_path, loader, device, use_amp, tta):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, _ = build_model(model_name, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    probabilities = []
    started = time.perf_counter()
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            if tta:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())

    seconds = time.perf_counter() - started
    del model
    torch.cuda.empty_cache()
    return np.concatenate(probabilities), seconds


def read_labels(rows):
    return np.asarray([int(row["class_index"]) for row in rows], dtype=np.int64)


def write_weight_search(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-selected ensemble of two deep models.")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/inat2021"))
    parser.add_argument("--split-dir", type=Path, default=Path("data_splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/deep_ensemble"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_rows = read_rows(args.split_dir / "val.csv")
    test_rows = read_rows(args.split_dir / "test.csv")
    num_classes = len({int(row["class_index"]) for row in val_rows})
    if num_classes != 500:
        raise ValueError(f"Expected 500 classes, found {num_classes}")

    _, eval_transform = build_transforms(args.image_size, "basic")
    val_dataset = SplitImageDataset(val_rows, args.data_root, eval_transform)
    test_dataset = SplitImageDataset(test_rows, args.data_root, eval_transform)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
    }
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    use_tta = not args.no_tta
    split_probabilities = {"val": {}, "test": {}}
    inference_seconds = {}

    for model_name, checkpoint_path, key in MODEL_SPECS:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        print(f"Predicting with {model_name}...")
        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            probabilities, seconds = predict_probabilities(
                model_name,
                checkpoint_path,
                loader,
                device,
                use_amp,
                use_tta,
            )
            split_probabilities[split_name][key] = probabilities
            inference_seconds[f"{key}_{split_name}"] = seconds
            np.save(args.output_dir / f"{key}_{split_name}_probabilities.npy", probabilities)

    val_labels = read_labels(val_rows)
    test_labels = read_labels(test_rows)
    weight_rows = []
    best = None
    # Choose weights on validation, then apply the winner once to test.
    for convnext_weight in np.linspace(0.0, 1.0, 11):
        probabilities = (
            (1.0 - convnext_weight) * split_probabilities["val"]["resnet50"]
            + convnext_weight * split_probabilities["val"]["convnext"]
        )
        metrics = metrics_from_probabilities(val_labels, probabilities)
        row = {
            "convnext_weight": round(float(convnext_weight), 2),
            "resnet50_weight": round(float(1.0 - convnext_weight), 2),
            **metrics,
        }
        weight_rows.append(row)
        score = (metrics["macro_f1"], metrics["top1_accuracy"])
        if best is None or score > best[0]:
            best = (score, float(convnext_weight), metrics)

    _, convnext_weight, val_metrics = best
    resnet50_weight = 1.0 - convnext_weight
    test_probabilities = (
        resnet50_weight * split_probabilities["test"]["resnet50"]
        + convnext_weight * split_probabilities["test"]["convnext"]
    )
    test_metrics = metrics_from_probabilities(test_labels, test_probabilities)
    test_predictions = test_probabilities.argmax(axis=1)

    write_weight_search(args.output_dir / "validation_weight_search.csv", weight_rows)
    save_predictions(
        args.output_dir / "test_predictions.csv",
        [row["file_name"] for row in test_rows],
        test_labels.tolist(),
        test_predictions.tolist(),
    )
    plot_confusion_matrix(
        args.output_dir / "test_confusion_matrix.png",
        test_labels.tolist(),
        test_predictions.tolist(),
        num_classes,
    )
    result = {
        "method": "validation-selected probability ensemble",
        "models": [item[0] for item in MODEL_SPECS],
        "weights": {
            "resnet50": resnet50_weight,
            "convnext": convnext_weight,
        },
        "selection_metric": "validation macro_f1",
        "validation": val_metrics,
        "test": test_metrics,
        "inference_seconds": inference_seconds,
        "tta": use_tta,
        "seed": args.seed,
        "rows": {"validation": len(val_rows), "test": len(test_rows)},
    }
    save_json(args.output_dir / "metrics.json", result)
    print(
        f"Selected weights: ResNet50={resnet50_weight:.1f}, ConvNeXt={convnext_weight:.1f}"
    )
    print(
        f"Ensemble test: top1={test_metrics['top1_accuracy']:.4f}, "
        f"top5={test_metrics['top5_accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
