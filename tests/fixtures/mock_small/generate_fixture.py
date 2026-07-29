"""Generate deterministic mock artifacts for evaluation-module tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_ROOT = Path(__file__).resolve().parent
NUM_CLASSES = 6
SAMPLES_PER_CLASS = 5
SCORES = [0.95, 0.80, 0.65, 0.50, 0.35]
CLASS_COLORS = [
    (220, 70, 70),
    (70, 130, 220),
    (70, 180, 100),
    (220, 170, 60),
    (150, 90, 200),
    (60, 180, 180),
]


def paired_class(class_index: int) -> int:
    """Return the deliberately confusable partner for a class."""
    return class_index + 1 if class_index % 2 == 0 else class_index - 1


def other_classes(class_index: int) -> list[int]:
    """Return all incorrect classes, with the paired class ranked first."""
    pair = paired_class(class_index)
    cyclic = [
        (class_index + offset) % NUM_CLASSES
        for offset in range(1, NUM_CLASSES)
    ]
    return [pair] + [value for value in cyclic if value != pair]


def ranked_predictions(class_index: int, sample_index: int) -> list[int]:
    """Create two Top-1 hits, two Top-5-only hits, and one total miss."""
    others = other_classes(class_index)
    if sample_index in (0, 1):
        return [class_index, *others[:4]]
    if sample_index == 2:
        return [others[0], class_index, *others[1:4]]
    if sample_index == 3:
        return [*others[:3], class_index, others[3]]
    return others


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_image(path: Path, class_index: int, sample_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 96), CLASS_COLORS[class_index])
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 88, 88), outline="white", width=3)
    draw.text((25, 31), f"Class {class_index}", fill="white")
    draw.text((23, 51), f"Sample {sample_index}", fill="white")
    image.save(path)


def compute_expected_metrics(
    y_true: list[int],
    y_pred: list[int],
    top5_predictions: list[list[int]],
) -> dict:
    supports = [0] * NUM_CLASSES
    predicted_counts = [0] * NUM_CLASSES
    true_positives = [0] * NUM_CLASSES

    for truth, prediction in zip(y_true, y_pred, strict=True):
        supports[truth] += 1
        predicted_counts[prediction] += 1
        if truth == prediction:
            true_positives[truth] += 1

    per_class = []
    for class_index in range(NUM_CLASSES):
        precision = true_positives[class_index] / predicted_counts[class_index]
        recall = true_positives[class_index] / supports[class_index]
        f1 = 2 * precision * recall / (precision + recall)
        per_class.append(
            {
                "class_index": class_index,
                "precision": round(precision, 12),
                "recall": round(recall, 12),
                "f1": round(f1, 12),
                "support": supports[class_index],
            }
        )

    top1_accuracy = sum(
        truth == prediction
        for truth, prediction in zip(y_true, y_pred, strict=True)
    ) / len(y_true)
    top5_accuracy = sum(
        truth in predictions
        for truth, predictions in zip(y_true, top5_predictions, strict=True)
    ) / len(y_true)

    return {
        "num_classes": NUM_CLASSES,
        "num_samples": len(y_true),
        "top1_accuracy": round(top1_accuracy, 12),
        "top5_accuracy": round(top5_accuracy, 12),
        "overall_accuracy": round(top1_accuracy, 12),
        "macro_precision": round(
            sum(row["precision"] for row in per_class) / NUM_CLASSES, 12
        ),
        "macro_recall": round(
            sum(row["recall"] for row in per_class) / NUM_CLASSES, 12
        ),
        "macro_f1": round(
            sum(row["f1"] for row in per_class) / NUM_CLASSES, 12
        ),
        "balanced_accuracy": round(
            sum(row["recall"] for row in per_class) / NUM_CLASSES, 12
        ),
        "per_class": per_class,
    }


def generate_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    class_rows = [
        {
            "class_index": class_index,
            "category_id": 10000 + class_index,
            "species_name": f"Mock species {class_index}",
        }
        for class_index in range(NUM_CLASSES)
    ]
    write_csv(
        root / "class_mapping.csv",
        ["class_index", "category_id", "species_name"],
        class_rows,
    )

    test_rows = []
    prediction_rows = []
    y_true = []
    y_pred = []
    top5_predictions = []
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]

    for class_index in range(NUM_CLASSES):
        for sample_index in range(SAMPLES_PER_CLASS):
            sample_id = f"mock_c{class_index:03d}_s{sample_index:02d}"
            image_path = f"images/class_{class_index}/sample_{sample_index}.png"
            predictions = ranked_predictions(class_index, sample_index)

            write_image(
                root / image_path,
                class_index=class_index,
                sample_index=sample_index,
            )
            test_rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "true_label": class_index,
                }
            )

            row = {
                "sample_id": sample_id,
                "image_path": image_path,
                "true_label": class_index,
            }
            for rank, (prediction, score) in enumerate(
                zip(predictions, SCORES, strict=True), start=1
            ):
                row[f"pred_{rank}"] = prediction
                row[f"score_{rank}"] = score
            prediction_rows.append(row)
            y_true.append(class_index)
            y_pred.append(predictions[0])
            top5_predictions.append(predictions)
            confusion[class_index][predictions[0]] += 1

    write_csv(
        root / "test.csv",
        ["sample_id", "image_path", "true_label"],
        test_rows,
    )
    prediction_fields = ["sample_id", "image_path", "true_label"]
    prediction_fields.extend(f"pred_{rank}" for rank in range(1, 6))
    prediction_fields.extend(f"score_{rank}" for rank in range(1, 6))
    write_csv(root / "predictions.csv", prediction_fields, prediction_rows)

    metadata = {
        "method_name": "mock_classifier",
        "model_type": "synthetic_fixture",
        "initialization": "not_applicable",
        "num_classes": NUM_CLASSES,
        "random_seed": 42,
        "split_id": "mock6_seed42",
        "score_type": "synthetic_rank_score",
        "training_time_seconds": 12.5,
        "inference_time_seconds": 0.3,
        "inference_num_images": NUM_CLASSES * SAMPLES_PER_CLASS,
        "inference_includes_preprocessing": True,
        "device": "synthetic",
        "batch_size": 10,
        "image_size": 96,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    history_rows = [
        [1, 2.00, 1.95, 0.20, 0.18, 0.55, 0.15, 0.0010, 10.1],
        [2, 1.70, 1.65, 0.30, 0.28, 0.65, 0.24, 0.0010, 10.0],
        [3, 1.40, 1.40, 0.40, 0.38, 0.72, 0.34, 0.0010, 10.2],
        [4, 1.15, 1.22, 0.52, 0.45, 0.80, 0.41, 0.0005, 10.0],
        [5, 0.90, 1.15, 0.65, 0.50, 0.84, 0.47, 0.0005, 10.1],
        [6, 0.70, 1.18, 0.75, 0.52, 0.86, 0.49, 0.0005, 10.0],
        [7, 0.55, 1.28, 0.83, 0.50, 0.85, 0.46, 0.0001, 10.2],
        [8, 0.42, 1.42, 0.90, 0.48, 0.83, 0.44, 0.0001, 10.1],
    ]
    history_fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_top1",
        "val_top1",
        "val_top5",
        "val_macro_f1",
        "learning_rate",
        "epoch_seconds",
    ]
    write_csv(
        root / "history.csv",
        history_fields,
        [dict(zip(history_fields, row, strict=True)) for row in history_rows],
    )

    expected_metrics = compute_expected_metrics(
        y_true,
        y_pred,
        top5_predictions,
    )
    (root / "expected_metrics.json").write_text(
        json.dumps(expected_metrics, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        root / "expected_confusion_matrix.csv",
        ["true_label", *[f"pred_{index}" for index in range(NUM_CLASSES)]],
        [
            {"true_label": index, **{
                f"pred_{column}": value
                for column, value in enumerate(confusion[index])
            }}
            for index in range(NUM_CLASSES)
        ],
    )


def main() -> None:
    generate_fixture(DEFAULT_ROOT)


if __name__ == "__main__":
    main()
