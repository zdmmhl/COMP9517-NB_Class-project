import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


METHODS = [
    ("hog_svm", "HOG + SVM", Path("outputs/hog_svm_full")),
    ("simple_cnn", "Simple CNN from scratch", Path("outputs/simple_cnn_full")),
    ("resnet18", "ImageNet-pretrained ResNet18", Path("outputs/resnet18_pretrained_full")),
    (
        "resnet50_optimized",
        "Optimized ImageNet-pretrained ResNet50",
        Path("outputs/resnet50_pretrained_optimized_full"),
    ),
    (
        "convnext_mixup",
        "ImageNet-pretrained ConvNeXt-Tiny with MixUp",
        Path("outputs/convnext_tiny_mixup_full"),
    ),
    (
        "deep_ensemble",
        "Validation-selected ResNet50/ConvNeXt ensemble",
        Path("outputs/deep_ensemble"),
    ),
]


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_classes(path):
    rows = read_csv(path)
    classes = {}
    for row in rows:
        class_index = int(row["class_index"])
        classes[class_index] = row
    return classes


def class_label(classes, class_index):
    item = classes[int(class_index)]
    common = item.get("common_name", "")
    name = item.get("name", "")
    if common and common != name:
        return f"{name} ({common})"
    return name


def short_label(classes, class_index):
    item = classes[int(class_index)]
    name = item.get("name", "")
    common = item.get("common_name", "")
    return common or name or str(class_index)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyse_method(method_key, method_name, result_dir, classes, output_dir):
    predictions = read_csv(result_dir / "test_predictions.csv")
    matrix = np.load(result_dir / "test_confusion_matrix.npy")
    metrics = json.load(open(result_dir / "metrics.json", encoding="utf-8"))

    per_class_total = Counter()
    per_class_correct = Counter()
    confusion_counter = Counter()
    examples_by_pair = defaultdict(list)
    correct_examples_by_class = {}

    for row in predictions:
        true_label = int(row["true_class_index"])
        pred_label = int(row["pred_class_index"])
        correct = int(row["correct"]) == 1
        per_class_total[true_label] += 1
        if correct:
            per_class_correct[true_label] += 1
            correct_examples_by_class.setdefault(true_label, row)
        else:
            confusion_counter[(true_label, pred_label)] += 1
            examples_by_pair[(true_label, pred_label)].append(row)

    top_confusions = []
    for (true_label, pred_label), count in confusion_counter.most_common(30):
        true_info = classes[true_label]
        pred_info = classes[pred_label]
        top_confusions.append(
            {
                "method": method_name,
                "true_class_index": true_label,
                "pred_class_index": pred_label,
                "count": count,
                "true_species": class_label(classes, true_label),
                "pred_species": class_label(classes, pred_label),
                "same_genus": int(true_info.get("genus") == pred_info.get("genus")),
                "same_family": int(true_info.get("family") == pred_info.get("family")),
                "true_family": true_info.get("family", ""),
                "pred_family": pred_info.get("family", ""),
                "example_file": examples_by_pair[(true_label, pred_label)][0]["file_name"],
            }
        )

    correct_examples = list(correct_examples_by_class.values())[:12]
    failure_examples = []
    used_true_classes = set()
    for row in top_confusions:
        pair_examples = examples_by_pair[(row["true_class_index"], row["pred_class_index"])]
        for example in pair_examples:
            true_label = int(example["true_class_index"])
            if true_label in used_true_classes:
                continue
            failure_examples.append(example)
            used_true_classes.add(true_label)
            break
        if len(failure_examples) >= 12:
            break

    worst_classes = []
    for class_index in sorted(per_class_total):
        total = per_class_total[class_index]
        correct = per_class_correct[class_index]
        accuracy = correct / total if total else 0.0
        most_common_wrong = [
            (pred, count)
            for (true, pred), count in confusion_counter.items()
            if true == class_index
        ]
        most_common_wrong.sort(key=lambda item: item[1], reverse=True)
        top_pred, top_count = most_common_wrong[0] if most_common_wrong else ("", 0)
        worst_classes.append(
            {
                "method": method_name,
                "class_index": class_index,
                "species": class_label(classes, class_index),
                "family": classes[class_index].get("family", ""),
                "genus": classes[class_index].get("genus", ""),
                "test_images": total,
                "correct": correct,
                "accuracy": round(accuracy, 4),
                "most_common_wrong_class": top_pred,
                "most_common_wrong_species": class_label(classes, top_pred)
                if top_pred != ""
                else "",
                "most_common_wrong_count": top_count,
            }
        )
    worst_classes.sort(key=lambda row: (row["accuracy"], -row["most_common_wrong_count"], row["class_index"]))

    method_dir = output_dir / method_key
    write_csv(
        method_dir / "top_confusions.csv",
        top_confusions,
        [
            "method",
            "true_class_index",
            "pred_class_index",
            "count",
            "true_species",
            "pred_species",
            "same_genus",
            "same_family",
            "true_family",
            "pred_family",
            "example_file",
        ],
    )
    write_csv(
        method_dir / "worst_classes.csv",
        worst_classes,
        [
            "method",
            "class_index",
            "species",
            "family",
            "genus",
            "test_images",
            "correct",
            "accuracy",
            "most_common_wrong_class",
            "most_common_wrong_species",
            "most_common_wrong_count",
        ],
    )

    plot_example_grid(
        method_dir / "correct_examples.png",
        correct_examples,
        classes,
        title=f"{method_name}: Correct Test Predictions",
    )
    plot_example_grid(
        method_dir / "failure_examples.png",
        failure_examples,
        classes,
        title=f"{method_name}: Failure Cases",
    )
    plot_confusion_bar(method_dir / "top_confusions.png", top_confusions, classes)

    return {
        "key": method_key,
        "name": method_name,
        "metrics": extract_test_metrics(metrics),
        "top_confusions": top_confusions[:10],
        "worst_classes": worst_classes[:10],
        "correct_examples": len(correct_examples),
        "failure_examples": len(failure_examples),
    }


def extract_test_metrics(metrics):
    if "splits" in metrics:
        test = metrics["splits"]["test"]
    else:
        test = metrics["test"]
    return {
        "top1": test["top1_accuracy"],
        "top5": test["top5_accuracy"],
        "macro_f1": test["macro_f1"],
    }


def load_image(path):
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def plot_example_grid(path, rows, classes, title):
    if not rows:
        return
    data_root = Path("datasets/inat2021")
    cols = 4
    rows_count = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3.4, rows_count * 3.6))
    axes = np.asarray(axes).reshape(-1)

    for ax, row in zip(axes, rows):
        image = load_image(data_root / row["file_name"])
        ax.imshow(image)
        true_label = int(row["true_class_index"])
        pred_label = int(row["pred_class_index"])
        status = "OK" if int(row["correct"]) == 1 else "WRONG"
        ax.set_title(
            f"{status}\nT: {short_label(classes, true_label)}\nP: {short_label(classes, pred_label)}",
            fontsize=8,
        )
        ax.axis("off")

    for ax in axes[len(rows) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_confusion_bar(path, top_confusions, classes):
    if not top_confusions:
        return
    rows = top_confusions[:12]
    labels = [
        f"{short_label(classes, row['true_class_index'])}\n-> {short_label(classes, row['pred_class_index'])}"
        for row in rows
    ]
    values = [row["count"] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(rows)), values)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Misclassified Test Images")
    ax.set_title("Most Frequent Confusions")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_markdown(path, summaries):
    lines = [
        "# Error Analysis Summary",
        "",
        "This document summarizes failure patterns from the held-out 500-class iNaturalist test split.",
        "",
        "## Quantitative Comparison",
        "",
        "| Method | Test Top-1 | Test Top-5 | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for summary in summaries:
        m = summary["metrics"]
        lines.append(
            f"| {summary['name']} | {m['top1']:.4f} | {m['top5']:.4f} | {m['macro_f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            "1. ImageNet-pretrained models are much stronger than both handcrafted HOG+SVM and the scratch CNN, especially on Top-5 accuracy and macro-F1.",
            "2. HOG+SVM and the scratch CNN both struggle with the 500-class fine-grained setting. Their low macro-F1 indicates weak per-species generalization.",
            "3. The optimized ResNet50 substantially improves on ResNet18 through higher input resolution, a stronger backbone, regularization, learning-rate scheduling, and test-time augmentation.",
            "4. ConvNeXt with MixUp further reduces overfitting, while a validation-selected probability ensemble provides the strongest final result.",
            "5. The pretrained models still make systematic mistakes, often between visually similar species or species with similar image context.",
            "",
        ]
    )

    for summary in summaries:
        lines.extend(
            [
                f"## {summary['name']}",
                "",
                "### Most Frequent Confusions",
                "",
                "| True Species | Predicted Species | Count | Same Family | Same Genus |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in summary["top_confusions"][:8]:
            lines.append(
                f"| {row['true_species']} | {row['pred_species']} | {row['count']} | {row['same_family']} | {row['same_genus']} |"
            )
        lines.extend(
            [
                "",
                "### Worst Classes By Test Accuracy",
                "",
                "| Species | Family | Accuracy | Most Common Wrong Prediction | Wrong Count |",
                "|---|---|---:|---|---:|",
            ]
        )
        for row in summary["worst_classes"][:8]:
            lines.append(
                f"| {row['species']} | {row['family']} | {row['accuracy']:.4f} | {row['most_common_wrong_species']} | {row['most_common_wrong_count']} |"
            )
        lines.extend(
            [
                "",
                "Suggested report interpretation:",
                "",
                "- Discuss whether the confused species share the same family or genus.",
                "- Inspect the generated failure images and describe whether the model appears to confuse shape, colour, background, scale, or pose.",
                "- Compare the handcrafted HOG failure pattern with the pretrained ResNet failure pattern.",
                "",
            ]
        )

    lines.extend(
        [
            "## Files Generated",
            "",
            "Each method directory contains:",
            "",
            "```text",
            "top_confusions.csv",
            "worst_classes.csv",
            "top_confusions.png",
            "correct_examples.png",
            "failure_examples.png",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report-ready error analysis artifacts.")
    parser.add_argument("--classes", type=Path, default=Path("data_splits/selected_classes.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/error_analysis"))
    return parser.parse_args()


def main():
    args = parse_args()
    classes = load_classes(args.classes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for method_key, method_name, result_dir in METHODS:
        print(f"Analysing {method_name}...")
        summaries.append(analyse_method(method_key, method_name, result_dir, classes, args.output_dir))

    write_markdown(args.output_dir / "error_analysis_summary.md", summaries)
    print(f"Saved error analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
