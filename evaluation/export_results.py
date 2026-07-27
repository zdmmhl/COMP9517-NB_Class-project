"""Export completed experiments into compact, report-ready shared artifacts."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from evaluation.top5_evidence import write_top5_evidence
from utils.serialization import save_rows_csv


METHODS = [
    {
        "key": "hog_svm",
        "name": "Legacy HOG + SGD linear SVM (20 iterations)",
        "result_dir": "hog_svm_full",
        "analysis_dir": "hog_svm",
        "initialization": "handcrafted",
    },
    {
        "key": "color_sgd_svm",
        "name": "HSV colour histogram + SGD linear SVM",
        "result_dir": "traditional_color_sgd_svm_full",
        "analysis_dir": "",
        "initialization": "handcrafted",
    },
    {
        "key": "lbp_sgd_svm",
        "name": "Uniform LBP + SGD linear SVM",
        "result_dir": "traditional_lbp_sgd_svm_full",
        "analysis_dir": "",
        "initialization": "handcrafted",
    },
    {
        "key": "hog_sgd_svm",
        "name": "HOG + SGD linear SVM",
        "result_dir": "traditional_hog_sgd_svm_full",
        "analysis_dir": "",
        "initialization": "handcrafted",
    },
    {
        "key": "sift_bovw_sgd_svm",
        "name": "SIFT Bag-of-Visual-Words + SGD linear SVM",
        "result_dir": "traditional_sift_bovw_sgd_svm_full",
        "analysis_dir": "",
        "initialization": "handcrafted",
    },
    {
        "key": "color_linear_svc",
        "name": "HSV colour histogram + LinearSVC",
        "result_dir": "traditional_color_linear_svc_full",
        "analysis_dir": "",
        "initialization": "handcrafted",
    },
    {
        "key": "hog_random_forest",
        "name": "HOG + Random Forest (300 trees)",
        "result_dir": "traditional_hog_random_forest_full",
        "analysis_dir": "",
        "initialization": "handcrafted",
    },
    {
        "key": "simple_cnn",
        "name": "SimpleCNN from scratch (50-epoch convergence run)",
        "result_dir": "simple_cnn_converged_full",
        "analysis_dir": "",
        "initialization": "random",
    },
    {
        "key": "resnet18_pretrained",
        "name": "ImageNet-pretrained ResNet18 (30-epoch convergence run)",
        "result_dir": "resnet18_pretrained_converged_full",
        "analysis_dir": "",
        "initialization": "imagenet",
    },
    {
        "key": "resnet50_optimized",
        "name": "Optimized ImageNet-pretrained ResNet50",
        "result_dir": "resnet50_pretrained_optimized_full",
        "analysis_dir": "resnet50_optimized",
        "initialization": "imagenet",
    },
    {
        "key": "convnext_mixup",
        "name": "ImageNet-pretrained ConvNeXt-Tiny + MixUp",
        "result_dir": "convnext_tiny_mixup_full",
        "analysis_dir": "convnext_mixup",
        "initialization": "imagenet",
    },
    {
        "key": "deep_ensemble",
        "name": "Validation-selected ResNet50/ConvNeXt ensemble",
        "result_dir": "deep_ensemble",
        "analysis_dir": "deep_ensemble",
        "initialization": "ensemble",
    },
]

INTEGER_SUMMARY_FIELDS = {
    "random_seed",
    "num_classes",
    "num_test_samples",
    "best_epoch",
}
FLOAT_SUMMARY_FIELDS = {
    "top1_accuracy",
    "top5_accuracy",
    "overall_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "balanced_accuracy",
    "test_loss",
    "training_time_seconds",
    "feature_extraction_time_seconds",
    "vocabulary_time_seconds",
    "inference_time_seconds",
    "inference_images_per_second",
    "recorded_total_run_seconds",
}
RUNTIME_FIELDS = [
    "method_key",
    "method_name",
    "device",
    "training_time_seconds",
    "inference_time_seconds",
    "num_test_samples",
    "inference_images_per_second",
    "timing_note",
]
CANONICAL_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def coerce_summary_types(row):
    row = dict(row)
    for key in INTEGER_SUMMARY_FIELDS:
        if key in row and row[key] != "":
            row[key] = int(row[key])
    for key in FLOAT_SUMMARY_FIELDS:
        if key in row and row[key] != "":
            row[key] = float(row[key])
    return row


def load_exported_comparison_inputs(output_dir):
    """Load the latest report-ready metrics without requiring raw runs."""
    summaries = []
    per_class_by_method = {}
    for method in METHODS:
        method_dir = output_dir / "methods" / method["key"]
        metric_rows = read_csv(method_dir / "metrics.csv")
        if len(metric_rows) != 1:
            raise ValueError(
                f"Expected one metrics row for {method['key']}, got {len(metric_rows)}"
            )
        summary = coerce_summary_types(metric_rows[0])
        if summary["method_key"] != method["key"]:
            raise ValueError(
                f"Method key mismatch in {method_dir / 'metrics.csv'}"
            )
        per_class_rows = read_csv(method_dir / "per_class_metrics.csv")
        if len(per_class_rows) != int(summary["num_classes"]):
            raise ValueError(
                f"Expected {summary['num_classes']} per-class rows for "
                f"{method['key']}, got {len(per_class_rows)}"
            )
        for row in per_class_rows:
            row["f1"] = float(row["f1"])
        summaries.append(summary)
        per_class_by_method[method["key"]] = per_class_rows
    return summaries, per_class_by_method


def load_classes(path):
    return {int(row["class_index"]): row for row in read_csv(path)}


def load_predictions(path):
    rows = read_csv(path)
    labels = np.asarray([int(row["true_class_index"]) for row in rows], dtype=np.int64)
    predictions = np.asarray(
        [int(row["pred_class_index"]) for row in rows], dtype=np.int64
    )
    return rows, labels, predictions


def test_metrics(metrics):
    if "splits" in metrics:
        return metrics["splits"]["test"]
    return metrics["test"]


def history_training_seconds(history):
    values = [row.get("train_seconds") for row in history]
    if history and all(value is not None for value in values):
        return float(sum(values))
    return ""


def inference_seconds(metrics):
    test = test_metrics(metrics)
    if "inference_seconds" in test:
        return float(test["inference_seconds"])
    if "seconds" in test:
        return float(test["seconds"])
    values = metrics.get("inference_seconds", {})
    test_values = [
        float(value) for key, value in values.items() if str(key).endswith("_test")
    ]
    return float(sum(test_values)) if test_values else ""


def build_summary(method, metrics, history):
    test = test_metrics(metrics)
    top1 = float(test["top1_accuracy"])
    training_seconds = (
        float(metrics["train_seconds"])
        if "train_seconds" in metrics
        else float(metrics["training_seconds"])
        if "training_seconds" in metrics
        else history_training_seconds(history)
    )
    test_rows = metrics.get("rows", {}).get("test")
    if test_rows is None:
        test_rows = test.get("rows") or metrics.get("rows", {}).get("test", 5000)
    infer_seconds = inference_seconds(metrics)
    images_per_second = (
        float(test_rows) / float(infer_seconds) if infer_seconds not in {"", 0} else ""
    )
    timing_note = "Recorded directly from the experiment."
    if method["key"] == "hog_svm":
        timing_note = (
            "Legacy result: classifier fit only; HOG extraction was not recorded. "
            "The estimator was SGDClassifier with hinge loss, not LinearSVC."
        )
    elif method["initialization"] == "handcrafted":
        timing_note = (
            "Classifier fit time is separate from first-pass feature extraction; "
            "see feature_extraction_time_seconds."
        )
    elif method["key"] == "resnet50_optimized":
        timing_note = "Original per-epoch training time was not recorded; left blank."
    elif method["key"] == "deep_ensemble":
        timing_note = "No training time; inference sums both component model test passes."
    elif method["result_dir"].endswith("_converged_full"):
        timing_note = (
            "Training time sums the recorded training phase of every epoch; "
            "validation and final evaluation are reported separately."
        )

    feature_extraction = metrics.get("feature_extraction", {})
    feature_extraction_seconds = sum(
        float(item.get("seconds", 0.0)) for item in feature_extraction.values()
    )
    vocabulary = metrics.get("vocabulary", {})
    vocabulary_seconds = sum(
        float(vocabulary.get(key, 0.0))
        for key in ["descriptor_extraction_seconds", "fit_seconds"]
    )

    return {
        "method_key": method["key"],
        "method_name": method["name"],
        "split_id": "inat500_seed9517",
        "initialization": method["initialization"],
        "device": metrics.get("device", ""),
        "random_seed": metrics.get("seed")
        or metrics.get("params", {}).get("seed")
        or 9517,
        "num_classes": metrics.get("num_classes", 500),
        "num_test_samples": test_rows,
        "top1_accuracy": top1,
        "top5_accuracy": float(test["top5_accuracy"]),
        "overall_accuracy": top1,
        "macro_precision": float(test["macro_precision"]),
        "macro_recall": float(test["macro_recall"]),
        "macro_f1": float(test["macro_f1"]),
        "balanced_accuracy": float(test["macro_recall"]),
        "test_loss": test.get("loss", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "training_time_seconds": training_seconds,
        "feature_extraction_time_seconds": (
            feature_extraction_seconds if feature_extraction else ""
        ),
        "vocabulary_time_seconds": vocabulary_seconds if vocabulary else "",
        "inference_time_seconds": infer_seconds,
        "inference_images_per_second": images_per_second,
        "recorded_total_run_seconds": (
            metrics.get("total_seconds", "")
            if metrics.get("run_mode") != "evaluation"
            else ""
        ),
        "timing_note": timing_note,
    }


def write_history(path, history):
    if not history:
        return
    preferred = [
        "epoch",
        "head_lr",
        "backbone_lr",
        "train_loss",
        "train_top1_accuracy",
        "train_seconds",
        "val_loss",
        "val_top1_accuracy",
        "val_top5_accuracy",
        "val_macro_f1",
        "val_seconds",
    ]
    extras = [key for row in history for key in row if key not in preferred]
    fields = preferred + list(dict.fromkeys(extras))
    save_rows_csv(path, history, fields)


def write_configuration(path, method, metrics):
    params = metrics.get("params", {})
    rows = [
        {"parameter": "method_key", "value": method["key"]},
        {"parameter": "method_name", "value": method["name"]},
        {"parameter": "initialization", "value": method["initialization"]},
    ]
    for key, value in params.items():
        rows.append(
            {
                "parameter": key,
                "value": json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value,
            }
        )
    save_rows_csv(path, rows, ["parameter", "value"])


def write_per_class(path, labels, predictions, classes, num_classes):
    class_indices = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_indices,
        zero_division=0,
    )
    rows = []
    for index in class_indices:
        item = classes.get(index, {})
        rows.append(
            {
                "class_index": index,
                "category_id": item.get("category_id", ""),
                "species_name": item.get("name", ""),
                "common_name": item.get("common_name", ""),
                "family": item.get("family", ""),
                "genus": item.get("genus", ""),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "class_accuracy": float(recall[index]),
            }
        )
    save_rows_csv(path, rows)
    return rows


def write_confusion_csv(path, matrix):
    fieldnames = ["true_class_index"] + [str(index) for index in range(matrix.shape[1])]
    rows = []
    for index, values in enumerate(matrix):
        row = {"true_class_index": index}
        row.update({str(column): int(value) for column, value in enumerate(values)})
        rows.append(row)
    save_rows_csv(path, rows, fieldnames)


def plot_confusions(full_path, subset_path, matrix, per_class_rows):
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=totals != 0,
    )
    fig, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axis.set_title(f"Normalized Confusion Matrix ({matrix.shape[0]} Classes)")
    axis.set_xlabel("Predicted class index")
    axis.set_ylabel("True class index")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(full_path, dpi=180)
    plt.close(fig)

    hardest = sorted(per_class_rows, key=lambda row: (row["f1"], row["class_index"]))[:25]
    indices = [int(row["class_index"]) for row in hardest]
    subset = normalized[np.ix_(indices, indices)]
    fig, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(subset, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(indices)))
    axis.set_yticks(range(len(indices)))
    axis.set_xticklabels(indices, rotation=90, fontsize=7)
    axis.set_yticklabels(indices, fontsize=7)
    axis.set_title("Confusion Matrix: 25 Lowest-F1 Classes")
    axis.set_xlabel("Predicted class index")
    axis.set_ylabel("True class index")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(subset_path, dpi=180)
    plt.close(fig)


def copy_if_exists(source, destination):
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def resize_grid(source, destination, max_width=1600):
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        image.save(destination, "JPEG", quality=84, optimize=True, progressive=True)


def plot_model_comparison(path, summaries):
    labels = [row["method_name"] for row in summaries]
    x = np.arange(len(labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(14, 6))
    for offset, key, name in [
        (-width, "top1_accuracy", "Top-1"),
        (0, "top5_accuracy", "Top-5"),
        (width, "macro_f1", "Macro-F1"),
    ]:
        bars = axis.bar(x + offset, [row[key] for row in summaries], width, label=name)
        axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title("500-Class iNaturalist Test Performance")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_runtime(path, summaries):
    rows = [row for row in summaries if row["inference_time_seconds"] != ""]
    traditional_handles = []
    traditional_labels = []
    fig, axis = plt.subplots(figsize=(9, 6))
    for row in rows:
        point = axis.scatter(
            row["inference_time_seconds"],
            row["top1_accuracy"],
            s=75,
        )
        if row["initialization"] == "handcrafted":
            traditional_handles.append(point)
            traditional_labels.append(row["method_key"])
        else:
            offset = (5, 5)
            horizontal_alignment = "left"
            vertical_alignment = "bottom"
            if row["method_key"] == "resnet50_optimized":
                offset = (5, -8)
                vertical_alignment = "top"
            elif row["method_key"] == "deep_ensemble":
                offset = (-5, 10)
                horizontal_alignment = "right"
            axis.annotate(
                row["method_key"],
                (row["inference_time_seconds"], row["top1_accuracy"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                ha=horizontal_alignment,
                va=vertical_alignment,
            )
    if traditional_handles:
        axis.legend(
            traditional_handles,
            traditional_labels,
            title="Traditional methods",
            loc="center left",
            bbox_to_anchor=(0.01, 0.55),
            fontsize=7,
            title_fontsize=8,
        )
    axis.set_xscale("log")
    axis.margins(x=0.12)
    axis.set_xlabel("Recorded test inference time in seconds (log scale)")
    axis.set_ylabel("Top-1 accuracy")
    axis.set_title("Runtime vs Test Performance")
    axis.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_f1_distribution(path, per_class_by_method):
    labels = list(per_class_by_method)
    values = [
        [float(row["f1"]) for row in per_class_by_method[key]] for key in labels
    ]
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.boxplot(values, tick_labels=labels, showfliers=False)
    axis.set_ylabel("Per-class F1")
    axis.set_title("Per-Class F1 Distribution")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_traditional_comparison(path, summaries):
    rows = [
        row
        for row in summaries
        if row["initialization"] == "handcrafted"
        and not row["method_key"].startswith("hog_svm")
    ]
    labels = [row["method_name"] for row in rows]
    x = np.arange(len(labels))
    width = 0.27
    fig, axis = plt.subplots(figsize=(14, 7))
    for offset, key, name in [
        (-width, "top1_accuracy", "Top-1"),
        (0, "top5_accuracy", "Top-5"),
        (width, "macro_f1", "Macro-F1"),
    ]:
        bars = axis.bar(x + offset, [row[key] for row in rows], width, label=name)
        axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=18, ha="right")
    axis.set_ylim(0, max(0.12, max(row["top5_accuracy"] for row in rows) * 1.25))
    axis.set_ylabel("Score")
    axis.set_title("Traditional Methods on the Fixed 500-Class Test Split")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_comparison_outputs(output_dir, summaries, per_class_by_method):
    """Write all primary comparison tables and figures from one data snapshot."""
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(comparison_dir / "summary_metrics.csv", summaries)
    traditional_summaries = [
        row for row in summaries if row["initialization"] == "handcrafted"
    ]
    save_rows_csv(
        comparison_dir / "traditional_summary_metrics.csv",
        traditional_summaries,
    )
    save_rows_csv(
        comparison_dir / "runtime_comparison.csv",
        summaries,
        RUNTIME_FIELDS,
    )
    plot_model_comparison(comparison_dir / "model_comparison.png", summaries)
    plot_runtime(comparison_dir / "runtime_vs_performance.png", summaries)
    plot_f1_distribution(
        comparison_dir / "per_class_f1_distribution.png",
        per_class_by_method,
    )
    plot_traditional_comparison(
        comparison_dir / "traditional_methods_comparison.png",
        summaries,
    )


def artifact_category(relative_path):
    first = relative_path.parts[0] if relative_path.parts else ""
    return {
        "comparison": "comparison",
        "methods": "method_result",
        "reproducibility": "reproducibility",
    }.get(first, "documentation")


def write_manifest(output_dir):
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        relative = path.relative_to(output_dir)
        data = path.read_bytes()
        if path.suffix.lower() in CANONICAL_TEXT_SUFFIXES:
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "category": artifact_category(relative),
                "size_bytes": len(data),
            }
        )
    save_rows_csv(
        output_dir / "artifact_manifest.csv",
        rows,
        ["relative_path", "category", "size_bytes"],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export completed experiments into a compact shared results package."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("analysis/error_analysis")
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data_splits"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Destination package. Defaults to "
            "outputs/final_results/inat<num_classes>."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    classes = load_classes(args.split_dir / "selected_classes.csv")
    if args.output_dir is None:
        args.output_dir = Path("outputs/final_results") / f"inat{len(classes)}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    per_class_by_method = {}

    for method in METHODS:
        result_dir = args.results_dir / method["result_dir"]
        metrics_path = result_dir / "metrics.json"
        predictions_path = result_dir / "test_predictions.csv"
        if not metrics_path.exists() or not predictions_path.exists():
            raise FileNotFoundError(f"Missing completed result for {method['key']}")

        metrics = load_json(metrics_path)
        history_path = result_dir / "history.json"
        history = load_json(history_path) if history_path.exists() else []
        summary = build_summary(method, metrics, history)
        summaries.append(summary)

        method_dir = args.output_dir / "methods" / method["key"]
        method_dir.mkdir(parents=True, exist_ok=True)
        save_rows_csv(method_dir / "metrics.csv", [summary])
        write_history(method_dir / "history.csv", history)
        write_configuration(method_dir / "configuration.csv", method, metrics)

        prediction_rows, labels, predictions = load_predictions(predictions_path)
        save_rows_csv(
            method_dir / "test_predictions_top1.csv",
            prediction_rows,
            ["file_name", "true_class_index", "pred_class_index", "correct"],
        )
        if method["initialization"] != "handcrafted":
            test = test_metrics(metrics)
            write_top5_evidence(
                method_dir,
                prediction_rows,
                float(test["top1_accuracy"]),
                float(test["top5_accuracy"]),
                classes,
                metrics.get("data_root", "datasets/inat2021"),
            )
        else:
            copy_if_exists(
                result_dir / "test_predictions_top5.csv",
                method_dir / "test_predictions_top5.csv",
            )
        num_classes = int(summary["num_classes"])
        per_class_rows = write_per_class(
            method_dir / "per_class_metrics.csv",
            labels,
            predictions,
            classes,
            num_classes,
        )
        per_class_by_method[method["key"]] = per_class_rows
        matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
        write_confusion_csv(method_dir / "confusion_matrix.csv", matrix)
        plot_confusions(
            method_dir / "confusion_matrix_full.png",
            method_dir / "confusion_matrix_subset.png",
            matrix,
            per_class_rows,
        )

        copy_if_exists(
            result_dir / "training_curves.png", method_dir / "training_curves.png"
        )
        copy_if_exists(
            result_dir / "learning_rate.png", method_dir / "learning_rate.png"
        )
        copy_if_exists(
            result_dir / "test_prediction_examples.png",
            method_dir / "prediction_examples.png",
        )
        if method["analysis_dir"]:
            analysis_dir = args.analysis_dir / method["analysis_dir"]
            copy_if_exists(
                analysis_dir / "top_confusions.csv", method_dir / "top_confusions.csv"
            )
            copy_if_exists(
                analysis_dir / "worst_classes.csv", method_dir / "worst_classes.csv"
            )
            copy_if_exists(
                analysis_dir / "top_confusions.png", method_dir / "top_confusions.png"
            )
            resize_grid(
                analysis_dir / "correct_examples.png",
                method_dir / "correct_examples.jpg",
            )
            resize_grid(
                analysis_dir / "failure_examples.png",
                method_dir / "failure_examples.jpg",
            )

    write_comparison_outputs(args.output_dir, summaries, per_class_by_method)

    reproducibility_dir = args.output_dir / "reproducibility" / "data_splits"
    for name in [
        "selected_classes.csv",
        "class_mapping.json",
        "train.csv",
        "val.csv",
        "test.csv",
    ]:
        copy_if_exists(args.split_dir / name, reproducibility_dir / name)
    split_summary = load_json(args.split_dir / "split_summary.json")
    split_summary["data_root"] = "datasets/inat2021"
    (reproducibility_dir / "split_summary.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    write_manifest(args.output_dir)
    print(f"Exported structured results to {args.output_dir}")


if __name__ == "__main__":
    main()
