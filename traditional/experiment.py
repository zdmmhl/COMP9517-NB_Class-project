"""Unified training and evaluation for handcrafted feature experiments."""

import argparse
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data.splits import filter_rows, read_rows
from traditional.features import (
    fit_or_load_sift_vocabulary,
    load_or_extract_features,
)
from utils.serialization import save_json, save_predictions, save_rows_csv


FEATURE_NAMES = {
    "color": "HSV colour histogram",
    "lbp": "uniform local binary pattern",
    "hog": "histogram of oriented gradients",
    "sift-bovw": "SIFT Bag-of-Visual-Words",
}

CLASSIFIER_NAMES = {
    "linear-svc": "LinearSVC",
    "sgd-svm": "SGD linear SVM",
    "random-forest": "Random Forest",
}


def build_classifier(args):
    if args.classifier == "linear-svc":
        estimator = LinearSVC(
            C=args.svm_c,
            max_iter=args.max_iter,
            dual="auto",
            random_state=args.seed,
        )
        return make_pipeline(StandardScaler(), estimator)
    if args.classifier == "sgd-svm":
        estimator = SGDClassifier(
            loss="hinge",
            alpha=args.sgd_alpha,
            max_iter=args.max_iter,
            tol=1e-3,
            random_state=args.seed,
            n_jobs=-1,
        )
        return make_pipeline(StandardScaler(), estimator)
    if args.classifier == "random-forest":
        return RandomForestClassifier(
            n_estimators=args.rf_estimators,
            max_features=args.rf_max_features,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight=args.rf_class_weight,
            random_state=args.seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported classifier: {args.classifier}")


def classifier_scores(model, X):
    # Top-5 needs a ranking, so raw decision scores are sufficient.
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    else:
        raise TypeError("Classifier must expose decision_function or predict_proba.")
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    return scores


def top_k_predictions(scores, classes, k):
    k = min(k, scores.shape[1])
    indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    selected_scores = np.take_along_axis(scores, indices, axis=1)
    order = np.argsort(selected_scores, axis=1)[:, ::-1]
    sorted_indices = np.take_along_axis(indices, order, axis=1)
    sorted_scores = np.take_along_axis(scores, sorted_indices, axis=1)
    return classes[sorted_indices], sorted_scores


def evaluate(model, X, y, split_name, class_labels):
    started = time.perf_counter()
    scores = classifier_scores(model, X)
    top5_labels, top5_scores = top_k_predictions(scores, model.classes_, 5)
    y_pred = top5_labels[:, 0]
    elapsed = time.perf_counter() - started

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        y_pred,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    top1 = float(accuracy_score(y, y_pred))
    top5 = float(np.mean(np.any(top5_labels == y[:, None], axis=1)))
    metrics = OrderedDict(
        split=split_name,
        rows=int(len(y)),
        top1_accuracy=top1,
        overall_accuracy=top1,
        top5_accuracy=top5,
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
        inference_seconds=float(elapsed),
        seconds_per_image=float(elapsed / max(1, len(y))),
    )
    return metrics, y_pred, top5_labels, top5_scores


def write_top5_predictions(path, paths, labels, top5_labels, top5_scores):
    rows = []
    for file_name, label, predictions, scores in zip(
        paths,
        labels,
        top5_labels,
        top5_scores,
    ):
        row = {
            "file_name": file_name,
            "true_class_index": int(label),
            "top5_correct": int(label in predictions),
        }
        for rank, (prediction, score) in enumerate(zip(predictions, scores), 1):
            row[f"top{rank}_class_index"] = int(prediction)
            row[f"top{rank}_score"] = float(score)
        rows.append(row)
    save_rows_csv(path, rows)


def feature_parameters(args):
    common = {"image_size": args.image_size}
    if args.feature == "color":
        return {**common, "color_space": "HSV", "bins_per_channel": args.color_bins}
    if args.feature == "lbp":
        return {
            **common,
            "method": "uniform",
            "points": args.lbp_points,
            "radius": args.lbp_radius,
        }
    if args.feature == "hog":
        return {
            **common,
            "orientations": args.hog_orientations,
            "pixels_per_cell": args.hog_pixels_per_cell,
            "cells_per_block": args.hog_cells_per_block,
        }
    return {
        **common,
        "vocabulary_size": args.sift_vocabulary_size,
        "max_descriptors_per_image": args.sift_max_descriptors,
        "vocabulary_images": args.sift_vocabulary_images,
        "vocabulary_descriptor_limit": args.sift_vocabulary_descriptors,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one reproducible handcrafted feature/classifier experiment."
    )
    parser.add_argument(
        "--feature",
        choices=list(FEATURE_NAMES),
        required=True,
    )
    parser.add_argument(
        "--classifier",
        choices=list(CLASSIFIER_NAMES),
        default="linear-svc",
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/inat2021"))
    parser.add_argument("--split-dir", type=Path, default=Path("data_splits"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("feature_cache/traditional"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--color-bins", type=int, default=32)
    parser.add_argument("--lbp-points", type=int, default=24)
    parser.add_argument("--lbp-radius", type=float, default=3.0)
    parser.add_argument("--hog-orientations", type=int, default=9)
    parser.add_argument("--hog-pixels-per-cell", type=int, default=16)
    parser.add_argument("--hog-cells-per-block", type=int, default=2)
    parser.add_argument("--sift-vocabulary-size", type=int, default=128)
    parser.add_argument("--sift-max-descriptors", type=int, default=64)
    parser.add_argument("--sift-vocabulary-images", type=int, default=5000)
    parser.add_argument("--sift-vocabulary-descriptors", type=int, default=100000)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--sgd-alpha", type=float, default=0.0001)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--rf-estimators", type=int, default=100)
    parser.add_argument("--rf-max-features", default="sqrt")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument(
        "--rf-class-weight",
        choices=["balanced", "balanced_subsample", "none"],
        default="balanced_subsample",
    )
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rf_class_weight == "none":
        args.rf_class_weight = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_started = time.perf_counter()

    train_rows = filter_rows(
        read_rows(args.split_dir / "train.csv"),
        max_classes=args.max_classes,
        max_per_class=args.max_per_class,
    )
    val_rows = filter_rows(
        read_rows(args.split_dir / "val.csv"),
        max_classes=args.max_classes,
        max_per_class=args.max_per_class,
    )
    test_rows = filter_rows(
        read_rows(args.split_dir / "test.csv"),
        max_classes=args.max_classes,
        max_per_class=args.max_per_class,
    )
    if not train_rows or not val_rows or not test_rows:
        raise ValueError("One or more splits are empty after filtering.")

    class_labels = sorted({int(row["class_index"]) for row in train_rows})
    print(
        f"{FEATURE_NAMES[args.feature]} + {CLASSIFIER_NAMES[args.classifier]}\n"
        f"  classes: {len(class_labels)}\n"
        f"  train/val/test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}"
    )

    vocabulary_info = {}
    if args.feature == "sift-bovw":
        vocabulary_info = fit_or_load_sift_vocabulary(
            train_rows,
            args.data_root,
            args.cache_dir / "sift-bovw",
            args,
        )
        args.sift_vocabulary = vocabulary_info["model"]

    payloads = {}
    for split_name, rows in [
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ]:
        payloads[split_name] = load_or_extract_features(
            rows,
            split_name,
            args.feature,
            args.data_root,
            args.cache_dir / args.feature,
            args,
        )

    model = build_classifier(args)
    train_started = time.perf_counter()
    model.fit(payloads["train"]["X"], payloads["train"]["y"])
    train_seconds = time.perf_counter() - train_started
    print(f"Trained classifier in {train_seconds:.1f}s")

    metrics = {
        "method": f"{FEATURE_NAMES[args.feature]} + {CLASSIFIER_NAMES[args.classifier]}",
        "feature": args.feature,
        "classifier": args.classifier,
        "seed": args.seed,
        "num_classes": len(class_labels),
        "rows": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "params": {
            "feature": feature_parameters(args),
            "classifier": {
                "name": args.classifier,
                "svm_c": args.svm_c if args.classifier == "linear-svc" else None,
                "sgd_alpha": args.sgd_alpha if args.classifier == "sgd-svm" else None,
                "max_iter": args.max_iter
                if args.classifier in {"linear-svc", "sgd-svm"}
                else None,
                "rf_estimators": args.rf_estimators
                if args.classifier == "random-forest"
                else None,
                "rf_max_features": args.rf_max_features
                if args.classifier == "random-forest"
                else None,
                "rf_min_samples_leaf": args.rf_min_samples_leaf
                if args.classifier == "random-forest"
                else None,
                "rf_class_weight": args.rf_class_weight
                if args.classifier == "random-forest"
                else None,
            },
            "workers": args.workers,
            "max_classes": args.max_classes,
            "max_per_class": args.max_per_class,
        },
        "feature_extraction": {
            split: {
                "seconds": payloads[split].get("extraction_seconds", 0.0),
                "cache_hit": payloads[split].get("cache_hit", False),
                "feature_dimension": int(payloads[split]["X"].shape[1]),
            }
            for split in payloads
        },
        "vocabulary": {
            key: value
            for key, value in vocabulary_info.items()
            if key not in {"model", "path"}
        },
        "train_seconds": float(train_seconds),
        "splits": {},
    }

    for split_name, payload in payloads.items():
        split_metrics, predictions, top5_labels, top5_scores = evaluate(
            model,
            payload["X"],
            payload["y"],
            split_name,
            class_labels,
        )
        metrics["splits"][split_name] = split_metrics
        print(
            f"{split_name}: top1={split_metrics['top1_accuracy']:.4f}, "
            f"top5={split_metrics['top5_accuracy']:.4f}, "
            f"macro_f1={split_metrics['macro_f1']:.4f}"
        )
        report = classification_report(
            payload["y"],
            predictions,
            labels=class_labels,
            zero_division=0,
            output_dict=True,
        )
        save_json(args.output_dir / f"{split_name}_classification_report.json", report)
        np.save(
            args.output_dir / f"{split_name}_confusion_matrix.npy",
            confusion_matrix(payload["y"], predictions, labels=class_labels),
        )
        save_predictions(
            args.output_dir / f"{split_name}_predictions.csv",
            payload["paths"],
            payload["y"],
            predictions,
        )
        write_top5_predictions(
            args.output_dir / f"{split_name}_predictions_top5.csv",
            payload["paths"],
            payload["y"],
            top5_labels,
            top5_scores,
        )

    metrics["total_seconds"] = float(time.perf_counter() - experiment_started)
    joblib.dump(model, args.output_dir / "model.joblib", compress=3)
    save_json(args.output_dir / "metrics.json", metrics)
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
