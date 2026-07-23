import argparse
import csv
import json
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def filter_rows(rows, max_classes=None, max_per_class=None):
    if max_classes is None and max_per_class is None:
        return rows

    selected_classes = []
    seen = set()
    for row in rows:
        class_index = int(row["class_index"])
        if class_index not in seen:
            seen.add(class_index)
            selected_classes.append(class_index)
        if max_classes is not None and len(selected_classes) >= max_classes:
            break

    selected_set = set(selected_classes)
    counts = {}
    filtered = []
    for row in rows:
        class_index = int(row["class_index"])
        if class_index not in selected_set:
            continue
        count = counts.get(class_index, 0)
        if max_per_class is not None and count >= max_per_class:
            continue
        filtered.append(row)
        counts[class_index] = count + 1
    return filtered


def load_grayscale_image(path, image_size):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def extract_hog_feature(image, orientations, pixels_per_cell, cells_per_block):
    return hog(
        image,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)


def feature_cache_name(split_name, args, rows):
    first = rows[0]["class_index"] if rows else "empty"
    last = rows[-1]["class_index"] if rows else "empty"
    return (
        f"{split_name}_hog_size{args.image_size}_ori{args.orientations}"
        f"_ppc{args.pixels_per_cell}_cpb{args.cells_per_block}"
        f"_n{len(rows)}_c{args.max_classes or 'all'}_p{args.max_per_class or 'all'}"
        f"_first{first}_last{last}.joblib"
    )


def load_or_extract_features(rows, split_name, data_root, cache_dir, args):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / feature_cache_name(split_name, args, rows)
    if cache_path.exists() and not args.no_cache:
        print(f"Loading cached {split_name} features: {cache_path}")
        return joblib.load(cache_path)

    started = time.perf_counter()
    features = []
    labels = []
    paths = []

    for index, row in enumerate(rows, 1):
        image_path = data_root / row["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        image = load_grayscale_image(image_path, args.image_size)
        feature = extract_hog_feature(
            image,
            args.orientations,
            args.pixels_per_cell,
            args.cells_per_block,
        )
        features.append(feature)
        labels.append(int(row["class_index"]))
        paths.append(row["file_name"])

        if index % args.progress_every == 0 or index == len(rows):
            elapsed = time.perf_counter() - started
            print(f"  {split_name}: extracted {index}/{len(rows)} in {elapsed:.1f}s")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    payload = {"X": X, "y": y, "paths": paths}

    if not args.no_cache:
        joblib.dump(payload, cache_path, compress=3)
        print(f"Saved {split_name} feature cache: {cache_path}")

    return payload


def top_k_accuracy(scores, y_true, k, classes):
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    k = min(k, scores.shape[1])
    top_k_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_k_labels = classes[top_k_indices]
    return float(np.mean([label in row for label, row in zip(y_true, top_k_labels)]))


def evaluate(model, X, y, split_name, class_labels):
    started = time.perf_counter()
    y_pred = model.predict(X)
    scores = model.decision_function(X)
    elapsed = time.perf_counter() - started

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        y_pred,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    metrics = OrderedDict(
        split=split_name,
        rows=int(len(y)),
        top1_accuracy=float(accuracy_score(y, y_pred)),
        top5_accuracy=top_k_accuracy(scores, y, 5, model.classes_),
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
        inference_seconds=float(elapsed),
        seconds_per_image=float(elapsed / max(1, len(y))),
    )
    return metrics, y_pred


def write_predictions(path, paths, y_true, y_pred):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "true_class_index", "pred_class_index", "correct"],
        )
        writer.writeheader()
        for file_name, true_label, pred_label in zip(paths, y_true, y_pred):
            writer.writerow(
                {
                    "file_name": file_name,
                    "true_class_index": int(true_label),
                    "pred_class_index": int(pred_label),
                    "correct": int(true_label == pred_label),
                }
            )


def write_metrics(path, metrics):
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")


def write_classification_report(path, y_true, y_pred, labels):
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


def write_confusion_matrix(path, y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    np.save(path, matrix)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a HOG + linear SVM baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/inat2021"))
    parser.add_argument("--split-dir", type=Path, default=Path("data_splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hog_svm"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/feature_cache/hog"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--orientations", type=int, default=9)
    parser.add_argument("--pixels-per-cell", type=int, default=16)
    parser.add_argument("--cells-per-block", type=int, default=2)
    parser.add_argument("--classifier", choices=["sgd-svm", "linear-svc"], default="sgd-svm")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--sgd-alpha", type=float, default=0.0001)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    print("HOG + Linear SVM baseline")
    print(f"  data_root: {args.data_root}")
    print(f"  classes: {len(class_labels)}")
    print(f"  train/val/test rows: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")

    train = load_or_extract_features(train_rows, "train", args.data_root, args.cache_dir, args)
    val = load_or_extract_features(val_rows, "val", args.data_root, args.cache_dir, args)
    test = load_or_extract_features(test_rows, "test", args.data_root, args.cache_dir, args)

    if args.classifier == "linear-svc":
        classifier = LinearSVC(C=args.svm_c, max_iter=args.max_iter, dual="auto", random_state=9517)
        method_name = "HOG + LinearSVC"
    else:
        classifier = SGDClassifier(
            loss="hinge",
            alpha=args.sgd_alpha,
            max_iter=args.max_iter,
            tol=1e-3,
            random_state=9517,
            n_jobs=-1,
        )
        method_name = "HOG + SGD linear SVM"

    model = make_pipeline(StandardScaler(), classifier)

    train_started = time.perf_counter()
    model.fit(train["X"], train["y"])
    train_seconds = time.perf_counter() - train_started
    print(f"Trained SVM in {train_seconds:.1f}s")

    metrics = {
        "method": method_name,
        "params": {
            "classifier": args.classifier,
            "image_size": args.image_size,
            "orientations": args.orientations,
            "pixels_per_cell": args.pixels_per_cell,
            "cells_per_block": args.cells_per_block,
            "svm_c": args.svm_c,
            "sgd_alpha": args.sgd_alpha,
            "max_iter": args.max_iter,
            "max_classes": args.max_classes,
            "max_per_class": args.max_per_class,
        },
        "train_seconds": float(train_seconds),
        "splits": {},
    }

    for split_name, payload in [("train", train), ("val", val), ("test", test)]:
        split_metrics, y_pred = evaluate(model, payload["X"], payload["y"], split_name, class_labels)
        metrics["splits"][split_name] = split_metrics
        print(
            f"{split_name}: top1={split_metrics['top1_accuracy']:.4f}, "
            f"top5={split_metrics['top5_accuracy']:.4f}, "
            f"macro_f1={split_metrics['macro_f1']:.4f}"
        )
        write_classification_report(
            args.output_dir / f"{split_name}_classification_report.json",
            payload["y"],
            y_pred,
            class_labels,
        )
        write_confusion_matrix(
            args.output_dir / f"{split_name}_confusion_matrix.npy",
            payload["y"],
            y_pred,
            class_labels,
        )
        write_predictions(
            args.output_dir / f"{split_name}_predictions.csv",
            payload["paths"],
            payload["y"],
            y_pred,
        )

    joblib.dump(model, args.output_dir / "model.joblib", compress=3)
    write_metrics(args.output_dir / "metrics.json", metrics)
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
