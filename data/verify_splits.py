"""Validate split sizes, class balance, paths, and leakage."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_args():
    parser = argparse.ArgumentParser(description="Verify generated iNaturalist split manifests.")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/inat2021"))
    parser.add_argument("--split-dir", type=Path, default=Path("data_splits"))
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--check-all-paths", action="store_true")
    return parser.parse_args()


def verify_split(name, rows, expected_per_class, num_classes, data_root, check_all_paths):
    expected_total = expected_per_class * num_classes
    if len(rows) != expected_total:
        raise AssertionError(f"{name}: expected {expected_total} rows, got {len(rows)}")

    counts = Counter(int(row["class_index"]) for row in rows)
    if len(counts) != num_classes:
        raise AssertionError(f"{name}: expected {num_classes} classes, got {len(counts)}")

    bad_counts = {
        class_index: count
        for class_index, count in counts.items()
        if count != expected_per_class
    }
    if bad_counts:
        raise AssertionError(f"{name}: per-class count mismatch: {list(bad_counts.items())[:10]}")

    sample_count = len(rows) if check_all_paths else min(100, len(rows))
    missing = [
        row["file_name"]
        for row in rows[:sample_count]
        if not (data_root / row["file_name"]).exists()
    ]
    if missing:
        raise AssertionError(f"{name}: missing image paths, first examples: {missing[:5]}")


def main():
    args = parse_args()
    split_dir = args.split_dir
    data_root = args.data_root

    train_rows = read_csv(split_dir / "train.csv")
    val_rows = read_csv(split_dir / "val.csv")
    test_rows = read_csv(split_dir / "test.csv")

    verify_split(
        "train",
        train_rows,
        args.train_per_class,
        args.num_classes,
        data_root,
        args.check_all_paths,
    )
    verify_split(
        "val",
        val_rows,
        args.val_per_class,
        args.num_classes,
        data_root,
        args.check_all_paths,
    )
    verify_split(
        "test",
        test_rows,
        args.test_per_class,
        args.num_classes,
        data_root,
        args.check_all_paths,
    )

    path_sets = {
        "train": {row["file_name"] for row in train_rows},
        "val": {row["file_name"] for row in val_rows},
        "test": {row["file_name"] for row in test_rows},
    }
    overlaps = {
        "train_val": len(path_sets["train"] & path_sets["val"]),
        "train_test": len(path_sets["train"] & path_sets["test"]),
        "val_test": len(path_sets["val"] & path_sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Split overlap found: {overlaps}")

    with (split_dir / "split_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    print("Split verification passed")
    print(f"  data_root: {data_root}")
    print(f"  seed: {summary['seed']}")
    print(f"  classes: {summary['num_classes']}")
    print(f"  train/val/test rows: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"  overlaps: {overlaps}")


if __name__ == "__main__":
    main()
