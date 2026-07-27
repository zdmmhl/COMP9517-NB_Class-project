"""Construct nested iNaturalist subsets for the class-scaling study."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from data.prepare_splits import (
    category_metadata,
    group_images_by_category,
    load_json,
    split_rows,
    stable_species_record,
    write_csv,
    write_json,
)


MANIFEST_FIELDS = [
    "split",
    "class_index",
    "category_id",
    "image_id",
    "annotation_id",
    "file_name",
    "species_name",
    "common_name",
    "kingdom",
    "family",
    "genus",
    "width",
    "height",
]
CLASS_FIELDS = [
    "class_index",
    "category_id",
    "name",
    "common_name",
    "kingdom",
    "phylum",
    "taxon_class",
    "order",
    "family",
    "genus",
    "specific_epithet",
    "image_dir_name",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_by_category(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["category_id"]), []).append(row)
    return grouped


def relabel_rows(
    rows: list[dict[str, str]],
    split_name: str,
    class_index: int,
) -> list[dict[str, object]]:
    output = []
    for row in rows:
        item = dict(row)
        item["split"] = split_name
        item["class_index"] = class_index
        output.append(item)
    return output


def deterministic_sample(
    images: list[dict],
    count: int,
    seed: int,
    category_id: int,
    stream: int,
) -> list[dict]:
    values = list(images)
    random.Random(seed + category_id * 1009 + stream * 1_000_003).shuffle(values)
    return values[:count]


def validate_split(
    output_dir: Path,
    data_root: Path,
    num_classes: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
) -> dict:
    manifests = {
        name: read_csv(output_dir / f"{name}.csv")
        for name in ["train", "val", "test"]
    }
    expected = {
        "train": train_per_class,
        "val": val_per_class,
        "test": test_per_class,
    }
    checks = {}
    for name, rows in manifests.items():
        counts = Counter(int(row["class_index"]) for row in rows)
        if set(counts) != set(range(num_classes)):
            raise ValueError(f"{output_dir}/{name}: class indices are incomplete")
        if set(counts.values()) != {expected[name]}:
            raise ValueError(f"{output_dir}/{name}: per-class counts are incorrect")
        checks[f"{name}_rows"] = len(rows)

    path_sets = {
        name: {row["file_name"] for row in rows}
        for name, rows in manifests.items()
    }
    if path_sets["train"] & path_sets["val"]:
        raise ValueError(f"{output_dir}: train and validation images overlap")
    if path_sets["train"] & path_sets["test"]:
        raise ValueError(f"{output_dir}: train and test images overlap")
    if path_sets["val"] & path_sets["test"]:
        raise ValueError(f"{output_dir}: validation and test images overlap")
    missing_paths = [
        file_name
        for file_name in set().union(*path_sets.values())
        if not (data_root / file_name).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{output_dir}: {len(missing_paths)} image files are missing; "
            f"examples={missing_paths[:5]}"
        )
    checks["cross_split_overlap"] = 0
    checks["missing_image_files"] = 0
    checks["valid"] = True
    return checks


def build_scaling_splits(config: dict, project_root: Path) -> dict:
    configured_data_root = Path(config["data_root"])
    data_root = configured_data_root
    if not data_root.is_absolute():
        data_root = project_root / data_root
    base_dir = project_root / config["base_split_dir"]
    output_root = project_root / config["split_root"]
    seed = int(config["seed"])
    train_dataset = load_json(data_root / "train_mini.json")
    test_dataset = load_json(data_root / "val.json")
    train_by_category = group_images_by_category(train_dataset)
    test_by_category = group_images_by_category(test_dataset)
    categories = category_metadata(train_dataset["categories"])

    base_classes = read_csv(base_dir / "selected_classes.csv")
    base_ids = [int(row["category_id"]) for row in base_classes]
    if len(base_ids) != 500:
        raise ValueError("The class-scaling study requires the fixed 500-class base split")

    eligible_1000 = [
        category_id
        for category_id in sorted(categories)
        if len(train_by_category.get(category_id, [])) >= 25
        and len(test_by_category.get(category_id, [])) >= 5
        and category_id not in set(base_ids)
    ]
    rng = random.Random(seed + 1000)
    added_1000 = sorted(rng.sample(eligible_1000, 500))
    classes_1000 = [*base_ids, *added_1000]

    selected_1000 = set(classes_1000)
    eligible_2500 = [
        category_id
        for category_id in sorted(categories)
        if len(train_by_category.get(category_id, [])) >= 10
        and len(test_by_category.get(category_id, [])) >= 2
        and category_id not in selected_1000
    ]
    rng = random.Random(seed + 2500)
    added_2500 = sorted(rng.sample(eligible_2500, 1500))
    classes_2500 = [*classes_1000, *added_2500]

    class_lists = {
        "classes_500": base_ids,
        "classes_1000": classes_1000,
        "classes_2500": classes_2500,
        "control_500x20": base_ids,
        "control_500x8": base_ids,
    }
    base_rows = {
        name: rows_by_category(read_csv(base_dir / f"{name}.csv"))
        for name in ["train", "val", "test"]
    }

    generated = {}
    for split_key, spec in config["split_specs"].items():
        category_ids = class_lists[split_key]
        class_index_by_category = {
            category_id: index for index, category_id in enumerate(category_ids)
        }
        manifests = {"train": [], "val": [], "test": []}
        class_rows = []
        for category_id in category_ids:
            class_index = class_index_by_category[category_id]
            category = categories[category_id]
            class_rows.append(stable_species_record(category, class_index))
            if category_id in base_rows["train"]:
                for name in manifests:
                    count = int(spec[f"{name}_per_class"])
                    selected = base_rows[name][category_id][:count]
                    manifests[name].extend(
                        relabel_rows(selected, name, class_index)
                    )
                continue

            train_count = int(spec["train_per_class"])
            val_count = int(spec["val_per_class"])
            if category_id in selected_1000:
                pool_train_count, pool_val_count = 20, 5
            else:
                pool_train_count, pool_val_count = 8, 2
            selected_train_mini = deterministic_sample(
                train_by_category[category_id],
                pool_train_count + pool_val_count,
                seed,
                category_id,
                1,
            )
            selected_test = deterministic_sample(
                test_by_category[category_id],
                int(spec["test_per_class"]),
                seed,
                category_id,
                2,
            )
            manifests["train"].extend(
                split_rows(
                    selected_train_mini[:train_count],
                    "train",
                    class_index,
                    category,
                )
            )
            manifests["val"].extend(
                split_rows(
                    selected_train_mini[
                        pool_train_count : pool_train_count + val_count
                    ],
                    "val",
                    class_index,
                    category,
                )
            )
            manifests["test"].extend(
                split_rows(selected_test, "test", class_index, category)
            )

        output_dir = output_root / split_key
        for name, rows in manifests.items():
            write_csv(output_dir / f"{name}.csv", rows, MANIFEST_FIELDS)
        write_csv(output_dir / "selected_classes.csv", class_rows, CLASS_FIELDS)
        write_json(
            output_dir / "class_mapping.json",
            {
                "seed": seed,
                "num_classes": int(spec["num_classes"]),
                "category_id_to_class_index": {
                    str(category_id): index
                    for category_id, index in class_index_by_category.items()
                },
                "classes": class_rows,
            },
        )
        checks = validate_split(
            output_dir,
            data_root,
            int(spec["num_classes"]),
            int(spec["train_per_class"]),
            int(spec["val_per_class"]),
            int(spec["test_per_class"]),
        )
        summary = {
            "study": config["study_name"],
            "split_key": split_key,
            "kind": spec["kind"],
            "data_root": configured_data_root.as_posix(),
            "train_annotations": "train_mini.json",
            "test_annotations": "val.json",
            "seed": seed,
            **spec,
            "rows": {
                name: len(rows) for name, rows in manifests.items()
            },
            "base_classes_preserved": category_ids[:500] == base_ids,
            "checks": checks,
        }
        write_json(output_dir / "split_summary.json", summary)
        summary["sha256"] = {
            name: file_sha256(output_dir / name)
            for name in [
                "train.csv",
                "val.csv",
                "test.csv",
                "selected_classes.csv",
            ]
        }
        write_json(output_dir / "split_summary.json", summary)
        generated[split_key] = summary

    if classes_1000[:500] != base_ids:
        raise ValueError("The 1000-class split does not preserve the base classes")
    if classes_2500[:1000] != classes_1000:
        raise ValueError("The 2500-class split does not preserve the 1000 classes")

    def category_files(split_key: str, split_name: str) -> dict[int, set[str]]:
        rows = read_csv(output_root / split_key / f"{split_name}.csv")
        grouped: dict[int, set[str]] = {}
        for row in rows:
            grouped.setdefault(int(row["category_id"]), set()).add(row["file_name"])
        return grouped

    sample_nesting = {}
    for split_name in ["train", "val", "test"]:
        base_files = category_files("classes_500", split_name)
        thousand_files = category_files("classes_1000", split_name)
        twenty_five_hundred_files = category_files("classes_2500", split_name)
        control_20_files = category_files("control_500x20", split_name)
        control_8_files = category_files("control_500x8", split_name)
        checks = {
            "classes_1000_samples_nested_in_500_for_base_classes": all(
                thousand_files[category_id] <= base_files[category_id]
                for category_id in base_ids
            ),
            "classes_2500_samples_nested_in_1000_for_first_1000_classes": all(
                twenty_five_hundred_files[category_id]
                <= thousand_files[category_id]
                for category_id in classes_1000
            ),
            "control_500x20_samples_nested_in_base": all(
                control_20_files[category_id] <= base_files[category_id]
                for category_id in base_ids
            ),
            "control_500x8_samples_nested_in_base": all(
                control_8_files[category_id] <= base_files[category_id]
                for category_id in base_ids
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"Sample nesting failed for {split_name}: {checks}")
        sample_nesting[split_name] = checks
    report = {
        "seed": seed,
        "base_500_nested_in_1000": True,
        "classes_1000_nested_in_2500": True,
        "sample_nesting": sample_nesting,
        "splits": generated,
        "valid": True,
    }
    write_json(output_root / "validation_report.json", report)
    return report
