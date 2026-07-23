"""Build reproducible iNaturalist train, validation, and test manifests."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_DATA_ROOT = Path("datasets/inat2021")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def group_images_by_category(dataset):
    image_by_id = {image["id"]: image for image in dataset["images"]}
    grouped = defaultdict(list)

    for annotation in dataset["annotations"]:
        image = image_by_id[annotation["image_id"]]
        grouped[annotation["category_id"]].append(
            {
                "image_id": image["id"],
                "annotation_id": annotation["id"],
                "category_id": annotation["category_id"],
                "file_name": image["file_name"],
                "width": image.get("width"),
                "height": image.get("height"),
            }
        )

    for images in grouped.values():
        images.sort(key=lambda row: row["file_name"])

    return grouped


def category_metadata(categories):
    return {category["id"]: category for category in categories}


def stable_species_record(category, class_index):
    return {
        "class_index": class_index,
        "category_id": category["id"],
        "name": category.get("name", ""),
        "common_name": category.get("common_name", ""),
        "kingdom": category.get("kingdom", ""),
        "phylum": category.get("phylum", ""),
        "taxon_class": category.get("class", ""),
        "order": category.get("order", ""),
        "family": category.get("family", ""),
        "genus": category.get("genus", ""),
        "specific_epithet": category.get("specific_epithet", ""),
        "image_dir_name": category.get("image_dir_name", ""),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def split_rows(images, split_name, class_index, category):
    rows = []
    for item in images:
        rows.append(
            {
                "split": split_name,
                "class_index": class_index,
                "category_id": item["category_id"],
                "image_id": item["image_id"],
                "annotation_id": item["annotation_id"],
                "file_name": item["file_name"],
                "species_name": category.get("name", ""),
                "common_name": category.get("common_name", ""),
                "kingdom": category.get("kingdom", ""),
                "family": category.get("family", ""),
                "genus": category.get("genus", ""),
                "width": item.get("width"),
                "height": item.get("height"),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create reproducible 500-class iNaturalist-2021 train/val/test split manifests."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("data_splits"))
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--train-annotations", default="train_mini.json")
    parser.add_argument("--test-annotations", default="val.json")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root
    output_dir = args.output_dir

    train_json_path = data_root / args.train_annotations
    test_json_path = data_root / args.test_annotations

    if not train_json_path.exists():
        raise FileNotFoundError(f"Missing train annotations: {train_json_path}")
    if not test_json_path.exists():
        raise FileNotFoundError(f"Missing test annotations: {test_json_path}")

    train_dataset = load_json(train_json_path)
    test_dataset = load_json(test_json_path)

    train_by_category = group_images_by_category(train_dataset)
    test_by_category = group_images_by_category(test_dataset)
    categories = category_metadata(train_dataset["categories"])

    required_train_mini = args.train_per_class + args.val_per_class
    eligible_category_ids = sorted(
        category_id
        for category_id in categories
        if len(train_by_category.get(category_id, [])) >= required_train_mini
        and len(test_by_category.get(category_id, [])) >= args.test_per_class
    )

    if len(eligible_category_ids) < args.num_classes:
        raise ValueError(
            f"Only {len(eligible_category_ids)} eligible classes, need {args.num_classes}."
        )

    rng = random.Random(args.seed)
    selected_category_ids = sorted(rng.sample(eligible_category_ids, args.num_classes))
    class_index_by_category = {
        category_id: class_index for class_index, category_id in enumerate(selected_category_ids)
    }

    train_rows = []
    val_rows = []
    test_rows = []
    class_rows = []

    for category_id in selected_category_ids:
        class_index = class_index_by_category[category_id]
        category = categories[category_id]
        class_rows.append(stable_species_record(category, class_index))

        train_mini_images = list(train_by_category[category_id])
        test_images = list(test_by_category[category_id])
        rng.shuffle(train_mini_images)
        rng.shuffle(test_images)

        selected_train = train_mini_images[: args.train_per_class]
        selected_val = train_mini_images[
            args.train_per_class : args.train_per_class + args.val_per_class
        ]
        selected_test = test_images[: args.test_per_class]

        train_rows.extend(split_rows(selected_train, "train", class_index, category))
        val_rows.extend(split_rows(selected_val, "val", class_index, category))
        test_rows.extend(split_rows(selected_test, "test", class_index, category))

    manifest_fields = [
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
    class_fields = [
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

    write_csv(output_dir / "train.csv", train_rows, manifest_fields)
    write_csv(output_dir / "val.csv", val_rows, manifest_fields)
    write_csv(output_dir / "test.csv", test_rows, manifest_fields)
    write_csv(output_dir / "selected_classes.csv", class_rows, class_fields)

    write_json(
        output_dir / "class_mapping.json",
        {
            "seed": args.seed,
            "num_classes": args.num_classes,
            "category_id_to_class_index": {
                str(category_id): class_index
                for category_id, class_index in class_index_by_category.items()
            },
            "classes": class_rows,
        },
    )

    summary = {
        "data_root": str(data_root).replace("\\", "/"),
        "train_annotations": args.train_annotations,
        "test_annotations": args.test_annotations,
        "seed": args.seed,
        "num_classes": args.num_classes,
        "eligible_classes": len(eligible_category_ids),
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "test_per_class": args.test_per_class,
        "rows": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "expected_rows": {
            "train": args.num_classes * args.train_per_class,
            "val": args.num_classes * args.val_per_class,
            "test": args.num_classes * args.test_per_class,
        },
        "split_files": {
            "train": str((output_dir / "train.csv").as_posix()),
            "val": str((output_dir / "val.csv").as_posix()),
            "test": str((output_dir / "test.csv").as_posix()),
            "selected_classes": str((output_dir / "selected_classes.csv").as_posix()),
            "class_mapping": str((output_dir / "class_mapping.json").as_posix()),
        },
    }
    write_json(output_dir / "split_summary.json", summary)

    print("Created iNaturalist-2021 split manifests")
    print(f"  output_dir: {output_dir}")
    print(f"  seed: {args.seed}")
    print(f"  classes: {args.num_classes}")
    print(f"  train rows: {len(train_rows)}")
    print(f"  val rows: {len(val_rows)}")
    print(f"  test rows: {len(test_rows)}")


if __name__ == "__main__":
    main()
