import json

import pandas as pd

from evaluation.recorded_results import prepare_shared_manifests
from scripts.evaluate_recorded_results import load_split_identity


def test_load_split_identity_uses_recorded_class_count_and_seed(tmp_path):
    split_root = tmp_path / "reproducibility" / "data_splits"
    split_root.mkdir(parents=True)
    (split_root / "split_summary.json").write_text(
        json.dumps({"num_classes": 500, "seed": 42}),
        encoding="utf-8",
    )

    assert load_split_identity(tmp_path) == ("inat500_seed42", 500, 42)


def test_prepare_shared_manifests_converts_recorded_split(tmp_path):
    split_root = tmp_path / "final" / "reproducibility" / "data_splits"
    split_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"class_index": 0, "category_id": 10, "name": "Species A"},
            {"class_index": 1, "category_id": 20, "name": "Species B"},
        ]
    ).to_csv(split_root / "selected_classes.csv", index=False)
    pd.DataFrame(
        [
            {"image_id": 100, "file_name": "a.jpg", "class_index": 0},
            {"image_id": 200, "file_name": "b.jpg", "class_index": 1},
        ]
    ).to_csv(split_root / "test.csv", index=False)

    mapping_path, test_path, original_test = prepare_shared_manifests(
        tmp_path / "final",
        tmp_path / "run" / "shared",
    )

    mapping = pd.read_csv(mapping_path)
    test = pd.read_csv(test_path)
    assert mapping.columns.tolist() == [
        "class_index",
        "category_id",
        "species_name",
    ]
    assert test.columns.tolist() == ["sample_id", "image_path", "true_label"]
    assert len(original_test) == 2
