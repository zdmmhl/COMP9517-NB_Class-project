import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    return json.loads(
        (PROJECT_ROOT / "configs" / "class_scaling.json").read_text(
            encoding="utf-8"
        )
    )


def test_main_scaling_keeps_total_image_budget_constant():
    specs = load_config()["split_specs"]
    main = [spec for spec in specs.values() if spec["kind"] == "main"]
    assert [spec["num_classes"] for spec in main] == [500, 1000, 2500]
    assert {
        spec["num_classes"] * spec["train_per_class"] for spec in main
    } == {20000}
    assert {
        spec["num_classes"] * spec["val_per_class"] for spec in main
    } == {5000}
    assert {
        spec["num_classes"] * spec["test_per_class"] for spec in main
    } == {5000}


def test_sample_controls_keep_500_classes():
    specs = load_config()["split_specs"]
    controls = [
        spec for spec in specs.values() if spec["kind"] == "sample_control"
    ]
    assert {spec["num_classes"] for spec in controls} == {500}
    assert {spec["train_per_class"] for spec in controls} == {8, 20}


def test_advanced_training_matches_deep_reference():
    scaling = load_config()
    deep = json.loads(
        (PROJECT_ROOT / "configs" / "deep_ablations.json").read_text(
            encoding="utf-8"
        )
    )
    reference = {
        **deep["common"],
        **deep["training_runs"][scaling["base_result_run"]],
    }
    assert scaling["common_training"] == reference
