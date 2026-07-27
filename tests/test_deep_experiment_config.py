import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_deep_ablation_studies_change_one_declared_factor():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "deep_ablations.json").read_text(
            encoding="utf-8"
        )
    )
    common = config["common"]
    runs = {
        key: {**common, **overrides}
        for key, overrides in config["training_runs"].items()
    }
    for key, spec in config["evaluation_runs"].items():
        runs[key] = {
            **runs[spec["source_run"]],
            **{name: value for name, value in spec.items() if name != "source_run"},
        }

    for study in config["studies"]:
        baseline = runs[study["variants"][study["baseline_variant"]]]
        allowed = set(study["allowed_config_differences"])
        for variant in study["variant_order"]:
            candidate = runs[study["variants"][variant]]
            changed = {
                key
                for key in set(baseline) | set(candidate)
                if baseline.get(key) != candidate.get(key)
            }
            if variant == study["baseline_variant"]:
                assert changed == set()
            elif study["factor"] == "initialization":
                assert changed == {"model"}
            else:
                assert changed == allowed


def test_all_training_runs_have_the_same_budget_and_seed():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "deep_ablations.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["seed"] == 9517
    for overrides in config["training_runs"].values():
        merged = {**config["common"], **overrides}
        assert merged["epochs"] == 50
        assert merged["early_stopping_patience"] == 8
        assert merged["batch_size"] == 512
        assert merged["eval_batch_size"] == 1024
        assert merged["cuda_memory_fraction"] == 1.0


def test_deep_ablation_paths_are_repository_relative():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "deep_ablations.json").read_text(
            encoding="utf-8"
        )
    )
    assert not Path(config["data_root"]).is_absolute()
    assert not Path(config["split_dir"]).is_absolute()
    assert not Path(config["results_root"]).is_absolute()
