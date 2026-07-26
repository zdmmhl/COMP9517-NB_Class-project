"""Build the compact, report-ready Advanced class-scaling package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.class_scaling import write_class_scaling_outputs
from evaluation.compact_experiment import export_deep_run


def copy_reproducibility_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in [
        "train.csv",
        "val.csv",
        "test.csv",
        "selected_classes.csv",
        "class_mapping.json",
        "split_summary.json",
    ]:
        shutil.copy2(source / name, destination / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "class_scaling.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = PROJECT_ROOT / config["output_dir"]
    run_root = output_dir / "runs"
    split_root = PROJECT_ROOT / config["split_root"]

    for run_key, spec in config["split_specs"].items():
        if run_key == "classes_500":
            raw_dir = (
                PROJECT_ROOT
                / config["deep_ablation_results_root"]
                / config["base_result_run"]
            )
        else:
            raw_dir = PROJECT_ROOT / config["results_root"] / run_key
        export_deep_run(
            raw_dir,
            run_root / run_key,
            split_root / run_key,
            method_key=f"class_scaling_{run_key}",
            method_name=f"ResNet18 class scaling: {run_key}",
            split_id=f"inat_{run_key}_seed{config['seed']}",
            include_confusion_csv=int(spec["num_classes"]) <= 1000,
        )
        copy_reproducibility_files(
            split_root / run_key,
            output_dir / "reproducibility" / run_key,
        )

    shutil.copy2(
        split_root / "validation_report.json",
        output_dir / "reproducibility" / "validation_report.json",
    )
    summary = write_class_scaling_outputs(
        run_root,
        split_root,
        output_dir,
        config["split_specs"],
    )
    print(summary.to_string(index=False))
    print(f"Exported Advanced class-scaling package to {output_dir}")


if __name__ == "__main__":
    main()
