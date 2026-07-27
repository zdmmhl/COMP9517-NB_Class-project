"""Export and validate all controlled deep-learning ablation studies."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ablation import write_ablation_outputs
from evaluation.compact_experiment import export_deep_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "deep_ablations.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "final_results"
        / "inat500"
        / "ablations"
        / "deep_learning",
    )
    return parser.parse_args()


def reset_run_directory(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved_root not in resolved.parents:
        raise ValueError(f"Refusing to replace a directory outside {resolved_root}")
    if path.exists():
        shutil.rmtree(path)


def make_run_paths_relative(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
        fieldnames = list(rows[0]) if rows else []
    for row in rows:
        run_dir = Path(row["run_dir"]).resolve()
        row["run_dir"] = run_dir.relative_to(path.parent.resolve()).as_posix()
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_root = PROJECT_ROOT / config["results_root"]
    split_dir = PROJECT_ROOT / config["split_dir"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for study in config["studies"]:
        study_dir = args.output_dir / study["key"]
        study_dir.mkdir(parents=True, exist_ok=True)
        study_spec = {
            key: study[key]
            for key in [
                "study_name",
                "factor",
                "baseline_variant",
                "variant_order",
                "display_names",
                "allowed_config_differences",
            ]
        }
        study_spec["ignored_config_keys"] = study.get(
            "ignored_config_keys",
            [],
        )
        (study_dir / "study.json").write_text(
            json.dumps(study_spec, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for variant, run_key in study["variants"].items():
            variant_dir = study_dir / variant / f"seed_{config['seed']}"
            reset_run_directory(variant_dir, args.output_dir)
            export_deep_run(
                raw_root / run_key,
                variant_dir,
                split_dir,
                method_key=run_key,
                method_name=study["display_names"][variant],
                split_id="inat500_seed9517",
            )
        write_ablation_outputs(study_dir, study_dir)
        make_run_paths_relative(study_dir / "ablation_runs.csv")
        print(f"Exported {study['key']} to {study_dir}")


if __name__ == "__main__":
    main()
