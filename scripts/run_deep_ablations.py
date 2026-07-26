"""Run the controlled ResNet18 deep-learning ablation matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-factor-at-a-time ResNet18 ablations."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "deep_ablations.json",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional run keys to execute; dependencies are added automatically.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun experiments even when a complete metrics.json exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def option(name: str) -> str:
    return "--" + name.replace("_", "-")


def add_argument(command: list[str], key: str, value: object) -> None:
    if isinstance(value, bool):
        if key in {"channels_last", "cudnn_benchmark"}:
            command.append(option(key) if value else f"--no-{key.replace('_', '-')}")
        elif value:
            command.append(option(key))
        return
    command.extend([option(key), str(value)])


def merged_run(config: dict, overrides: dict) -> dict:
    values = dict(config["common"])
    values.update(overrides)
    return values


def build_training_command(
    config: dict,
    values: dict,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "train_deep.py"),
        "--data-root",
        str(config["data_root"]),
        "--split-dir",
        str(PROJECT_ROOT / config["split_dir"]),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(config["seed"]),
    ]
    for key, value in values.items():
        add_argument(command, key, value)
    return command


def run_command(command: list[str], output_dir: Path, dry_run: bool) -> None:
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    if dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "console.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def is_complete(output_dir: Path) -> bool:
    required = [
        "metrics.json",
        "history.csv",
        "test_predictions.csv",
        "training_curves.png",
        "test_confusion_matrix.png",
    ]
    return all((output_dir / name).is_file() for name in required)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    results_root = (PROJECT_ROOT / config["results_root"]).resolve()
    selected = set(args.only or config["training_runs"])

    evaluation_dependencies = {
        spec["source_run"]
        for key, spec in config.get("evaluation_runs", {}).items()
        if not args.only or key in selected
    }
    selected.update(evaluation_dependencies)

    for run_key, overrides in config["training_runs"].items():
        if run_key not in selected:
            continue
        output_dir = results_root / run_key
        if is_complete(output_dir) and not args.force:
            print(f"Skipping complete run: {run_key}")
            continue
        values = merged_run(config, overrides)
        command = build_training_command(config, values, output_dir)
        resume_checkpoint = output_dir / "last_checkpoint.pt"
        if resume_checkpoint.is_file() and not args.force:
            command.extend(["--resume-checkpoint", str(resume_checkpoint)])
        run_command(command, output_dir, args.dry_run)

    for run_key, evaluation in config.get("evaluation_runs", {}).items():
        if args.only and run_key not in selected:
            continue
        output_dir = results_root / run_key
        if is_complete(output_dir) and not args.force:
            print(f"Skipping complete run: {run_key}")
            continue
        source_dir = results_root / evaluation["source_run"]
        checkpoint = source_dir / "best_model.pt"
        if not args.dry_run and not checkpoint.is_file():
            raise FileNotFoundError(f"Missing source checkpoint: {checkpoint}")
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ["history.json", "history.csv", "training_state.json"]:
                source = source_dir / name
                if source.is_file():
                    shutil.copy2(source, output_dir / name)
        source_overrides = config["training_runs"][evaluation["source_run"]]
        values = merged_run(config, source_overrides)
        values.update(
            {key: value for key, value in evaluation.items() if key != "source_run"}
        )
        command = build_training_command(config, values, output_dir)
        command.extend(["--evaluate-checkpoint", str(checkpoint)])
        run_command(command, output_dir, args.dry_run)


if __name__ == "__main__":
    main()
