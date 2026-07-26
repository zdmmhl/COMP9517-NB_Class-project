"""Run the non-base subsets in the Advanced class-scaling study."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def option(name: str) -> str:
    return "--" + name.replace("_", "-")


def add_argument(command: list[str], key: str, value: object) -> None:
    if isinstance(value, bool):
        if key in {"channels_last", "cudnn_benchmark"}:
            command.append(option(key) if value else f"--no-{key.replace('_', '-')}")
        elif value:
            command.append(option(key))
    else:
        command.extend([option(key), str(value)])


def complete(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in [
            "metrics.json",
            "history.csv",
            "test_predictions.csv",
            "training_curves.png",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "class_scaling.json",
    )
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = set(args.only or config["split_specs"])
    selected.discard("classes_500")

    for run_key in config["split_specs"]:
        if run_key not in selected or run_key == "classes_500":
            continue
        output_dir = PROJECT_ROOT / config["results_root"] / run_key
        if complete(output_dir) and not args.force:
            print(f"Skipping complete run: {run_key}")
            continue
        command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / "train_deep.py"),
            "--data-root",
            config["data_root"],
            "--split-dir",
            str(PROJECT_ROOT / config["split_root"] / run_key),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(config["seed"]),
        ]
        for key, value in config["common_training"].items():
            add_argument(command, key, value)
        resume_checkpoint = output_dir / "last_checkpoint.pt"
        if resume_checkpoint.is_file() and not args.force:
            command.extend(["--resume-checkpoint", str(resume_checkpoint)])
        print(" ".join(f'"{part}"' if " " in part else part for part in command))
        if args.dry_run:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "console.log").open("a", encoding="utf-8") as log:
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
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


if __name__ == "__main__":
    main()
