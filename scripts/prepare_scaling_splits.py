"""Create and validate nested subsets for the Advanced scaling study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scaling_splits import build_scaling_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "class_scaling.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_scaling_splits(config, PROJECT_ROOT)
    for split_key, summary in report["splits"].items():
        rows = summary["rows"]
        print(
            f"{split_key}: {summary['num_classes']} classes, "
            f"train/val/test={rows['train']}/{rows['val']}/{rows['test']}"
        )
    print("Nested scaling splits validated successfully.")


if __name__ == "__main__":
    main()
