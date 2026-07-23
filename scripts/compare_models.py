"""Generate report-ready comparisons from evaluated model directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.comparison import write_comparison_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare every evaluated model under one directory."
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "runs"
            / "mock_evaluation"
            / "methods"
        ),
        help="Directory containing one evaluated subdirectory per method.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "runs"
            / "mock_evaluation"
            / "comparison"
        ),
        help="Directory for comparison tables and figures.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, _ = write_comparison_outputs(
        args.evaluation_root,
        args.output_dir,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
