"""Validate and render one controlled ablation study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ablation import write_ablation_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one-factor-at-a-time runs and generate ablation tables "
            "and report figures."
        )
    )
    parser.add_argument(
        "--study-dir",
        type=Path,
        required=True,
        help="Directory containing study.json and variant run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to <study-dir>/report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.study_dir / "report"
    summary, deltas = write_ablation_outputs(args.study_dir, output_dir)
    print(summary.to_string(index=False))
    print("\nChanges from baseline:")
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
