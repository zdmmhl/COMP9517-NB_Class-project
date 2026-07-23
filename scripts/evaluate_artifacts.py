"""Evaluate one method from its saved predictions and metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.artifacts import load_evaluation_artifacts
from evaluation.error_analysis import (
    find_top_confusions,
    select_hardest_classes,
    select_representative_examples,
)
from evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
)
from evaluation.plots import (
    plot_confusion_matrix_frame,
    plot_example_grid,
    plot_training_history,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one model from the shared prediction artifacts."
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Directory containing predictions.csv and metadata.json.",
    )
    parser.add_argument(
        "--class-mapping",
        type=Path,
        required=True,
        help="Shared class_mapping.csv.",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        required=True,
        help="Shared test.csv.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="Optional deep-learning history.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-model metrics, tables, and plots.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Root used to resolve image_path values for example grids.",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        help="Optional output directory for example CSV files and image grids.",
    )
    parser.add_argument("--top-confusions", type=int, default=20)
    parser.add_argument("--subset-classes", type=int, default=20)
    parser.add_argument("--example-count", type=int, default=12)
    return parser


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    prediction_dir = args.prediction_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_evaluation_artifacts(
        predictions_path=prediction_dir / "predictions.csv",
        metadata_path=prediction_dir / "metadata.json",
        class_mapping_path=args.class_mapping,
        test_manifest_path=args.test_manifest,
        history_path=args.history,
    )
    metrics = compute_classification_metrics(
        artifacts.predictions,
        artifacts.labels,
    )
    inference_seconds = float(artifacts.metadata["inference_time_seconds"])
    inference_count = int(artifacts.metadata["inference_num_images"])
    summary = {
        "method_name": artifacts.method_name,
        "split_id": str(artifacts.metadata["split_id"]),
        "score_type": str(artifacts.metadata["score_type"]),
        "device": str(artifacts.metadata.get("device", "unknown")),
        "random_seed": int(artifacts.metadata["random_seed"]),
        **metrics,
        "training_time_seconds": float(
            artifacts.metadata["training_time_seconds"]
        ),
        "inference_time_seconds": inference_seconds,
        "inference_num_images": inference_count,
        "inference_images_per_second": (
            inference_count / inference_seconds if inference_seconds > 0 else None
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    per_class = compute_per_class_metrics(
        artifacts.predictions,
        artifacts.class_mapping,
    )
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)

    confusion = compute_confusion_matrix(
        artifacts.predictions,
        artifacts.labels,
    )
    confusion.to_csv(output_dir / "confusion_matrix.csv", index_label="true_label")
    top_confusions = find_top_confusions(
        confusion,
        artifacts.class_mapping,
        limit=args.top_confusions,
    )
    top_confusions.to_csv(output_dir / "top_confusions.csv", index=False)

    plot_confusion_matrix_frame(
        confusion,
        artifacts.class_mapping,
        output_dir / "confusion_matrix_full.png",
        normalize=True,
        title=f"Confusion Matrix: {artifacts.method_name}",
    )
    subset_size = min(args.subset_classes, len(artifacts.labels))
    hardest_labels = select_hardest_classes(per_class, limit=subset_size)
    plot_confusion_matrix_frame(
        confusion,
        artifacts.class_mapping,
        output_dir / "confusion_matrix_subset.png",
        selected_labels=hardest_labels,
        normalize=False,
        title=f"Hardest-class Confusion Matrix: {artifacts.method_name}",
    )

    if artifacts.history is not None:
        plot_training_history(
            artifacts.history,
            output_dir / "training_curves.png",
            artifacts.method_name,
        )

    examples = select_representative_examples(
        artifacts.predictions,
        limit=args.example_count,
    )
    examples_dir = args.examples_dir or (
        PROJECT_ROOT / "outputs" / "examples" / artifacts.method_name
    )
    examples_dir.mkdir(parents=True, exist_ok=True)
    for group_name, group in examples.items():
        group.to_csv(examples_dir / f"{group_name}.csv", index=False)
        if args.image_root is not None:
            plot_example_grid(
                group,
                args.image_root,
                artifacts.class_mapping,
                examples_dir / f"{group_name}.png",
                title=f"{group_name.replace('_', ' ').title()}: {artifacts.method_name}",
            )

    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_evaluation(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
