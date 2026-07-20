"""Integration test for evaluation and comparison output generation."""

from argparse import Namespace

import pandas as pd

from evaluation.comparison import write_comparison_outputs
from scripts.evaluate import run_evaluation


def test_complete_mock_evaluation_pipeline(mock_root, tmp_path):
    evaluation_root = tmp_path / "evaluation"
    model_output = evaluation_root / "mock_classifier"
    examples_output = tmp_path / "examples" / "mock_classifier"
    summary = run_evaluation(
        Namespace(
            prediction_dir=mock_root,
            class_mapping=mock_root / "class_mapping.csv",
            test_manifest=mock_root / "test.csv",
            history=mock_root / "history.csv",
            output_dir=model_output,
            image_root=mock_root,
            examples_dir=examples_output,
            top_confusions=20,
            subset_classes=6,
            example_count=12,
        )
    )

    assert summary["top1_accuracy"] == 0.4
    assert summary["top5_accuracy"] == 0.8
    for name in (
        "metrics.json",
        "per_class_metrics.csv",
        "confusion_matrix.csv",
        "top_confusions.csv",
        "confusion_matrix_full.png",
        "confusion_matrix_subset.png",
        "training_curves.png",
    ):
        assert (model_output / name).is_file()
    for name in (
        "successes.csv",
        "successes.png",
        "top5_only.csv",
        "top5_only.png",
        "failures.csv",
        "failures.png",
    ):
        assert (examples_output / name).is_file()

    comparison_output = tmp_path / "comparison"
    comparison, per_class = write_comparison_outputs(
        evaluation_root,
        comparison_output,
    )
    assert comparison["method_name"].tolist() == ["mock_classifier"]
    assert len(per_class) == 6
    assert pd.read_csv(comparison_output / "summary_metrics.csv").shape[0] == 1
    for name in (
        "runtime_comparison.csv",
        "model_comparison.png",
        "runtime_vs_performance.png",
        "per_class_f1_distribution.png",
    ):
        assert (comparison_output / name).is_file()
