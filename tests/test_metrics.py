"""Tests for required classification metrics."""

import json

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    metrics_from_probabilities,
)


def test_metrics_match_expected_fixture(mock_artifacts, mock_root):
    expected = json.loads(
        (mock_root / "expected_metrics.json").read_text(encoding="utf-8")
    )
    actual = compute_classification_metrics(
        mock_artifacts.predictions,
        mock_artifacts.labels,
    )

    for key in (
        "top1_accuracy",
        "top5_accuracy",
        "overall_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
    ):
        assert actual[key] == pytest.approx(expected[key])
    assert actual["num_classes"] == 6
    assert actual["num_samples"] == 30


def test_per_class_metrics_are_balanced(mock_artifacts):
    per_class = compute_per_class_metrics(
        mock_artifacts.predictions,
        mock_artifacts.class_mapping,
    )
    assert len(per_class) == 6
    assert np.allclose(per_class["precision"], 0.4)
    assert np.allclose(per_class["recall"], 0.4)
    assert np.allclose(per_class["f1"], 0.4)
    assert per_class["support"].tolist() == [5] * 6


def test_confusion_matrix_matches_expected(mock_artifacts, mock_root):
    actual = compute_confusion_matrix(
        mock_artifacts.predictions,
        mock_artifacts.labels,
    )
    expected = pd.read_csv(
        mock_root / "expected_confusion_matrix.csv",
        index_col="true_label",
    )
    expected.columns = [int(column.removeprefix("pred_")) for column in expected]
    expected.index.name = None
    pd.testing.assert_frame_equal(actual, expected)


def test_existing_probability_metrics_remain_compatible():
    labels = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [0.9, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0.05, 0.7, 0.1, 0.05, 0.05, 0.05],
            [0.4, 0.1, 0.2, 0.1, 0.1, 0.1],
        ]
    )

    actual = metrics_from_probabilities(labels, probabilities)

    assert actual["top1_accuracy"] == pytest.approx(2 / 3)
    assert actual["overall_accuracy"] == pytest.approx(2 / 3)
    assert actual["top5_accuracy"] == 1.0
