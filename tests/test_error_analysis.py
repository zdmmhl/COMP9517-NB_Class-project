"""Tests for confusion and representative-example analysis."""

from evaluation.error_analysis import (
    find_top_confusions,
    select_representative_examples,
)
from evaluation.metrics import compute_confusion_matrix


def test_finds_deliberate_confusion_pairs(mock_artifacts):
    confusion = compute_confusion_matrix(
        mock_artifacts.predictions,
        mock_artifacts.labels,
    )
    pairs = find_top_confusions(
        confusion,
        mock_artifacts.class_mapping,
        limit=3,
    )
    assert {
        tuple(sorted((int(row.class_a), int(row.class_b))))
        for row in pairs.itertuples(index=False)
    } == {(0, 1), (2, 3), (4, 5)}
    assert pairs["total_confusions"].tolist() == [6, 6, 6]


def test_selects_expected_example_groups(mock_artifacts):
    groups = select_representative_examples(
        mock_artifacts.predictions,
        limit=30,
    )
    assert len(groups["successes"]) == 12
    assert len(groups["top5_only"]) == 12
    assert len(groups["failures"]) == 6
    assert len(groups["high_confidence_errors"]) == 18
