from __future__ import annotations

import csv

import pytest

from evaluation.top5_evidence import validate_top5_rows
from utils.serialization import save_predictions


def prediction_rows():
    return [
        {
            "file_name": "a.jpg",
            "true_class_index": "0",
            "pred_class_index": "0",
            "correct": "1",
            "pred_1": "0",
            "pred_2": "1",
            "pred_3": "2",
            "pred_4": "3",
            "pred_5": "4",
            "score_1": "0.5",
            "score_2": "0.2",
            "score_3": "0.15",
            "score_4": "0.1",
            "score_5": "0.05",
            "top5_correct": "1",
        },
        {
            "file_name": "b.jpg",
            "true_class_index": "4",
            "pred_class_index": "1",
            "correct": "0",
            "pred_1": "1",
            "pred_2": "4",
            "pred_3": "0",
            "pred_4": "2",
            "pred_5": "3",
            "score_1": "0.4",
            "score_2": "0.3",
            "score_3": "0.15",
            "score_4": "0.1",
            "score_5": "0.05",
            "top5_correct": "1",
        },
    ]


def test_independently_recomputes_top5_and_rescued_cases():
    validation = validate_top5_rows(
        prediction_rows(),
        recorded_top1=0.5,
        recorded_top5=1.0,
    )
    assert validation["recomputed_top1_accuracy"] == 0.5
    assert validation["recomputed_top5_accuracy"] == 1.0
    assert validation["top1_wrong_top5_correct_count"] == 1
    assert validation["true_label_rank_counts"] == {
        "1": 1,
        "2": 1,
        "3": 0,
        "4": 0,
        "5": 0,
    }


def test_rejects_recorded_top5_that_does_not_match_rows():
    with pytest.raises(ValueError, match="Top-5 mismatch"):
        validate_top5_rows(
            prediction_rows(),
            recorded_top1=0.5,
            recorded_top5=0.5,
        )


def test_save_predictions_preserves_top1_and_writes_ranked_top5(tmp_path):
    path = tmp_path / "test_predictions.csv"
    save_predictions(
        path,
        ["a.jpg"],
        [4],
        [1],
        [[1, 4, 0, 2, 3]],
        [[0.4, 0.3, 0.15, 0.1, 0.05]],
    )
    with path.open("r", encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["pred_class_index"] == "1"
    assert row["correct"] == "0"
    assert row["pred_2"] == "4"
    assert row["score_2"] == "0.3"
    assert row["top5_correct"] == "1"
