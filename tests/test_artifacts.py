"""Tests for loading and validating evaluation artifacts."""

import pandas as pd
import pytest

from evaluation.artifacts import ArtifactValidationError, load_evaluation_artifacts


def test_load_mock_artifacts(mock_artifacts):
    assert mock_artifacts.method_name == "mock_classifier"
    assert mock_artifacts.labels == list(range(6))
    assert len(mock_artifacts.predictions) == 30
    assert len(mock_artifacts.test_manifest) == 30
    assert len(mock_artifacts.history) == 8
    assert mock_artifacts.predictions["sample_id"].is_unique


def test_rejects_duplicate_prediction_ids(mock_root, tmp_path):
    predictions = pd.read_csv(mock_root / "predictions.csv")
    predictions.loc[1, "sample_id"] = predictions.loc[0, "sample_id"]
    invalid_path = tmp_path / "predictions.csv"
    predictions.to_csv(invalid_path, index=False)

    with pytest.raises(ArtifactValidationError, match="duplicate sample_id"):
        load_evaluation_artifacts(
            predictions_path=invalid_path,
            metadata_path=mock_root / "metadata.json",
            class_mapping_path=mock_root / "class_mapping.csv",
            test_manifest_path=mock_root / "test.csv",
            history_path=mock_root / "history.csv",
        )


def test_rejects_unsorted_top5_scores(mock_root, tmp_path):
    predictions = pd.read_csv(mock_root / "predictions.csv")
    predictions.loc[0, "score_2"] = predictions.loc[0, "score_1"] + 1
    invalid_path = tmp_path / "predictions.csv"
    predictions.to_csv(invalid_path, index=False)

    with pytest.raises(ArtifactValidationError, match="highest to lowest"):
        load_evaluation_artifacts(
            predictions_path=invalid_path,
            metadata_path=mock_root / "metadata.json",
            class_mapping_path=mock_root / "class_mapping.csv",
            test_manifest_path=mock_root / "test.csv",
        )
