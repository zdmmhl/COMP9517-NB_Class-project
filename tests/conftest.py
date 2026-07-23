"""Shared pytest fixtures for the evaluation module."""

from pathlib import Path

import pytest

from evaluation.artifacts import load_evaluation_artifacts


MOCK_ROOT = Path(__file__).parent / "fixtures" / "mock_small"


@pytest.fixture
def mock_root() -> Path:
    return MOCK_ROOT


@pytest.fixture
def mock_artifacts():
    return load_evaluation_artifacts(
        predictions_path=MOCK_ROOT / "predictions.csv",
        metadata_path=MOCK_ROOT / "metadata.json",
        class_mapping_path=MOCK_ROOT / "class_mapping.csv",
        test_manifest_path=MOCK_ROOT / "test.csv",
        history_path=MOCK_ROOT / "history.csv",
    )
