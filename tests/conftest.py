"""Shared pytest fixtures for the evaluation module."""

from pathlib import Path

import pytest

from evaluation.artifacts import load_evaluation_artifacts
from tests.fixtures.mock_small.generate_fixture import generate_fixture


@pytest.fixture(scope="session")
def mock_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("mock_small")
    generate_fixture(root)
    return root


@pytest.fixture
def mock_artifacts(mock_root):
    return load_evaluation_artifacts(
        predictions_path=mock_root / "predictions.csv",
        metadata_path=mock_root / "metadata.json",
        class_mapping_path=mock_root / "class_mapping.csv",
        test_manifest_path=mock_root / "test.csv",
        history_path=mock_root / "history.csv",
    )
