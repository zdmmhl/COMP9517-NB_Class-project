"""Tests for controlled ablation validation and report outputs."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from evaluation.ablation import (
    AblationValidationError,
    aggregate_ablation_metrics,
    load_ablation_study,
    validate_ablation_study,
    write_ablation_outputs,
)


FIXTURE = Path(__file__).parent / "fixtures" / "mock_ablation"


def test_loads_and_aggregates_controlled_ablation():
    study = load_ablation_study(FIXTURE)
    report = validate_ablation_study(study)
    summary = aggregate_ablation_metrics(study)

    assert len(study.runs) == 4
    assert report["factor"] == "augmentation"
    assert report["seeds"] == ["42", "43"]
    assert all(
        pair["changed_parameters"] == ["augmentation"]
        for pair in report["checked_pairs"]
    )
    assert summary["variant"].tolist() == ["basic", "strong"]
    assert summary["top1_accuracy_mean"].tolist() == pytest.approx([0.51, 0.57])
    assert summary["macro_f1_mean"].tolist() == pytest.approx([0.49, 0.55])


def test_rejects_an_uncontrolled_learning_rate_change(tmp_path):
    study_dir = tmp_path / "invalid_ablation"
    shutil.copytree(FIXTURE, study_dir)
    config_path = study_dir / "strong" / "seed_42" / "configuration.csv"
    config = pd.read_csv(config_path, dtype=str)
    config.loc[config["parameter"] == "lr", "value"] = "0.001"
    config.to_csv(config_path, index=False)

    with pytest.raises(
        AblationValidationError,
        match="uncontrolled parameters",
    ):
        load_ablation_study(study_dir)


def test_writes_report_ready_ablation_outputs(tmp_path):
    output_dir = tmp_path / "report"
    summary, deltas = write_ablation_outputs(FIXTURE, output_dir)

    assert summary["n_runs"].tolist() == [2, 2]
    assert deltas["delta_macro_f1"].tolist() == pytest.approx([0.0, 0.06])
    for name in (
        "ablation_runs.csv",
        "ablation_summary.csv",
        "ablation_deltas.csv",
        "validation_report.json",
        "ablation_metrics.png",
        "ablation_deltas.png",
        "training_curves.png",
    ):
        assert (output_dir / name).is_file()
