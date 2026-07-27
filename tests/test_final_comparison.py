import csv
from pathlib import Path

import pytest

from evaluation.export_results import write_manifest
from scripts.evaluate_recorded_results import ABLATION_STUDIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = PROJECT_ROOT / "outputs" / "final_results" / "inat500"
COMPARISON_ROOT = FINAL_ROOT / "comparison"


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_final_summary_matches_each_method_metrics():
    summary = {
        row["method_key"]: row
        for row in read_rows(COMPARISON_ROOT / "summary_metrics.csv")
    }
    method_dirs = sorted(
        path for path in (FINAL_ROOT / "methods").iterdir() if path.is_dir()
    )
    assert set(summary) == {path.name for path in method_dirs}
    for method_dir in method_dirs:
        metrics = read_rows(method_dir / "metrics.csv")
        assert len(metrics) == 1
        assert metrics[0] == summary[method_dir.name]


def test_runtime_comparison_is_a_projection_of_summary():
    summary = {
        row["method_key"]: row
        for row in read_rows(COMPARISON_ROOT / "summary_metrics.csv")
    }
    runtime_rows = read_rows(COMPARISON_ROOT / "runtime_comparison.csv")
    assert set(summary) == {row["method_key"] for row in runtime_rows}
    for runtime in runtime_rows:
        source = summary[runtime["method_key"]]
        for field in runtime:
            assert runtime[field] == source[field]


def test_traditional_ablation_summaries_match_method_metrics():
    for definition in ABLATION_STUDIES:
        summary = {
            row["variant"]: row
            for row in read_rows(
                FINAL_ROOT
                / "ablations"
                / definition["key"]
                / "ablation_summary.csv"
            )
        }
        assert set(summary) == {
            variant for variant, _, _ in definition["variants"]
        }
        for variant, method_key, _ in definition["variants"]:
            metrics = read_rows(
                FINAL_ROOT / "methods" / method_key / "metrics.csv"
            )[0]
            for metric in ("top1_accuracy", "top5_accuracy", "macro_f1"):
                assert float(summary[variant][f"{metric}_mean"]) == pytest.approx(
                    float(metrics[metric]),
                    abs=1e-12,
                )


def test_deep_ablation_summaries_match_current_run_metrics():
    ablation_root = FINAL_ROOT / "ablations" / "deep_learning"
    for study_path in sorted(ablation_root.glob("*/study.json")):
        study_dir = study_path.parent
        summary = {
            row["variant"]: row
            for row in read_rows(study_dir / "ablation_summary.csv")
        }
        for variant, row in summary.items():
            metrics = read_rows(
                next((study_dir / variant).glob("seed_*/metrics.csv"))
            )[0]
            for metric in ("top1_accuracy", "top5_accuracy", "macro_f1"):
                assert float(row[f"{metric}_mean"]) == pytest.approx(
                    float(metrics[metric]),
                    abs=1e-12,
                )


def test_class_scaling_summary_matches_current_run_metrics():
    scaling_root = FINAL_ROOT / "advanced" / "class_scaling"
    summary = {
        row["run_key"]: row
        for row in read_rows(scaling_root / "scaling_summary.csv")
    }
    run_dirs = sorted(
        path for path in (scaling_root / "runs").iterdir() if path.is_dir()
    )
    assert set(summary) == {path.name for path in run_dirs}
    for run_dir in run_dirs:
        metrics = read_rows(run_dir / "metrics.csv")[0]
        for metric in ("top1_accuracy", "top5_accuracy", "macro_f1"):
            assert float(summary[run_dir.name][metric]) == pytest.approx(
                float(metrics[metric]),
                abs=1e-12,
            )


def test_manifest_uses_canonical_lf_sizes(tmp_path):
    text_path = tmp_path / "sample.csv"
    text_path.write_bytes(b"a,b\r\n1,2\r\n")
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n")

    write_manifest(tmp_path)

    manifest = {
        row["relative_path"]: int(row["size_bytes"])
        for row in read_rows(tmp_path / "artifact_manifest.csv")
    }
    assert manifest["sample.csv"] == len(b"a,b\n1,2\n")
    assert manifest["sample.png"] == len(b"\x89PNG\r\n")
