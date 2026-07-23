"""Adapt committed final-result records to the generic evaluation contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


TOP5_COLUMNS = [
    "top1_class_index",
    "top1_score",
    "top2_class_index",
    "top2_score",
    "top3_class_index",
    "top3_score",
    "top4_class_index",
    "top4_score",
    "top5_class_index",
    "top5_score",
]


class RecordedResultsError(ValueError):
    """Raised when committed result records cannot be adapted safely."""


def _single_row_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RecordedResultsError(f"Missing result file: {path}")
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RecordedResultsError(f"{path} must contain exactly one row")
    return frame.iloc[0].to_dict()


def prepare_shared_manifests(
    final_results_root: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, pd.DataFrame]:
    """Create class_mapping.csv and test.csv expected by artifact evaluation."""
    final_results_root = Path(final_results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_root = final_results_root / "reproducibility" / "data_splits"

    selected = pd.read_csv(split_root / "selected_classes.csv")
    required_mapping = {"class_index", "category_id", "name"}
    if not required_mapping.issubset(selected.columns):
        raise RecordedResultsError(
            "selected_classes.csv lacks class_index, category_id, or name"
        )
    class_mapping = selected[
        ["class_index", "category_id", "name"]
    ].rename(columns={"name": "species_name"})
    class_mapping_path = output_dir / "class_mapping.csv"
    class_mapping.to_csv(class_mapping_path, index=False)

    original_test = pd.read_csv(split_root / "test.csv")
    required_test = {"image_id", "file_name", "class_index"}
    if not required_test.issubset(original_test.columns):
        raise RecordedResultsError(
            "test.csv lacks image_id, file_name, or class_index"
        )
    test_manifest = pd.DataFrame(
        {
            "sample_id": original_test["image_id"].astype(str),
            "image_path": original_test["file_name"].astype(str),
            "true_label": original_test["class_index"].astype(int),
        }
    )
    test_manifest_path = output_dir / "test.csv"
    test_manifest.to_csv(test_manifest_path, index=False)
    return class_mapping_path, test_manifest_path, original_test


def prepare_top5_artifact(
    method_dir: str | Path,
    original_test: pd.DataFrame,
    output_dir: str | Path,
) -> str:
    """Convert one exported per-image Top-5 file into the shared contract."""
    method_dir = Path(method_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top5_path = method_dir / "test_predictions_top5.csv"
    predictions = pd.read_csv(top5_path)
    required = {"file_name", "true_class_index", *TOP5_COLUMNS}
    if not required.issubset(predictions.columns):
        missing = sorted(required - set(predictions.columns))
        raise RecordedResultsError(f"{top5_path} is missing columns: {missing}")

    lookup = original_test[["image_id", "file_name", "class_index"]].copy()
    if lookup["file_name"].duplicated().any():
        raise RecordedResultsError("Test manifest contains duplicate file_name values")
    merged = lookup.merge(
        predictions,
        on="file_name",
        how="left",
        validate="one_to_one",
    )
    if merged["top1_class_index"].isna().any():
        raise RecordedResultsError(
            f"{method_dir.name} does not contain predictions for every test image"
        )
    if not (
        merged["class_index"].astype(int)
        == merged["true_class_index"].astype(int)
    ).all():
        raise RecordedResultsError(
            f"{method_dir.name} true labels disagree with the test manifest"
        )

    standardized = pd.DataFrame(
        {
            "sample_id": merged["image_id"].astype(str),
            "image_path": merged["file_name"].astype(str),
            "true_label": merged["class_index"].astype(int),
        }
    )
    for rank in range(1, 6):
        standardized[f"pred_{rank}"] = merged[
            f"top{rank}_class_index"
        ].astype(int)
        standardized[f"score_{rank}"] = merged[
            f"top{rank}_score"
        ].astype(float)
    standardized.to_csv(output_dir / "predictions.csv", index=False)

    metrics = _single_row_csv(method_dir / "metrics.csv")
    metadata = {
        "method_name": str(metrics["method_name"]),
        "num_classes": int(metrics["num_classes"]),
        "random_seed": int(metrics["random_seed"]),
        "split_id": str(metrics["split_id"]),
        "score_type": "classifier_decision_score",
        "training_time_seconds": float(metrics["training_time_seconds"]),
        "inference_time_seconds": float(metrics["inference_time_seconds"]),
        "inference_num_images": int(metrics["num_test_samples"]),
        "device": "cpu",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(metrics["method_key"])


def exported_metrics_summary(method_dir: str | Path) -> dict[str, Any]:
    """Convert one metrics.csv row to the generic comparison JSON schema."""
    metrics = _single_row_csv(Path(method_dir) / "metrics.csv")

    def optional_float(name: str) -> float | None:
        value = metrics.get(name)
        if value is None or pd.isna(value) or value == "":
            return None
        return float(value)

    return {
        "method_name": str(metrics["method_name"]),
        "split_id": str(metrics["split_id"]),
        "score_type": "exported_summary",
        "device": (
            str(metrics["device"])
            if not pd.isna(metrics.get("device"))
            else "unknown"
        ),
        "random_seed": int(metrics["random_seed"]),
        "num_classes": int(metrics["num_classes"]),
        "num_samples": int(metrics["num_test_samples"]),
        "top1_accuracy": float(metrics["top1_accuracy"]),
        "top5_accuracy": float(metrics["top5_accuracy"]),
        "overall_accuracy": float(metrics["overall_accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "training_time_seconds": optional_float("training_time_seconds"),
        "inference_time_seconds": optional_float("inference_time_seconds"),
        "inference_num_images": int(metrics["num_test_samples"]),
        "inference_images_per_second": optional_float(
            "inference_images_per_second"
        ),
    }


def prepare_summary_only_entry(
    method_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Prepare comparison inputs when per-image Top-5 records are unavailable."""
    method_dir = Path(method_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = exported_metrics_summary(method_dir)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    per_class = method_dir / "per_class_metrics.csv"
    if not per_class.is_file():
        raise RecordedResultsError(f"Missing per-class metrics: {per_class}")
    shutil.copy2(per_class, output_dir / "per_class_metrics.csv")
    (output_dir / "SOURCE_NOTE.txt").write_text(
        "Comparison-only entry copied from committed aggregate records. "
        "Per-image Top-5 predictions were not available, so detailed "
        "artifact evaluation was not rerun for this method.\n",
        encoding="utf-8",
    )


def prepare_ablation_study(
    final_results_root: str | Path,
    output_dir: str | Path,
    *,
    study_name: str,
    factor: str,
    baseline_variant: str,
    variants: list[tuple[str, str, str]],
) -> None:
    """Build one study directory from exported method records."""
    final_results_root = Path(final_results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "study_name": study_name,
        "factor": factor,
        "baseline_variant": baseline_variant,
        "variant_order": [variant for variant, _, _ in variants],
        "display_names": {
            variant: display for variant, _, display in variants
        },
        "allowed_config_differences": [factor],
    }
    (output_dir / "study.json").write_text(
        json.dumps(spec, indent=2) + "\n",
        encoding="utf-8",
    )
    for variant, method_key, _ in variants:
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        source = final_results_root / "methods" / method_key
        shutil.copy2(
            source / "configuration.csv",
            variant_dir / "configuration.csv",
        )
        shutil.copy2(source / "metrics.csv", variant_dir / "metrics.csv")
