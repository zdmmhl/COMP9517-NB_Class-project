"""Load and validate model outputs consumed by the evaluation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = [
    "sample_id",
    "image_path",
    "true_label",
    *[f"pred_{rank}" for rank in range(1, 6)],
    *[f"score_{rank}" for rank in range(1, 6)],
]
TEST_COLUMNS = ["sample_id", "image_path", "true_label"]
CLASS_MAPPING_COLUMNS = ["class_index", "category_id", "species_name"]
HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_top1",
    "val_top1",
    "val_top5",
    "val_macro_f1",
    "learning_rate",
    "epoch_seconds",
]
METADATA_COLUMNS = [
    "method_name",
    "num_classes",
    "random_seed",
    "split_id",
    "score_type",
    "training_time_seconds",
    "inference_time_seconds",
    "inference_num_images",
]


class ArtifactValidationError(ValueError):
    """Raised when model outputs do not satisfy the agreed artifact contract."""


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Validated files required to evaluate one model."""

    predictions: pd.DataFrame
    metadata: dict[str, Any]
    class_mapping: pd.DataFrame
    test_manifest: pd.DataFrame
    history: pd.DataFrame | None = None

    @property
    def method_name(self) -> str:
        return str(self.metadata["method_name"])

    @property
    def labels(self) -> list[int]:
        return self.class_mapping["class_index"].astype(int).tolist()


def _read_csv(path: str | Path, description: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise ArtifactValidationError(f"Missing {description}: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ArtifactValidationError(
            f"Could not read {description} at {path}: {exc}"
        ) from exc


def _require_columns(
    frame: pd.DataFrame,
    required: list[str],
    description: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ArtifactValidationError(
            f"{description} is missing required columns: {missing}"
        )


def _coerce_integer_columns(
    frame: pd.DataFrame,
    columns: list[str],
    description: str,
) -> None:
    for column in columns:
        try:
            numeric = pd.to_numeric(frame[column], errors="raise")
        except Exception as exc:
            raise ArtifactValidationError(
                f"{description}.{column} must contain integers"
            ) from exc
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ArtifactValidationError(
                f"{description}.{column} must contain integers"
            )
        frame[column] = numeric.astype(int)


def _normalise_paths(frame: pd.DataFrame) -> None:
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["image_path"] = (
        frame["image_path"].astype(str).str.replace("\\", "/", regex=False)
    )


def _validate_class_mapping(class_mapping: pd.DataFrame) -> list[int]:
    _require_columns(class_mapping, CLASS_MAPPING_COLUMNS, "class_mapping.csv")
    _coerce_integer_columns(
        class_mapping,
        ["class_index"],
        "class_mapping.csv",
    )
    if class_mapping["class_index"].duplicated().any():
        raise ArtifactValidationError("class_mapping.csv has duplicate class_index values")
    if class_mapping["category_id"].duplicated().any():
        raise ArtifactValidationError("class_mapping.csv has duplicate category_id values")
    if class_mapping["species_name"].isna().any():
        raise ArtifactValidationError("class_mapping.csv has missing species_name values")

    class_mapping.sort_values("class_index", inplace=True, ignore_index=True)
    labels = class_mapping["class_index"].tolist()
    expected = list(range(len(labels)))
    if labels != expected:
        raise ArtifactValidationError(
            "class_index values must be contiguous and start at zero"
        )
    return labels


def _validate_metadata(
    metadata: dict[str, Any],
    num_classes: int,
    num_samples: int,
) -> None:
    missing = [key for key in METADATA_COLUMNS if key not in metadata]
    if missing:
        raise ArtifactValidationError(
            f"metadata.json is missing required fields: {missing}"
        )
    if not str(metadata["method_name"]).strip():
        raise ArtifactValidationError("metadata.method_name cannot be empty")
    if int(metadata["num_classes"]) != num_classes:
        raise ArtifactValidationError(
            "metadata.num_classes does not match class_mapping.csv"
        )
    if int(metadata["inference_num_images"]) != num_samples:
        raise ArtifactValidationError(
            "metadata.inference_num_images does not match predictions.csv"
        )
    for key in ("training_time_seconds", "inference_time_seconds"):
        if float(metadata[key]) < 0:
            raise ArtifactValidationError(f"metadata.{key} cannot be negative")


def _validate_predictions(
    predictions: pd.DataFrame,
    test_manifest: pd.DataFrame,
    labels: list[int],
) -> pd.DataFrame:
    _require_columns(predictions, PREDICTION_COLUMNS, "predictions.csv")
    _require_columns(test_manifest, TEST_COLUMNS, "test.csv")
    _normalise_paths(predictions)
    _normalise_paths(test_manifest)

    prediction_label_columns = [
        "true_label",
        *[f"pred_{rank}" for rank in range(1, 6)],
    ]
    _coerce_integer_columns(
        predictions,
        prediction_label_columns,
        "predictions.csv",
    )
    _coerce_integer_columns(test_manifest, ["true_label"], "test.csv")

    if predictions["sample_id"].duplicated().any():
        raise ArtifactValidationError("predictions.csv has duplicate sample_id values")
    if test_manifest["sample_id"].duplicated().any():
        raise ArtifactValidationError("test.csv has duplicate sample_id values")

    prediction_ids = set(predictions["sample_id"])
    test_ids = set(test_manifest["sample_id"])
    missing = sorted(test_ids - prediction_ids)
    unexpected = sorted(prediction_ids - test_ids)
    if missing or unexpected:
        raise ArtifactValidationError(
            "predictions.csv and test.csv contain different samples; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    predictions = predictions.set_index("sample_id").loc[
        test_manifest["sample_id"]
    ].reset_index()
    expected_truth = test_manifest["true_label"].to_numpy()
    if not np.array_equal(predictions["true_label"].to_numpy(), expected_truth):
        raise ArtifactValidationError(
            "predictions.csv true_label values do not match test.csv"
        )
    if not predictions["image_path"].equals(
        test_manifest["image_path"].reset_index(drop=True)
    ):
        raise ArtifactValidationError(
            "predictions.csv image_path values do not match test.csv"
        )

    valid_labels = set(labels)
    for column in prediction_label_columns:
        invalid = sorted(set(predictions[column]) - valid_labels)
        if invalid:
            raise ArtifactValidationError(
                f"predictions.csv {column} contains unknown labels: {invalid[:5]}"
            )

    prediction_matrix = predictions[
        [f"pred_{rank}" for rank in range(1, 6)]
    ].to_numpy()
    if any(len(set(row)) != 5 for row in prediction_matrix):
        raise ArtifactValidationError(
            "Each predictions.csv row must contain five distinct Top-5 labels"
        )

    score_columns = [f"score_{rank}" for rank in range(1, 6)]
    try:
        scores = predictions[score_columns].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=float)
    except Exception as exc:
        raise ArtifactValidationError(
            "predictions.csv score columns must be numeric"
        ) from exc
    if not np.isfinite(scores).all():
        raise ArtifactValidationError(
            "predictions.csv score columns must contain finite values"
        )
    if (np.diff(scores, axis=1) > 0).any():
        raise ArtifactValidationError(
            "Top-5 scores must be sorted from highest to lowest"
        )
    predictions[score_columns] = scores
    return predictions


def _validate_history(history: pd.DataFrame) -> pd.DataFrame:
    _require_columns(history, HISTORY_COLUMNS, "history.csv")
    try:
        history[HISTORY_COLUMNS] = history[HISTORY_COLUMNS].apply(
            pd.to_numeric,
            errors="raise",
        )
    except Exception as exc:
        raise ArtifactValidationError(
            "history.csv metric columns must be numeric"
        ) from exc
    if history["epoch"].duplicated().any():
        raise ArtifactValidationError("history.csv contains duplicate epochs")
    if not history["epoch"].is_monotonic_increasing:
        raise ArtifactValidationError("history.csv epochs must be increasing")
    return history


def load_evaluation_artifacts(
    predictions_path: str | Path,
    metadata_path: str | Path,
    class_mapping_path: str | Path,
    test_manifest_path: str | Path,
    history_path: str | Path | None = None,
) -> EvaluationArtifacts:
    """Load one method's outputs and enforce the shared evaluation contract."""
    predictions = _read_csv(predictions_path, "predictions.csv")
    class_mapping = _read_csv(class_mapping_path, "class_mapping.csv")
    test_manifest = _read_csv(test_manifest_path, "test.csv")

    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise ArtifactValidationError(f"Missing metadata.json: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactValidationError(
            f"Could not read metadata.json at {metadata_path}: {exc}"
        ) from exc

    labels = _validate_class_mapping(class_mapping)
    predictions = _validate_predictions(predictions, test_manifest, labels)
    _validate_metadata(metadata, len(labels), len(predictions))

    history = None
    if history_path is not None:
        history = _validate_history(_read_csv(history_path, "history.csv"))

    return EvaluationArtifacts(
        predictions=predictions,
        metadata=metadata,
        class_mapping=class_mapping,
        test_manifest=test_manifest,
        history=history,
    )
