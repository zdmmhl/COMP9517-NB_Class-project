"""Metrics, plots, model comparison, ensembling, and error analysis."""

from .ablation import (
    AblationStudy,
    AblationValidationError,
    load_ablation_study,
    write_ablation_outputs,
)
from .artifacts import (
    ArtifactValidationError,
    EvaluationArtifacts,
    load_evaluation_artifacts,
)
from .metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    metrics_from_probabilities,
)

__all__ = [
    "AblationStudy",
    "AblationValidationError",
    "ArtifactValidationError",
    "EvaluationArtifacts",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_per_class_metrics",
    "load_evaluation_artifacts",
    "load_ablation_study",
    "metrics_from_probabilities",
    "write_ablation_outputs",
]
