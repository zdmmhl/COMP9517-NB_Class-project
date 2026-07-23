import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def metrics_from_probabilities(labels, probabilities):
    labels = np.asarray(labels)
    predictions = probabilities.argmax(axis=1)
    top5 = np.argpartition(probabilities, -min(5, probabilities.shape[1]), axis=1)[
        :, -5:
    ]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(probabilities.shape[1])),
        average="macro",
        zero_division=0,
    )
    top1 = float(np.mean(predictions == labels))
    return {
        "top1_accuracy": top1,
        "overall_accuracy": top1,
        "top5_accuracy": float(np.mean(np.any(top5 == labels[:, None], axis=1))),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def compute_classification_metrics(predictions, labels):
    """Compute report metrics from a validated Top-5 prediction table."""
    labels = list(labels)
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_pred = predictions["pred_1"].to_numpy(dtype=int)
    top5 = predictions[
        [f"pred_{rank}" for rank in range(1, 6)]
    ].to_numpy(dtype=int)

    top1_accuracy = float(accuracy_score(y_true, y_pred))
    top5_accuracy = float(np.mean(np.any(top5 == y_true[:, None], axis=1)))
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    rounded = {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "overall_accuracy": top1_accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    return {
        "num_classes": len(labels),
        "num_samples": len(predictions),
        **{name: round(value, 12) for name, value in rounded.items()},
    }


def compute_per_class_metrics(predictions, class_mapping):
    """Return precision, recall, F1, support, and accuracy for every class."""
    labels = class_mapping["class_index"].astype(int).tolist()
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_pred = predictions["pred_1"].to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    output = class_mapping[
        ["class_index", "category_id", "species_name"]
    ].copy()
    output["precision"] = precision
    output["recall"] = recall
    output["f1"] = f1
    output["support"] = support.astype(int)
    output["class_accuracy"] = recall
    return output


def compute_confusion_matrix(predictions, labels):
    """Return a labelled confusion matrix with true labels on rows."""
    labels = list(labels)
    matrix = confusion_matrix(
        predictions["true_label"].to_numpy(dtype=int),
        predictions["pred_1"].to_numpy(dtype=int),
        labels=labels,
    )
    return pd.DataFrame(matrix, index=labels, columns=labels)
