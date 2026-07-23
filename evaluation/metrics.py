import numpy as np
from sklearn.metrics import precision_recall_fscore_support


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
