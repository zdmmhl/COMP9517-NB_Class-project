"""Find difficult classes, common confusions, and representative examples."""

from __future__ import annotations

import pandas as pd


def find_top_confusions(
    confusion: pd.DataFrame,
    class_mapping: pd.DataFrame,
    limit: int = 20,
) -> pd.DataFrame:
    """Rank symmetric class pairs by A-to-B plus B-to-A errors."""
    names = class_mapping.set_index("class_index")["species_name"].to_dict()
    labels = [int(label) for label in confusion.index]
    rows = []

    for position, class_a in enumerate(labels):
        for class_b in labels[position + 1 :]:
            a_to_b = int(confusion.loc[class_a, class_b])
            b_to_a = int(confusion.loc[class_b, class_a])
            total = a_to_b + b_to_a
            if total == 0:
                continue
            rows.append(
                {
                    "class_a": class_a,
                    "class_b": class_b,
                    "species_a": names[class_a],
                    "species_b": names[class_b],
                    "a_to_b": a_to_b,
                    "b_to_a": b_to_a,
                    "total_confusions": total,
                }
            )

    columns = [
        "class_a",
        "class_b",
        "species_a",
        "species_b",
        "a_to_b",
        "b_to_a",
        "total_confusions",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["total_confusions", "a_to_b", "class_a", "class_b"],
            ascending=[False, False, True, True],
        )
        .head(limit)
        .reset_index(drop=True)
    )


def select_hardest_classes(
    per_class_metrics: pd.DataFrame,
    limit: int = 20,
) -> list[int]:
    """Select the classes with the lowest F1 and recall."""
    return (
        per_class_metrics.sort_values(
            ["f1", "recall", "class_index"],
            ascending=[True, True, True],
        )
        .head(limit)["class_index"]
        .astype(int)
        .tolist()
    )


def select_representative_examples(
    predictions: pd.DataFrame,
    limit: int = 12,
) -> dict[str, pd.DataFrame]:
    """Select deterministic success and failure groups for visual analysis."""
    ranked_columns = [f"pred_{rank}" for rank in range(1, 6)]
    frame = predictions.copy()
    frame["top1_correct"] = frame["true_label"] == frame["pred_1"]
    frame["top5_correct"] = frame.apply(
        lambda row: int(row["true_label"])
        in {int(row[column]) for column in ranked_columns},
        axis=1,
    )
    frame["score_margin"] = frame["score_1"] - frame["score_2"]

    successes = frame[frame["top1_correct"]].sort_values(
        ["score_1", "score_margin", "sample_id"],
        ascending=[False, False, True],
    )
    top5_only = frame[
        (~frame["top1_correct"]) & frame["top5_correct"]
    ].sort_values(
        ["score_1", "sample_id"],
        ascending=[False, True],
    )
    failures = frame[~frame["top5_correct"]].sort_values(
        ["score_1", "score_margin", "sample_id"],
        ascending=[False, False, True],
    )
    high_confidence_errors = frame[~frame["top1_correct"]].sort_values(
        ["score_1", "score_margin", "sample_id"],
        ascending=[False, False, True],
    )

    return {
        "successes": successes.head(limit).reset_index(drop=True),
        "top5_only": top5_only.head(limit).reset_index(drop=True),
        "failures": failures.head(limit).reset_index(drop=True),
        "high_confidence_errors": high_confidence_errors.head(limit).reset_index(
            drop=True
        ),
    }
