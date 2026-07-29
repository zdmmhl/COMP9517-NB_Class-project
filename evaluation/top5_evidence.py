"""Build and validate per-image Top-5 evidence for classification runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from utils.serialization import save_rows_csv


TOP5_LABEL_COLUMNS = [f"pred_{rank}" for rank in range(1, 6)]
TOP5_SCORE_COLUMNS = [f"score_{rank}" for rank in range(1, 6)]
TOP5_COLUMNS = [
    "file_name",
    "true_class_index",
    "pred_class_index",
    "correct",
    *TOP5_LABEL_COLUMNS,
    *TOP5_SCORE_COLUMNS,
    "top5_correct",
]


def read_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def validate_top5_rows(
    rows: list[dict[str, str]],
    recorded_top1: float,
    recorded_top5: float,
    *,
    atol: float = 1e-6,
) -> dict:
    missing = [
        column
        for column in TOP5_COLUMNS
        if not rows or column not in rows[0]
    ]
    if missing:
        raise ValueError(f"Per-image Top-5 predictions are missing columns: {missing}")

    labels = np.asarray([int(row["true_class_index"]) for row in rows])
    top5 = np.asarray(
        [[int(row[column]) for column in TOP5_LABEL_COLUMNS] for row in rows]
    )
    scores = np.asarray(
        [[float(row[column]) for column in TOP5_SCORE_COLUMNS] for row in rows]
    )
    if any(len(set(values)) != 5 for values in top5):
        raise ValueError("Each image must have five distinct Top-5 labels")
    if not np.isfinite(scores).all():
        raise ValueError("Top-5 scores must be finite")
    if (np.diff(scores, axis=1) > 1e-12).any():
        raise ValueError("Top-5 scores must be sorted from highest to lowest")
    if not np.array_equal(
        top5[:, 0],
        np.asarray([int(row["pred_class_index"]) for row in rows]),
    ):
        raise ValueError("pred_class_index must equal pred_1")

    top1_correct = labels == top5[:, 0]
    top5_correct = np.any(top5 == labels[:, None], axis=1)
    stored_top5_correct = np.asarray([int(row["top5_correct"]) for row in rows])
    if not np.array_equal(stored_top5_correct, top5_correct.astype(int)):
        raise ValueError("Stored top5_correct flags do not match the ranked predictions")

    recomputed_top1 = float(np.mean(top1_correct))
    recomputed_top5 = float(np.mean(top5_correct))
    if not np.isclose(recomputed_top1, recorded_top1, atol=atol):
        raise ValueError(
            f"Top-1 mismatch: per-image={recomputed_top1}, recorded={recorded_top1}"
        )
    if not np.isclose(recomputed_top5, recorded_top5, atol=atol):
        raise ValueError(
            f"Top-5 mismatch: per-image={recomputed_top5}, recorded={recorded_top5}"
        )

    rescued = np.logical_and(~top1_correct, top5_correct)
    ranks = np.where(top5 == labels[:, None])[1] + 1
    rank_counts = {
        str(rank): int(np.sum(ranks == rank))
        for rank in range(1, 6)
    }
    return {
        "num_predictions": len(rows),
        "recomputed_top1_accuracy": recomputed_top1,
        "recorded_top1_accuracy": float(recorded_top1),
        "recomputed_top5_accuracy": recomputed_top5,
        "recorded_top5_accuracy": float(recorded_top5),
        "top1_wrong_top5_correct_count": int(np.sum(rescued)),
        "true_label_rank_counts": rank_counts,
        "metrics_match_predictions": True,
    }


def _class_name(class_rows: dict[int, dict[str, str]], label: int) -> str:
    row = class_rows.get(label, {})
    return (
        row.get("common_name")
        or row.get("name")
        or row.get("species_name")
        or str(label)
    )


def build_rescued_cases(
    rows: list[dict[str, str]],
    class_rows: dict[int, dict[str, str]],
) -> list[dict]:
    cases = []
    for row in rows:
        truth = int(row["true_class_index"])
        predictions = [int(row[column]) for column in TOP5_LABEL_COLUMNS]
        if predictions[0] == truth or truth not in predictions:
            continue
        true_rank = predictions.index(truth) + 1
        item = dict(row)
        item["true_rank"] = true_rank
        item["true_species"] = _class_name(class_rows, truth)
        for rank, prediction in enumerate(predictions, start=1):
            item[f"pred_{rank}_species"] = _class_name(class_rows, prediction)
        cases.append(item)
    return cases


def save_rescued_case_plot(
    path: str | Path,
    cases: list[dict],
    data_root: str | Path,
    *,
    max_images: int = 12,
) -> None:
    selected = []
    per_rank = max(1, max_images // 4)
    # Sample every true-label rank so rank-2 cases do not dominate the plot.
    for rank in range(2, 6):
        ranked_cases = sorted(
            (row for row in cases if int(row["true_rank"]) == rank),
            key=lambda row: -float(row[f"score_{rank}"]),
        )
        selected.extend(ranked_cases[:per_rank])
    if len(selected) < max_images:
        selected_paths = {row["file_name"] for row in selected}
        remaining = [
            row for row in cases if row["file_name"] not in selected_paths
        ]
        selected.extend(remaining[: max_images - len(selected)])
    if not selected:
        return
    columns = 4
    rows_count = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(16, 4.2 * rows_count))
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, row in zip(axes, selected):
        image_path = Path(data_root) / row["file_name"]
        with Image.open(image_path) as image:
            axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        axis.set_title(
            f"True: {row['true_species']}\n"
            f"Top-1: {row['pred_1_species']} | true rank: {row['true_rank']}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes[len(selected) :]:
        axis.axis("off")
    figure.suptitle("Top-1 wrong but Top-5 correct examples", fontsize=14)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_top5_evidence(
    output_dir: str | Path,
    rows: list[dict[str, str]],
    recorded_top1: float,
    recorded_top5: float,
    class_rows: dict[int, dict[str, str]],
    data_root: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_top5_rows(rows, recorded_top1, recorded_top5)
    save_rows_csv(output_dir / "test_predictions_top5.csv", rows, TOP5_COLUMNS)
    cases = build_rescued_cases(rows, class_rows)
    if len(cases) != validation["top1_wrong_top5_correct_count"]:
        raise ValueError("Rescued-case count does not match Top-5 validation")
    save_rows_csv(output_dir / "top1_wrong_top5_correct.csv", cases)
    save_rescued_case_plot(
        output_dir / "top1_wrong_top5_correct.png",
        cases,
        data_root,
    )
    (output_dir / "top5_prediction_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return validation
