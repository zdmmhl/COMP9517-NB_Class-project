import csv
import json


def save_rows_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def save_predictions(
    path,
    paths,
    labels,
    predictions,
    top5_predictions=None,
    top5_scores=None,
):
    if (top5_predictions is None) != (top5_scores is None):
        raise ValueError("Top-5 predictions and scores must be provided together")
    include_top5 = top5_predictions is not None
    if include_top5 and not (
        len(paths)
        == len(labels)
        == len(predictions)
        == len(top5_predictions)
        == len(top5_scores)
    ):
        raise ValueError("Prediction arrays must have the same length")

    fieldnames = [
        "file_name",
        "true_class_index",
        "pred_class_index",
        "correct",
    ]
    if include_top5:
        fieldnames.extend(
            [f"pred_{rank}" for rank in range(1, 6)]
            + [f"score_{rank}" for rank in range(1, 6)]
            + ["top5_correct"]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (file_name, label, prediction) in enumerate(
            zip(paths, labels, predictions)
        ):
            row = {
                "file_name": file_name,
                "true_class_index": int(label),
                "pred_class_index": int(prediction),
                "correct": int(label == prediction),
            }
            if include_top5:
                labels_top5 = [int(value) for value in top5_predictions[index]]
                scores_top5 = [float(value) for value in top5_scores[index]]
                if len(labels_top5) != 5 or len(scores_top5) != 5:
                    raise ValueError("Every Top-5 prediction must contain five entries")
                row.update(
                    {
                        **{
                            f"pred_{rank}": labels_top5[rank - 1]
                            for rank in range(1, 6)
                        },
                        **{
                            f"score_{rank}": scores_top5[rank - 1]
                            for rank in range(1, 6)
                        },
                        "top5_correct": int(int(label) in labels_top5),
                    }
                )
            writer.writerow(row)
