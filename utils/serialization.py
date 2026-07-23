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
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def save_predictions(path, paths, labels, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["file_name", "true_class_index", "pred_class_index", "correct"],
        )
        writer.writeheader()
        for file_name, label, prediction in zip(paths, labels, predictions):
            writer.writerow(
                {
                    "file_name": file_name,
                    "true_class_index": int(label),
                    "pred_class_index": int(prediction),
                    "correct": int(label == prediction),
                }
            )
