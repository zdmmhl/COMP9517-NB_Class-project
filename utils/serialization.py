import csv
import json


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
