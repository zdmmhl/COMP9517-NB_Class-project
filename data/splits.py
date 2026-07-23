"""Fixed split manifest readers, filters, and label remapping."""

import csv
from collections import Counter


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def filter_rows(rows, max_classes=None, max_per_class=None):
    """Keep a deterministic class prefix and optional number of rows per class."""
    if max_classes is None and max_per_class is None:
        return rows

    selected_classes = []
    seen = set()
    for row in rows:
        class_index = int(row["class_index"])
        if class_index not in seen:
            seen.add(class_index)
            selected_classes.append(class_index)
        if max_classes is not None and len(selected_classes) >= max_classes:
            break

    selected_set = set(selected_classes)
    counts = Counter()
    filtered = []
    for row in rows:
        class_index = int(row["class_index"])
        if class_index not in selected_set:
            continue
        if max_per_class is not None and counts[class_index] >= max_per_class:
            continue
        filtered.append(row)
        counts[class_index] += 1
    return filtered


def remap_rows(rows):
    """Map an arbitrary selected class subset to contiguous model labels."""
    original_classes = sorted({int(row["class_index"]) for row in rows})
    remap = {old: new for new, old in enumerate(original_classes)}
    remapped = []
    for row in rows:
        copy = dict(row)
        copy["original_class_index"] = int(copy["class_index"])
        copy["class_index"] = remap[int(copy["class_index"])]
        remapped.append(copy)
    return remapped, remap


def apply_remap(rows, remap):
    remapped = []
    for row in rows:
        old = int(row["class_index"])
        if old not in remap:
            continue
        copy = dict(row)
        copy["original_class_index"] = old
        copy["class_index"] = remap[old]
        remapped.append(copy)
    return remapped
