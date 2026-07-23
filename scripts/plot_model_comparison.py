import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    hog = load_metrics(Path("outputs/hog_svm_full/metrics.json"))["splits"]["test"]
    simple = load_metrics(Path("outputs/simple_cnn_full/metrics.json"))["test"]
    resnet = load_metrics(Path("outputs/resnet18_pretrained_full/metrics.json"))["test"]
    optimized = load_metrics(Path("outputs/resnet50_pretrained_optimized_full/metrics.json"))["test"]
    convnext = load_metrics(Path("outputs/convnext_tiny_mixup_full/metrics.json"))["test"]
    ensemble = load_metrics(Path("outputs/deep_ensemble/metrics.json"))["test"]

    methods = [
        "HOG+SVM",
        "Scratch CNN",
        "Pretrained\nResNet18",
        "Optimized\nResNet50",
        "ConvNeXt\n+ MixUp",
        "Validation-selected\nEnsemble",
    ]
    top1 = [hog["top1_accuracy"], simple["top1_accuracy"], resnet["top1_accuracy"], optimized["top1_accuracy"], convnext["top1_accuracy"], ensemble["top1_accuracy"]]
    top5 = [hog["top5_accuracy"], simple["top5_accuracy"], resnet["top5_accuracy"], optimized["top5_accuracy"], convnext["top5_accuracy"], ensemble["top5_accuracy"]]
    f1 = [hog["macro_f1"], simple["macro_f1"], resnet["macro_f1"], optimized["macro_f1"], convnext["macro_f1"], ensemble["macro_f1"]]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width, top1, width, label="Top-1")
    bars2 = ax.bar(x, top5, width, label="Top-5")
    bars3 = ax.bar(x + width, f1, width, label="Macro-F1")

    ax.set_ylabel("Score")
    ax.set_title("500-Class iNaturalist Test Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, max(top5) * 1.20)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    summary = {
        "methods": {
            "HOG+SVM": {"top1": top1[0], "top5": top5[0], "macro_f1": f1[0]},
            "Scratch CNN": {"top1": top1[1], "top5": top5[1], "macro_f1": f1[1]},
            "Pretrained ResNet18": {"top1": top1[2], "top5": top5[2], "macro_f1": f1[2]},
            "Optimized ResNet50": {"top1": top1[3], "top5": top5[3], "macro_f1": f1[3]},
            "ConvNeXt + MixUp": {"top1": top1[4], "top5": top5[4], "macro_f1": f1[4]},
            "Validation-selected Ensemble": {"top1": top1[5], "top5": top5[5], "macro_f1": f1[5]},
        }
    }
    with (output_dir / "model_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
