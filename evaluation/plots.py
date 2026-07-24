import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix

from data.transforms import IMAGENET_MEAN, IMAGENET_STD


def plot_history(path, history):
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    if all("val_loss" in item for item in history):
        axes[0].plot(epochs, [item["val_loss"] for item in history], label="val")
    axes[0].set_title(
        "Loss" if all("val_loss" in item for item in history) else "Training Loss"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(
        epochs,
        [item["train_top1_accuracy"] for item in history],
        label="train top-1",
    )
    axes[1].plot(
        epochs, [item["val_top1_accuracy"] for item in history], label="val top-1"
    )
    axes[1].set_title("Top-1 Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[2].plot(
        epochs,
        [item["val_macro_f1"] for item in history],
        label="validation macro-F1",
    )
    axes[2].plot(
        epochs,
        [item["val_top5_accuracy"] for item in history],
        label="validation top-5",
    )
    peak = max(history, key=lambda item: item["val_macro_f1"])
    axes[2].axvline(
        peak["epoch"],
        color="black",
        linestyle="--",
        alpha=0.45,
        label=f"peak val F1 epoch {peak['epoch']}",
    )
    axes[2].set_title("Validation Metrics")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning_rate(path, history):
    epochs = [item["epoch"] for item in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        epochs, [item["head_lr"] for item in history], label="classification head"
    )
    if any("backbone_lr" in item for item in history):
        ax.plot(
            epochs,
            [item.get("backbone_lr", item["head_lr"]) for item in history],
            label="backbone",
        )
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(path, labels, predictions, num_classes, max_classes_to_plot=30):
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    np.save(path.with_suffix(".npy"), matrix)

    full_size = min(20, max(8, num_classes / 25))
    fig, ax = plt.subplots(figsize=(full_size, full_size))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax.set_title(f"Full Confusion Matrix ({num_classes} Classes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path.with_name(f"{path.stem}_full.png"), dpi=200)
    plt.close(fig)

    show_n = min(max_classes_to_plot, num_classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix[:show_n, :show_n], cmap="Blues")
    ax.set_title(f"Confusion Matrix First {show_n} Classes")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = tensor.cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def save_prediction_grid(path, model, dataset, device, num_images, use_amp):
    del use_amp
    model.eval()
    indices = list(range(min(len(dataset), max(num_images * 4, num_images))))
    images = []
    titles = []

    for index in indices:
        image, label, _ = dataset[index]
        prediction = int(model(image.unsqueeze(0).to(device)).argmax(dim=1).item())
        images.append(denormalize(image))
        status = "OK" if label == prediction else "WRONG"
        titles.append(f"T:{label} P:{prediction} {status}")
        if len(images) >= num_images:
            break

    cols = min(4, len(images))
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.asarray(axes).reshape(-1)
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    for axis in axes[len(images) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _prepare_output(path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _species_names(class_mapping):
    return {
        int(row.class_index): str(row.species_name)
        for row in class_mapping.itertuples(index=False)
    }


def plot_confusion_matrix_frame(
    confusion,
    class_mapping,
    output_path,
    *,
    selected_labels=None,
    normalize=True,
    title="Confusion Matrix",
):
    """Plot a full or selected matrix supplied as a labelled DataFrame."""
    matrix = confusion
    if selected_labels is not None:
        matrix = confusion.loc[selected_labels, selected_labels]

    values = matrix.astype(float)
    if normalize:
        row_totals = values.sum(axis=1).replace(0, 1)
        values = values.div(row_totals, axis=0)

    names = _species_names(class_mapping)
    tick_labels = [names[int(label)] for label in matrix.index]
    class_count = len(matrix)
    figure_size = min(max(8, class_count * 0.45), 22)
    show_ticks = class_count <= 50
    annotate = class_count <= 20

    fig, ax = plt.subplots(figsize=(figure_size, figure_size * 0.85))
    sns.heatmap(
        values,
        ax=ax,
        cmap="Blues",
        annot=annotate,
        fmt=".2f" if normalize else ".0f",
        xticklabels=tick_labels if show_ticks else False,
        yticklabels=tick_labels if show_ticks else False,
        cbar_kws={"label": "Row-normalized proportion" if normalize else "Count"},
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    if show_ticks:
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(_prepare_output(output_path), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history, output_path, method_name):
    """Plot loss, accuracy, validation metrics, and learning rate by epoch."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = history["epoch"]

    axes[0, 0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0, 0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["train_top1"], label="Train Top-1")
    axes[0, 1].plot(epochs, history["val_top1"], label="Validation Top-1")
    axes[0, 1].set_title("Top-1 Accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history["val_top5"], label="Validation Top-5")
    axes[1, 0].plot(
        epochs,
        history["val_macro_f1"],
        label="Validation Macro-F1",
    )
    axes[1, 0].set_title("Validation Metrics")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history["learning_rate"], label="Learning rate")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_yscale("log")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Training History: {method_name}")
    fig.tight_layout()
    fig.savefig(_prepare_output(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(summary, output_path):
    """Plot the main classification metrics for evaluated methods."""
    metric_labels = {
        "top1_accuracy": "Top-1",
        "top5_accuracy": "Top-5",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro-F1",
    }
    label_column = "method_key" if len(summary) > 6 else "method_name"
    long = summary[[label_column, *metric_labels]].melt(
        id_vars=label_column,
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(metric_labels)
    if len(summary) > 6:
        fig, ax = plt.subplots(figsize=(11, max(6, len(summary) * 0.65)))
        sns.barplot(
            data=long,
            y=label_column,
            x="value",
            hue="metric",
            orient="h",
            ax=ax,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_ylabel("Method")
    else:
        fig, ax = plt.subplots(figsize=(max(9, len(summary) * 2.5), 6))
        sns.barplot(
            data=long,
            x=label_column,
            y="value",
            hue="metric",
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Method")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=20)
    ax.set_title("Model Performance Comparison")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(_prepare_output(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_vs_performance(summary, output_path):
    """Plot inference runtime against Macro-F1."""
    valid = summary.dropna(
        subset=["inference_time_seconds", "macro_f1"]
    ).copy()
    valid = valid[valid["inference_time_seconds"] > 0]
    label_column = "method_key" if "method_key" in valid else "method_name"
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=valid,
        x="inference_time_seconds",
        y="macro_f1",
        hue=label_column,
        s=120,
        ax=ax,
    )
    if (
        not valid.empty
        and valid["inference_time_seconds"].max()
        / valid["inference_time_seconds"].min()
        > 10
    ):
        ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Inference time (seconds, log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Runtime versus Performance")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(_prepare_output(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1_distribution(per_class, output_path):
    """Compare per-class F1 distributions across methods."""
    method_count = per_class["method_name"].nunique()
    if method_count > 6:
        label_column = (
            "method_key" if "method_key" in per_class else "method_name"
        )
        fig, ax = plt.subplots(figsize=(11, max(6, method_count * 0.6)))
        sns.boxplot(data=per_class, y=label_column, x="f1", ax=ax)
        sns.stripplot(
            data=per_class,
            y=label_column,
            x="f1",
            color="black",
            alpha=0.2,
            size=2,
            ax=ax,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Per-class F1")
        ax.set_ylabel("Method")
    else:
        fig, ax = plt.subplots(figsize=(max(9, method_count * 2), 6))
        sns.boxplot(data=per_class, x="method_name", y="f1", ax=ax)
        sns.stripplot(
            data=per_class,
            x="method_name",
            y="f1",
            color="black",
            alpha=0.25,
            size=2,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Method")
        ax.set_ylabel("Per-class F1")
        ax.tick_params(axis="x", rotation=20)
    ax.set_title("Per-class F1 Distribution")
    fig.tight_layout()
    fig.savefig(_prepare_output(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_example_grid(
    examples,
    image_root,
    class_mapping,
    output_path,
    title,
):
    """Plot representative images with their true and Top-1 labels."""
    if examples.empty:
        return
    names = _species_names(class_mapping)
    image_root = Path(image_root)
    columns = min(4, len(examples))
    rows = math.ceil(len(examples) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows))
    axes = [axes] if rows * columns == 1 else list(axes.flat)

    for ax, row in zip(axes, examples.itertuples(index=False)):
        image_path = image_root / row.image_path
        if image_path.is_file():
            with Image.open(image_path) as image:
                ax.imshow(image.convert("RGB"))
        else:
            ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
        ax.set_title(
            f"True: {names[int(row.true_label)]}\n"
            f"Pred: {names[int(row.pred_1)]}\n"
            f"Score: {float(row.score_1):.3f}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes[len(examples) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(_prepare_output(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
