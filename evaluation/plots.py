"""Plotting helpers for per-model evaluation and cross-model comparison."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


sns.set_theme(style="whitegrid")


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _species_names(class_mapping: pd.DataFrame) -> dict[int, str]:
    return {
        int(row.class_index): str(row.species_name)
        for row in class_mapping.itertuples(index=False)
    }


def plot_confusion_matrix(
    confusion: pd.DataFrame,
    class_mapping: pd.DataFrame,
    output_path: str | Path,
    *,
    selected_labels: list[int] | None = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Plot either the full matrix or a selected class subset."""
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


def plot_training_history(
    history: pd.DataFrame,
    output_path: str | Path,
    method_name: str,
) -> None:
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


def plot_metric_comparison(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot the main classification metrics for all evaluated methods."""
    metric_columns = [
        "top1_accuracy",
        "top5_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]
    long = summary[["method_name", *metric_columns]].melt(
        id_vars="method_name",
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(max(9, len(summary) * 2.5), 6))
    sns.barplot(data=long, x="method_name", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(_prepare_output(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_vs_performance(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot inference runtime against Macro-F1."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=summary,
        x="inference_time_seconds",
        y="macro_f1",
        hue="method_name",
        s=120,
        ax=ax,
    )
    for row in summary.itertuples(index=False):
        ax.annotate(
            row.method_name,
            (row.inference_time_seconds, row.macro_f1),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Inference time (seconds)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Runtime versus Performance")
    ax.legend().remove()
    fig.tight_layout()
    fig.savefig(_prepare_output(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1_distribution(
    per_class: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Compare the distribution of per-class F1 across methods."""
    fig, ax = plt.subplots(figsize=(max(9, per_class["method_name"].nunique() * 2), 6))
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
    ax.set_title("Per-class F1 Distribution")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(_prepare_output(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_example_grid(
    examples: pd.DataFrame,
    image_root: str | Path,
    class_mapping: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
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
