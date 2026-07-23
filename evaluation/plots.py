import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from data.transforms import IMAGENET_MEAN, IMAGENET_STD


def plot_history(path, history):
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
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
