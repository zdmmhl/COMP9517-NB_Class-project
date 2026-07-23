import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class SplitImageDataset(Dataset):
    def __init__(self, rows, data_root, transform):
        self.rows = rows
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_root / row["file_name"]
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            image = self.transform(img)
        label = int(row["class_index"])
        return image, label, row["file_name"]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def filter_rows(rows, max_classes=None, max_per_class=None):
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size, augmentation):
    train_steps = [
        transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if augmentation == "strong":
        train_steps.append(transforms.RandAugment(num_ops=2, magnitude=7))
    else:
        train_steps.append(
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if augmentation == "strong":
        train_steps.append(
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.15), ratio=(0.5, 2.0))
        )
    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_model(model_name, num_classes):
    if model_name == "simple-cnn":
        return SimpleCNN(num_classes), "random"
    if model_name == "resnet18-pretrained":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, "imagenet"
    if model_name == "resnet18-scratch":
        model = models.resnet18(weights=None, num_classes=num_classes)
        return model, "random"
    if model_name == "resnet50-pretrained":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, "imagenet"
    if model_name == "efficientnet-b0-pretrained":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model, "imagenet"
    if model_name == "convnext-tiny-pretrained":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model, "imagenet"
    raise ValueError(f"Unknown model: {model_name}")


def classifier_parameters(model, model_name):
    if model_name.startswith("resnet"):
        return list(model.fc.parameters())
    if model_name.startswith("efficientnet"):
        return list(model.classifier.parameters())
    if model_name.startswith("convnext"):
        return list(model.classifier.parameters())
    return list(model.classifier.parameters())


def build_optimizer(model, model_name, lr, weight_decay, backbone_lr_multiplier):
    head_params = classifier_parameters(model, model_name)
    head_ids = {id(parameter) for parameter in head_params}
    backbone_params = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    if model_name.endswith("pretrained") and backbone_lr_multiplier < 1.0:
        parameter_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_multiplier},
            {"params": head_params, "lr": lr},
        ]
    else:
        parameter_groups = [{"params": model.parameters(), "lr": lr}]
    return optim.AdamW(parameter_groups, weight_decay=weight_decay)


def topk_accuracy(logits, labels, k):
    k = min(k, logits.shape[1])
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(labels.view(-1, 1)).any(dim=1)
    return correct.float().mean().item()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_amp,
    grad_clip,
    mixup_alpha,
):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    running_loss = 0.0
    top1_sum = 0.0
    total = 0
    started = time.perf_counter()

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mixed_labels = labels
        mixup_lambda = 1.0
        if mixup_alpha > 0:
            mixup_lambda = float(np.random.beta(mixup_alpha, mixup_alpha))
            permutation = torch.randperm(images.size(0), device=device)
            images = mixup_lambda * images + (1.0 - mixup_lambda) * images[permutation]
            mixed_labels = labels[permutation]
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = mixup_lambda * criterion(logits, labels)
            if mixup_alpha > 0:
                loss = loss + (1.0 - mixup_lambda) * criterion(logits, mixed_labels)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        batch_top1 = mixup_lambda * topk_accuracy(logits.detach(), labels, 1)
        if mixup_alpha > 0:
            batch_top1 += (1.0 - mixup_lambda) * topk_accuracy(
                logits.detach(), mixed_labels, 1
            )
        top1_sum += batch_top1 * batch_size
        total += batch_size

    return {
        "loss": running_loss / total,
        "top1_accuracy": top1_sum / total,
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, use_amp, tta=False):
    model.eval()
    running_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    total = 0
    all_labels = []
    all_preds = []
    all_paths = []
    started = time.perf_counter()

    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            if tta:
                flipped_logits = model(torch.flip(images, dims=[3]))
                logits = (logits + flipped_logits) / 2
            loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        top1_sum += topk_accuracy(logits, labels, 1) * batch_size
        top5_sum += topk_accuracy(logits, labels, 5) * batch_size
        total += batch_size
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_paths.extend(paths)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    return {
        "loss": running_loss / total,
        "top1_accuracy": top1_sum / total,
        "top5_accuracy": top5_sum / total,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "seconds": time.perf_counter() - started,
        "labels": all_labels,
        "preds": all_preds,
        "paths": all_paths,
    }


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_predictions(path, paths, labels, preds):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "true_class_index", "pred_class_index", "correct"],
        )
        writer.writeheader()
        for file_name, label, pred in zip(paths, labels, preds):
            writer.writerow(
                {
                    "file_name": file_name,
                    "true_class_index": int(label),
                    "pred_class_index": int(pred),
                    "correct": int(label == pred),
                }
            )


def plot_history(path, history):
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    if all("val_loss" in item for item in history):
        axes[0].plot(epochs, [item["val_loss"] for item in history], label="val")
    axes[0].set_title("Loss" if all("val_loss" in item for item in history) else "Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [item["train_top1_accuracy"] for item in history], label="train top-1")
    axes[1].plot(epochs, [item["val_top1_accuracy"] for item in history], label="val top-1")
    axes[1].set_title("Top-1 Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning_rate(path, history):
    epochs = [item["epoch"] for item in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [item["head_lr"] for item in history], label="classification head")
    if any("backbone_lr" in item for item in history):
        ax.plot(epochs, [item.get("backbone_lr", item["head_lr"]) for item in history], label="backbone")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(path, labels, preds, num_classes, max_classes_to_plot=30):
    matrix = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    np.save(path.with_suffix(".npy"), matrix)
    show_n = min(max_classes_to_plot, num_classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix[:show_n, :show_n], cmap="Blues")
    ax.set_title(f"Confusion Matrix First {show_n} Classes")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = tensor.cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def save_prediction_grid(path, model, dataset, device, num_images, use_amp):
    model.eval()
    indices = list(range(min(len(dataset), max(num_images * 4, num_images))))
    images = []
    titles = []

    for index in indices:
        image, label, _ = dataset[index]
        logits = model(image.unsqueeze(0).to(device))
        pred = int(logits.argmax(dim=1).item())
        images.append(denormalize(image))
        titles.append(f"T:{label} P:{pred} {'OK' if label == pred else 'WRONG'}")
        if len(images) >= num_images:
            break

    cols = min(4, len(images))
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.asarray(axes).reshape(-1)
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train scratch CNN or pretrained ResNet on iNat splits.")
    parser.add_argument(
        "--model",
        choices=[
            "simple-cnn",
            "resnet18-pretrained",
            "resnet18-scratch",
            "resnet50-pretrained",
            "efficientnet-b0-pretrained",
            "convnext-tiny-pretrained",
        ],
        required=True,
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/inat2021"))
    parser.add_argument("--split-dir", type=Path, default=Path("data_splits"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--augmentation", choices=["basic", "strong"], default="basic")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-eval-per-class", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--sample-images", type=int, default=12)
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        default=None,
        help="Skip training and evaluate an existing checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = filter_rows(
        read_rows(args.split_dir / "train.csv"),
        args.max_classes,
        args.max_train_per_class,
    )
    train_rows, remap = remap_rows(train_rows)
    val_rows = apply_remap(
        filter_rows(read_rows(args.split_dir / "val.csv"), args.max_classes, args.max_eval_per_class),
        remap,
    )
    test_rows = apply_remap(
        filter_rows(read_rows(args.split_dir / "test.csv"), args.max_classes, args.max_eval_per_class),
        remap,
    )

    num_classes = len(remap)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    train_transform, eval_transform = build_transforms(args.image_size, args.augmentation)
    train_dataset = SplitImageDataset(train_rows, args.data_root, train_transform)
    val_dataset = SplitImageDataset(val_rows, args.data_root, eval_transform)
    test_dataset = SplitImageDataset(test_rows, args.data_root, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    model, init = build_model(args.model, num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(
        model,
        args.model,
        args.lr,
        args.weight_decay,
        args.backbone_lr_multiplier,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * args.min_lr_ratio,
        )

    run_mode = "evaluation" if args.evaluate_checkpoint else "training"
    print(f"Deep learning run: {args.model} ({run_mode})")
    print(f"  init: {init}")
    print(f"  device: {device}")
    print(f"  classes: {num_classes}")
    print(f"  train/val/test rows: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"  epochs: {args.epochs}, batch_size: {args.batch_size}, amp: {use_amp}")
    print(
        f"  augmentation: {args.augmentation}, label_smoothing: {args.label_smoothing}, "
        f"mixup_alpha: {args.mixup_alpha}, scheduler: {args.scheduler}, tta: {args.tta}"
    )

    history = []
    best_val_f1 = -1.0
    best_path = args.evaluate_checkpoint or (args.output_dir / "best_model.pt")
    total_started = time.perf_counter()

    if args.evaluate_checkpoint:
        history_path = args.output_dir / "history.json"
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as f:
                history = json.load(f)
    else:
        for epoch in range(1, args.epochs + 1):
            current_lrs = [group["lr"] for group in optimizer.param_groups]
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                use_amp,
                args.grad_clip,
                args.mixup_alpha,
            )
            val_metrics = evaluate(model, val_loader, criterion, device, num_classes, use_amp, args.tta)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_top1_accuracy": train_metrics["top1_accuracy"],
                "train_seconds": train_metrics["seconds"],
                "val_loss": val_metrics["loss"],
                "val_top1_accuracy": val_metrics["top1_accuracy"],
                "val_top5_accuracy": val_metrics["top5_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_seconds": val_metrics["seconds"],
                "head_lr": current_lrs[-1],
            }
            if len(current_lrs) > 1:
                row["backbone_lr"] = current_lrs[0]
            history.append(row)
            save_json(args.output_dir / "history.json", history)
            print(
                f"epoch {epoch}/{args.epochs}: "
                f"train_loss={row['train_loss']:.4f}, train_top1={row['train_top1_accuracy']:.4f}, "
                f"val_top1={row['val_top1_accuracy']:.4f}, val_top5={row['val_top5_accuracy']:.4f}, "
                f"val_f1={row['val_macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model": args.model,
                        "num_classes": num_classes,
                        "class_remap": {str(k): v for k, v in remap.items()},
                        "epoch": epoch,
                        "val_macro_f1": best_val_f1,
                    },
                    best_path,
                )
            if scheduler is not None:
                scheduler.step()

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes, use_amp, args.tta)
    train_eval_metrics = evaluate(model, train_loader, criterion, device, num_classes, use_amp)

    labels = list(range(num_classes))
    report = classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        labels=labels,
        zero_division=0,
        output_dict=True,
    )

    save_json(args.output_dir / "history.json", history)
    save_json(args.output_dir / "test_classification_report.json", report)
    save_predictions(
        args.output_dir / "test_predictions.csv",
        test_metrics["paths"],
        test_metrics["labels"],
        test_metrics["preds"],
    )
    if history:
        plot_history(args.output_dir / "training_curves.png", history)
        plot_learning_rate(args.output_dir / "learning_rate.png", history)
    plot_confusion_matrix(
        args.output_dir / "test_confusion_matrix.png",
        test_metrics["labels"],
        test_metrics["preds"],
        num_classes,
    )
    save_prediction_grid(
        args.output_dir / "test_prediction_examples.png",
        model,
        test_dataset,
        device,
        args.sample_images,
        use_amp,
    )

    metrics = {
        "method": args.model,
        "run_mode": run_mode,
        "initialization": init,
        "device": str(device),
        "num_classes": num_classes,
        "data_root": str(args.data_root).replace("\\", "/"),
        "params": {
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "backbone_lr_multiplier": args.backbone_lr_multiplier,
            "scheduler": args.scheduler,
            "min_lr_ratio": args.min_lr_ratio,
            "label_smoothing": args.label_smoothing,
            "augmentation": args.augmentation,
            "grad_clip": args.grad_clip,
            "mixup_alpha": args.mixup_alpha,
            "tta": args.tta,
            "seed": args.seed,
            "max_classes": args.max_classes,
            "max_train_per_class": args.max_train_per_class,
            "max_eval_per_class": args.max_eval_per_class,
        },
        "rows": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "best_epoch": checkpoint["epoch"],
        "total_seconds": time.perf_counter() - total_started,
        "train_eval": {
            key: value
            for key, value in train_eval_metrics.items()
            if key not in {"labels", "preds", "paths"}
        },
        "test": {
            key: value
            for key, value in test_metrics.items()
            if key not in {"labels", "preds", "paths"}
        },
    }
    save_json(args.output_dir / "metrics.json", metrics)
    print("Final test metrics:")
    print(
        f"  top1={metrics['test']['top1_accuracy']:.4f}, "
        f"top5={metrics['test']['top5_accuracy']:.4f}, "
        f"macro_f1={metrics['test']['macro_f1']:.4f}"
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
