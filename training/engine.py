import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support


def topk_accuracy(logits, labels, k):
    k = min(k, logits.shape[1])
    _, predictions = logits.topk(k, dim=1)
    correct = predictions.eq(labels.view(-1, 1)).any(dim=1)
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
    scaler=None,
    channels_last=False,
):
    model.train()
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    running_loss = 0.0
    top1_sum = 0.0
    total = 0
    started = time.perf_counter()

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        mixed_labels = labels
        mixup_lambda = 1.0
        if mixup_alpha > 0:
            # Use the same ratio for mixed images, loss, and accuracy.
            mixup_lambda = float(np.random.beta(mixup_alpha, mixup_alpha))
            permutation = torch.randperm(images.size(0), device=device)
            images = mixup_lambda * images + (1.0 - mixup_lambda) * images[permutation]
            mixed_labels = labels[permutation]
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = mixup_lambda * criterion(logits, labels)
            if mixup_alpha > 0:
                loss += (1.0 - mixup_lambda) * criterion(logits, mixed_labels)

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

    elapsed = time.perf_counter() - started
    return {
        "loss": running_loss / total,
        "top1_accuracy": top1_sum / total,
        "seconds": elapsed,
        "images_per_second": total / max(elapsed, 1e-9),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    num_classes,
    use_amp,
    tta=False,
    channels_last=False,
):
    model.eval()
    running_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    total = 0
    all_labels = []
    all_preds = []
    all_top5_preds = []
    all_top5_scores = []
    all_paths = []
    started = time.perf_counter()

    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            if tta:
                # Average original and flipped logits before computing metrics.
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2
            loss = criterion(logits, labels)
        probabilities = logits.softmax(dim=1)
        top5_scores, top5_predictions = probabilities.topk(5, dim=1)
        predictions = top5_predictions[:, 0]

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        top1_sum += (top5_predictions[:, 0] == labels).sum().item()
        top5_sum += (
            (top5_predictions == labels[:, None]).any(dim=1).sum().item()
        )
        total += batch_size
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(predictions.cpu().numpy().tolist())
        all_top5_preds.extend(top5_predictions.cpu().numpy().tolist())
        all_top5_scores.extend(top5_scores.cpu().numpy().tolist())
        all_paths.extend(paths)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    top1 = top1_sum / total
    elapsed = time.perf_counter() - started
    return {
        "loss": running_loss / total,
        "top1_accuracy": top1,
        "overall_accuracy": top1,
        "top5_accuracy": top5_sum / total,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "seconds": elapsed,
        "images_per_second": total / max(elapsed, 1e-9),
        "labels": all_labels,
        "preds": all_preds,
        "top5_preds": all_top5_preds,
        "top5_scores": all_top5_scores,
        "paths": all_paths,
    }
