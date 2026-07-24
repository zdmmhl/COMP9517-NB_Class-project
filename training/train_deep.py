"""Deep-learning experiment orchestration."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from data.dataset import SplitImageDataset
from data.splits import apply_remap, filter_rows, read_rows, remap_rows
from data.transforms import build_transforms
from evaluation.plots import (
    plot_confusion_matrix,
    plot_history,
    plot_learning_rate,
    save_prediction_grid,
)
from models.factory import MODEL_NAMES, build_model
from training.engine import evaluate, train_one_epoch
from training.optimizers import build_optimizer
from utils.reproducibility import seed_everything
from utils.serialization import save_json, save_predictions, save_rows_csv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a scratch CNN or transfer-learning model on iNat splits."
    )
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/inat2021"))
    parser.add_argument("--split-dir", type=Path, default=Path("data_splits"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Validation/test batch size; 0 reuses --batch-size.",
    )
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
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=0,
        help="Validation/test workers; 0 avoids Windows worker memory duplication.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9517)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-eval-per-class", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation macro-F1 improvement; 0 disables.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model, optimizer, scheduler, AMP scaler, and history.",
    )
    parser.add_argument("--sample-images", type=int, default=12)
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        default=None,
        help="Skip training and evaluate an existing checkpoint.",
    )
    return parser.parse_args(argv)


def build_dataloaders(args, device):
    train_rows = filter_rows(
        read_rows(args.split_dir / "train.csv"),
        args.max_classes,
        args.max_train_per_class,
    )
    train_rows, remap = remap_rows(train_rows)
    val_rows = apply_remap(
        filter_rows(
            read_rows(args.split_dir / "val.csv"),
            args.max_classes,
            args.max_eval_per_class,
        ),
        remap,
    )
    test_rows = apply_remap(
        filter_rows(
            read_rows(args.split_dir / "test.csv"),
            args.max_classes,
            args.max_eval_per_class,
        ),
        remap,
    )

    train_transform, eval_transform = build_transforms(
        args.image_size, args.augmentation
    )
    datasets = {
        "train": SplitImageDataset(train_rows, args.data_root, train_transform),
        "val": SplitImageDataset(val_rows, args.data_root, eval_transform),
        "test": SplitImageDataset(test_rows, args.data_root, eval_transform),
    }
    train_worker_options = {}
    if args.num_workers > 0:
        train_worker_options = {
            "persistent_workers": True,
            "prefetch_factor": args.prefetch_factor,
        }
    eval_worker_options = {}
    if args.eval_num_workers > 0:
        eval_worker_options = {
            "persistent_workers": False,
            "prefetch_factor": args.prefetch_factor,
        }
    eval_batch_size = args.eval_batch_size or args.batch_size
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            **train_worker_options,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.eval_num_workers,
            pin_memory=device.type == "cuda",
            **eval_worker_options,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.eval_num_workers,
            pin_memory=device.type == "cuda",
            **eval_worker_options,
        ),
    }
    return (train_rows, val_rows, test_rows), remap, datasets, loaders


def shutdown_dataloader_workers(loaders):
    """Release persistent Windows workers before memory-heavy final evaluation."""
    for loader in loaders.values():
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            loader._iterator = None


def build_final_dataloaders(datasets, batch_size, device):
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }


def train_model(
    args,
    model,
    loaders,
    criterion,
    optimizer,
    scheduler,
    device,
    use_amp,
    remap,
):
    history = []
    best_val_f1 = -1.0
    best_path = args.output_dir / "best_model.pt"
    last_path = args.output_dir / "last_checkpoint.pt"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    epochs_without_improvement = 0
    stopped_early = False
    best_epoch = 0

    if args.resume_checkpoint:
        checkpoint = torch.load(
            args.resume_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("model") != args.model:
            raise ValueError(
                f"Resume checkpoint model {checkpoint.get('model')} "
                f"does not match requested {args.model}."
            )
        if int(checkpoint.get("num_classes", -1)) != len(remap):
            raise ValueError("Resume checkpoint class count does not match the split.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        history = checkpoint.get("history", [])
        best_val_f1 = float(checkpoint.get("best_val_f1", -1.0))
        best_epoch = int(
            checkpoint.get(
                "best_epoch",
                min(
                    history,
                    key=lambda item: abs(
                        item["val_macro_f1"] - best_val_f1
                    ),
                )["epoch"]
                if history
                else 0,
            )
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch} using {args.resume_checkpoint}")

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            use_amp,
            args.grad_clip,
            args.mixup_alpha,
            scaler,
            args.channels_last,
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            len(remap),
            use_amp,
            args.tta,
            args.channels_last,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1_accuracy": train_metrics["top1_accuracy"],
            "train_seconds": train_metrics["seconds"],
            "train_images_per_second": train_metrics["images_per_second"],
            "val_loss": val_metrics["loss"],
            "val_top1_accuracy": val_metrics["top1_accuracy"],
            "val_top5_accuracy": val_metrics["top5_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_seconds": val_metrics["seconds"],
            "val_images_per_second": val_metrics["images_per_second"],
            "head_lr": current_lrs[-1],
        }
        if device.type == "cuda":
            row["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(
                device
            ) / (1024**2)
        if len(current_lrs) > 1:
            row["backbone_lr"] = current_lrs[0]
        history.append(row)
        save_json(args.output_dir / "history.json", history)
        save_rows_csv(args.output_dir / "history.csv", history)
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"train_loss={row['train_loss']:.4f}, "
            f"train_top1={row['train_top1_accuracy']:.4f}, "
            f"val_top1={row['val_top1_accuracy']:.4f}, "
            f"val_top5={row['val_top5_accuracy']:.4f}, "
            f"val_f1={row['val_macro_f1']:.4f}"
        )

        improved = (
            val_metrics["macro_f1"]
            > best_val_f1 + args.early_stopping_min_delta
        )
        if improved:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict(),
            "model": args.model,
            "num_classes": len(remap),
            "class_remap": {str(key): value for key, value in remap.items()},
            "epoch": epoch,
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "best_val_f1": float(best_val_f1),
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, best_path)

        training_state = {
            "max_epochs": args.epochs,
            "completed_epochs": epoch,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "stopped_early": False,
            "last_checkpoint": str(last_path),
            "best_checkpoint": str(best_path),
        }
        save_json(args.output_dir / "training_state.json", training_state)

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: validation macro-F1 "
                f"did not improve by {args.early_stopping_min_delta:g} for "
                f"{epochs_without_improvement} epochs."
            )
            break

    if not best_path.exists():
        raise RuntimeError("Training completed without writing a best checkpoint.")
    training_state = {
        "max_epochs": args.epochs,
        "completed_epochs": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "epochs_without_improvement": epochs_without_improvement,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "stopped_early": stopped_early,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
    }
    save_json(args.output_dir / "training_state.json", training_state)
    return history, best_path, training_state


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.set_float32_matmul_precision("high")
    rows, remap, datasets, loaders = build_dataloaders(args, device)
    train_rows, val_rows, test_rows = rows
    num_classes = len(remap)

    model, initialization = build_model(args.model, num_classes)
    model.to(device)
    if device.type == "cuda" and args.channels_last:
        model.to(memory_format=torch.channels_last)
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
    print(f"  init: {initialization}")
    print(f"  device: {device}")
    print(f"  classes: {num_classes}")
    print(f"  train/val/test rows: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(
        f"  epochs: {args.epochs}, batch_size: {args.batch_size}, "
        f"eval_batch_size: {args.eval_batch_size or args.batch_size}, amp: {use_amp}"
    )
    if device.type == "cuda":
        print(
            f"  gpu: {torch.cuda.get_device_name(device)}, "
            f"channels_last: {args.channels_last}, "
            f"cudnn_benchmark: {args.cudnn_benchmark}"
        )
    print(
        f"  augmentation: {args.augmentation}, "
        f"label_smoothing: {args.label_smoothing}, "
        f"mixup_alpha: {args.mixup_alpha}, scheduler: {args.scheduler}, "
        f"tta: {args.tta}"
    )

    total_started = time.perf_counter()
    if args.evaluate_checkpoint:
        best_path = args.evaluate_checkpoint
        history_path = args.output_dir / "history.json"
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.exists()
            else []
        )
        training_state_path = args.output_dir / "training_state.json"
        training_state = (
            json.loads(training_state_path.read_text(encoding="utf-8"))
            if training_state_path.exists()
            else {}
        )
    else:
        history, best_path, training_state = train_model(
            args,
            model,
            loaders,
            criterion,
            optimizer,
            scheduler,
            device,
            use_amp,
            remap,
        )

    shutdown_dataloader_workers(loaders)
    final_loaders = build_final_dataloaders(
        datasets,
        args.eval_batch_size or args.batch_size,
        device,
    )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(
        model,
        final_loaders["test"],
        criterion,
        device,
        num_classes,
        use_amp,
        args.tta,
        args.channels_last,
    )
    train_eval_metrics = evaluate(
        model,
        final_loaders["train"],
        criterion,
        device,
        num_classes,
        use_amp,
        False,
        args.channels_last,
    )

    report = classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        labels=list(range(num_classes)),
        zero_division=0,
        output_dict=True,
    )
    save_json(args.output_dir / "history.json", history)
    save_rows_csv(args.output_dir / "history.csv", history)
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
        datasets["test"],
        device,
        args.sample_images,
        use_amp,
    )

    metrics = {
        "method": args.model,
        "run_mode": run_mode,
        "initialization": initialization,
        "device": str(device),
        "num_classes": num_classes,
        "data_root": str(args.data_root).replace("\\", "/"),
        "params": {
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size or args.batch_size,
            "num_workers": args.num_workers,
            "eval_num_workers": args.eval_num_workers,
            "prefetch_factor": args.prefetch_factor,
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
            "channels_last": args.channels_last,
            "cudnn_benchmark": args.cudnn_benchmark,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "resume_checkpoint": str(args.resume_checkpoint)
            if args.resume_checkpoint
            else None,
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
        "training_state": training_state,
        "training_seconds": sum(
            float(row.get("train_seconds", 0.0)) for row in history
        ),
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
