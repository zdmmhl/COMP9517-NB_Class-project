# COMP9517 iNaturalist Species Classification

This project compares handcrafted computer-vision and deep-learning methods on
a reproducible 500-class subset of iNaturalist 2021.

## Implemented Methods

- HOG features with a linear SVM classifier.
- A small CNN trained from random initialization.
- ResNet18 trained from scratch.
- ImageNet-pretrained ResNet18, ResNet50, EfficientNet-B0, and ConvNeXt-Tiny.
- MixUp, strong augmentation, label smoothing, cosine scheduling, TTA, and
  validation-selected probability ensembling.
- Top-1/overall accuracy, Top-5 accuracy, macro precision, macro recall,
  macro F1, confusion matrices, runtime tracking, and error analysis.

## Repository Structure

```text
configs/          Reproducible experiment settings
data/             Downloading, manifests, datasets, and image transforms
data_splits/      Generated fixed class and split manifests
demo/             Software demonstration code (not implemented yet)
evaluation/       Metrics, plots, ensembling, comparison, and error analysis
models/           Scratch and torchvision model definitions
notebooks/        Optional exploration notebooks
outputs/          Generated predictions and figures; not committed
report/           CVPR report source (not implemented yet)
scripts/          Thin command-line entry points
traditional/      HOG feature extraction and linear SVM pipeline
training/         Deep-learning orchestration, epoch loops, and optimizers
utils/            Reproducibility and serialization helpers
```

The files in `scripts/` contain only command-line entry points. Reusable
implementation belongs to the corresponding package so training, evaluation,
and future demo code can share the same behavior.

## Environment

```powershell
python -m pip install -r requirements.txt
```

All commands below are run from the repository root. Dataset and output paths
can be overridden through command-line arguments.

## Reproduce The Data Split

```powershell
python scripts\prepare_splits.py `
  --data-root datasets\inat2021 `
  --output-dir data_splits `
  --seed 9517 `
  --num-classes 500 `
  --train-per-class 40 `
  --val-per-class 10 `
  --test-per-class 10

python scripts\verify_splits.py `
  --data-root datasets\inat2021 `
  --split-dir data_splits `
  --check-all-paths
```

This produces 20,000 training, 5,000 validation, and 5,000 held-out test
samples with no image overlap.

## Run The Baselines

```powershell
python scripts\train_hog_svm.py `
  --data-root datasets\inat2021 `
  --split-dir data_splits `
  --output-dir results\hog_svm_full

python scripts\train_deep.py `
  --model simple-cnn `
  --data-root datasets\inat2021 `
  --split-dir data_splits `
  --output-dir results\simple_cnn_full

python scripts\train_deep.py `
  --model resnet18-pretrained `
  --data-root datasets\inat2021 `
  --split-dir data_splits `
  --output-dir results\resnet18_pretrained_full
```

Use `--help` on any entry point for all optimization and evaluation options.

## Generated Files

Training and evaluation write checkpoints, histories, metrics, predictions,
confusion matrices, plots, and example images to the selected output
directory. Datasets, model weights, caches, and generated outputs must not be
committed to GitHub or included in the final code ZIP.

## Shared Final Results

The compact, report-ready results package is tracked under:

```text
outputs/final_results/
```

It contains standardized CSV metrics, training histories, per-class metrics,
confusion matrices, selected result figures, and the fixed 500-class split
manifests. It excludes datasets, model weights, caches, probability arrays,
and smoke-test artifacts.

Regenerate the package from completed local experiments with:

```powershell
python scripts/export_final_results.py
```

See `outputs/final_results/README.md` for the Chinese result guide, metric
definitions, known timing limitations, and file descriptions.
