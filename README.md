# COMP9517 iNaturalist Species Classification

This repository contains the source code for the COMP9517 2026 Term 2 group
project. The project classifies iNaturalist-2021 images into species and
compares handcrafted computer-vision pipelines with deep-learning models.

This submission contains source code, configuration files, and tests only.
Datasets, generated split manifests, model checkpoints, feature caches,
experimental outputs, and result images are intentionally excluded, as required
by the project specification.

## Implemented Methods

### Traditional computer vision

- HSV colour histograms
- Uniform Local Binary Patterns (LBP)
- Histogram of Oriented Gradients (HOG)
- SIFT Bag-of-Visual-Words
- SGD linear SVM, LinearSVC, and Random Forest classifiers

### Deep learning

- SimpleCNN trained from random initialization
- ResNet18 trained from random initialization
- ImageNet-pretrained ResNet18
- ImageNet-pretrained ResNet50
- ImageNet-pretrained ConvNeXt-Tiny
- Validation-selected ResNet50 and ConvNeXt probability ensemble

### Controlled studies

- Initialization: scratch versus ImageNet pretraining
- Basic versus strong augmentation
- Label smoothing off versus 0.1
- MixUp off versus MixUp alpha 0.2
- Test-time augmentation off versus horizontal-flip TTA
- Advanced class-scaling study with 500, 1,000, and 2,500 classes

## Repository Structure

```text
configs/       Controlled ablation and class-scaling configurations
data/          Dataset download, split generation, transforms, and loaders
evaluation/    Metrics, plots, comparisons, ablations, and error analysis
models/        SimpleCNN and torchvision model construction
scripts/       Command-line entry points
tests/         Unit and integration tests with generated synthetic fixtures
traditional/   Handcrafted features and classical classifiers
training/      Deep-learning training and evaluation loops
utils/         Reproducibility and serialization helpers
```

The following directories are created locally when commands are run and are
not included in the submission:

```text
datasets/       Extracted iNaturalist data and downloaded archives
data_splits/    Generated train, validation, and test manifests
feature_cache/  Cached handcrafted features and SIFT vocabularies
results/        Checkpoints and complete local experiment records
outputs/        Compact evaluation tables and report figures
analysis/       Error-analysis tables and selected examples
```

## Environment

Python 3.10 or later is recommended. A CUDA-capable GPU is recommended for deep
learning, but the traditional methods can run on CPU.

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The main dependencies are PyTorch, torchvision, scikit-learn, scikit-image,
NumPy, pandas, Matplotlib, seaborn, Pillow, joblib, tqdm, and pytest.

## Dataset

The graded experiments use the official iNaturalist-2021 data:

- `train_mini.tar.gz`
- `train_mini.json.tar.gz`
- `val.tar.gz`
- `val.json.tar.gz`

Download and verify all four archives:

```bash
python scripts/download_inat2021.py --root datasets/inat2021
```

The downloader stores archives in `datasets/inat2021/archives/` and verifies
their official MD5 hashes. Extract each archive into `datasets/inat2021/`.
After extraction, the dataset root must contain the `train_mini/` and `val/`
image directories and the `train_mini.json` and `val.json` annotation files.

The base experiment uses:

- 500 randomly selected eligible species
- random seed 9517
- 40 training images per species
- 10 validation images per species from `train_mini`
- 10 held-out test images per species from the official validation split

Generate the split manifests:

```bash
python scripts/prepare_splits.py \
  --data-root datasets/inat2021 \
  --output-dir data_splits \
  --seed 9517 \
  --num-classes 500 \
  --train-per-class 40 \
  --val-per-class 10 \
  --test-per-class 10
```

On PowerShell, place the command on one line or replace `\` with the PowerShell
continuation character.

Validate the generated manifests and image paths:

```bash
python scripts/verify_splits.py \
  --data-root datasets/inat2021 \
  --split-dir data_splits
```

The generated class list and manifests must be shared between all experiments.
Do not regenerate them with a different seed when comparing methods.

## Traditional Experiments

Run one feature and classifier combination with
`scripts/run_traditional_experiment.py`.

HOG with an SGD linear SVM:

```bash
python scripts/run_traditional_experiment.py \
  --feature hog \
  --classifier sgd-svm \
  --data-root datasets/inat2021 \
  --split-dir data_splits \
  --output-dir results/traditional_hog_sgd_svm_full
```

SIFT Bag-of-Visual-Words with an SGD linear SVM:

```bash
python scripts/run_traditional_experiment.py \
  --feature sift-bovw \
  --classifier sgd-svm \
  --data-root datasets/inat2021 \
  --split-dir data_splits \
  --output-dir results/traditional_sift_bovw_sgd_svm_full
```

The `--feature` choices are `color`, `lbp`, `hog`, and `sift-bovw`. The
`--classifier` choices are `linear-svc`, `sgd-svm`, and `random-forest`.
Use `--help` to inspect all feature and classifier parameters.

## Deep-Learning Experiments

Train the scratch CNN:

```bash
python scripts/train_deep.py \
  --model simple-cnn \
  --data-root datasets/inat2021 \
  --split-dir data_splits \
  --output-dir results/simple_cnn_converged_full \
  --epochs 50 \
  --augmentation basic
```

Train an ImageNet-pretrained ResNet18:

```bash
python scripts/train_deep.py \
  --model resnet18-pretrained \
  --data-root datasets/inat2021 \
  --split-dir data_splits \
  --output-dir results/resnet18_pretrained_converged_full \
  --epochs 50 \
  --augmentation strong \
  --label-smoothing 0.1 \
  --scheduler cosine \
  --early-stopping-patience 8
```

Available model names are:

```text
simple-cnn
resnet18-scratch
resnet18-pretrained
resnet50-pretrained
convnext-tiny-pretrained
```

Training writes metrics, per-epoch history, per-image Top-1 and Top-5
predictions, confusion matrices, plots, and resumable checkpoints to the chosen
`results/` directory. Use `--resume-checkpoint` to resume an interrupted run.

Run the validation-selected two-model ensemble after training its component
models:

```bash
python scripts/train_deep.py \
  --model resnet50-pretrained \
  --data-root datasets/inat2021 \
  --split-dir data_splits \
  --output-dir results/resnet50_pretrained_optimized_full

python scripts/train_deep.py \
  --model convnext-tiny-pretrained \
  --data-root datasets/inat2021 \
  --split-dir data_splits \
  --output-dir results/convnext_tiny_mixup_full \
  --mixup-alpha 0.2
```

The ensemble script reads `best_model.pt` from these two exact result
directories:

```bash
python scripts/ensemble_deep_models.py \
  --data-root datasets/inat2021 \
  --split-dir data_splits \
  --output-dir results/deep_ensemble
```

Ensemble weights are selected using validation macro-F1. The test set is
evaluated only after the weights are fixed.

## Controlled Deep Ablations

The complete one-factor-at-a-time design is recorded in
`configs/deep_ablations.json`. Preview the required runs:

```bash
python scripts/run_deep_ablations.py --dry-run
```

Run all controlled experiments:

```bash
python scripts/run_deep_ablations.py
```

Export compact tables and figures after all runs finish:

```bash
python scripts/export_deep_ablations.py
```

The validation code rejects an ablation if an undeclared training setting
changes relative to its baseline.

## Advanced Class-Scaling Study

The class-scaling design is recorded in `configs/class_scaling.json`. It keeps
the total training-image budget fixed for the main 500, 1,000, and 2,500-class
experiments and includes 500-class sample-count controls.

Generate the nested splits:

```bash
python scripts/prepare_scaling_splits.py
```

Preview or run the experiments:

```bash
python scripts/run_class_scaling.py --dry-run
python scripts/run_class_scaling.py
```

Export the compact comparison:

```bash
python scripts/export_class_scaling.py
```

## Evaluation and Report Figures

Each training pipeline saves per-image predictions and aggregate metrics.
The evaluation modules compute:

- Top-1 and Top-5 accuracy
- overall and balanced accuracy
- macro precision, recall, and F1
- per-class precision, recall, F1, and support
- full and selected-class confusion matrices
- training and inference time
- successful, failed, and Top-5-only examples

Export the completed base experiments into a compact local result package:

```bash
python scripts/export_final_results.py
```

This exporter expects every method registered in
`evaluation/export_results.py` to have completed its formal result directory.
Run `python scripts/export_final_results.py --help` to inspect its input and
output roots.

Refresh comparison, ablation, class-scaling, and compact confusion-matrix
figures from the current exported CSV files:

```bash
python scripts/refresh_final_comparison.py
```

For an external prediction artifact that follows the generic evaluation
contract, use:

```bash
python scripts/evaluate_artifacts.py --help
python scripts/compare_models.py --help
python scripts/analyze_errors.py --help
```

Generated result tables and images are written under `outputs/` and are not
part of the source-code submission.

## Tests

Run the complete test suite:

```bash
python -m pytest -q
```

The tests generate their small image fixtures in a temporary directory. No
dataset or result image is bundled with the submission.

## Reproducibility

- The base split and all training entry points use seed 9517 by default.
- Split manifests record selected classes, image paths, and remapped labels.
- Training history, configuration, timing, and per-image predictions are saved
  for each run.
- Ablation validation enforces the same split, seed, class count, and test
  sample count while allowing only the declared factor to change.
- The class-scaling split builder verifies nested class and image subsets.

## Third-Party Libraries and Resources

This submission uses third-party libraries through their public Python APIs:

- PyTorch and torchvision for neural-network training, model definitions, and
  ImageNet-pretrained weights
- scikit-learn for classical classifiers, metrics, and clustering
- scikit-image for HOG, LBP, SIFT, and image processing
- NumPy and pandas for numerical and tabular processing
- Matplotlib and seaborn for plots
- Pillow for image loading
- joblib for model and feature-cache serialization
- tqdm for progress reporting
- pytest for automated tests

No third-party source files, trained weights, or dataset files are bundled in
this submission.

Dataset reference:

Grant Van Horn, Elijah Cole, Sara Beery, Kimberly Wilber, Serge Belongie, and
Oisin Mac Aodha. "Benchmarking Representation Learning for Natural World Image
Collections." CVPR, 2021.

Official dataset repository:
<https://github.com/visipedia/inat_comp/tree/master/2021>

torchvision model documentation:
<https://pytorch.org/vision/stable/models.html>

## Creating the Submission ZIP

Create the ZIP from the checked-out `main` branch. Include the visible project
files and directories, but do not include the hidden `.git/` directory or any
locally generated ignored directory such as `datasets/`, `data_splits/`,
`feature_cache/`, `results/`, `outputs/`, or `analysis/`.
