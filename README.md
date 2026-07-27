# COMP9517 iNaturalist Species Classification

This project compares handcrafted computer-vision and deep-learning methods on
a fixed, reproducible 500-class subset of iNaturalist 2021. The 500-class split
is the primary benchmark for all headline comparisons. A separate Advanced
study uses nested 500-, 1,000-, and 2,500-class subsets to measure class-scaling
effects without replacing the primary benchmark.

## Implemented Methods

- Colour histogram, LBP, HOG, and SIFT Bag-of-Visual-Words features.
- SGD linear SVM, LinearSVC, and Random Forest classifier comparisons.
- A small CNN trained from random initialization.
- ResNet18 trained from scratch.
- ImageNet-pretrained ResNet18, ResNet50, EfficientNet-B0, and ConvNeXt-Tiny.
- MixUp, strong augmentation, label smoothing, cosine scheduling, TTA, and
  validation-selected probability ensembling.
- Controlled ResNet18 ablations for initialization, augmentation, label
  smoothing, MixUp, and test-time augmentation.
- An Advanced class-scaling study with fixed-total-image and fixed-class
  sample-count controls.
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
outputs/          Local runs plus compact report-ready result packages
report/           CVPR report source (not implemented yet)
scripts/          Thin command-line entry points
traditional/      Handcrafted feature extraction and classical classifiers
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

## Run The Experiments

```powershell
python scripts\train_hog_svm.py `
  --data-root datasets\inat2021 `
  --split-dir data_splits `
  --output-dir results\hog_svm_full

python scripts\train_deep.py `
  --model simple-cnn `
  --data-root datasets\inat2021 `
  --split-dir data_splits `
  --output-dir results\simple_cnn_converged_full `
  --image-size 128 --epochs 50 `
  --batch-size 512 --eval-batch-size 1024 `
  --lr 0.002 --weight-decay 0.0001 `
  --scheduler cosine --min-lr-ratio 0.01 `
  --label-smoothing 0.1 --augmentation basic `
  --grad-clip 1.0 `
  --early-stopping-patience 8 `
  --early-stopping-min-delta 0.0005

python scripts\train_deep.py `
  --model resnet18-pretrained `
  --data-root datasets\inat2021 `
  --split-dir data_splits `
  --output-dir results\resnet18_pretrained_converged_full `
  --image-size 224 --epochs 30 `
  --batch-size 512 --eval-batch-size 1024 `
  --lr 0.001 --backbone-lr-multiplier 0.1 `
  --weight-decay 0.0001 `
  --scheduler cosine --min-lr-ratio 0.01 `
  --label-smoothing 0.1 --augmentation strong `
  --grad-clip 1.0 --tta `
  --early-stopping-patience 6 `
  --early-stopping-min-delta 0.0005 `
  --num-workers 4 --eval-num-workers 0
```

Use `--resume-checkpoint results\<run>\last_checkpoint.pt` to recover an
interrupted training run. Use `--help` on any entry point for all optimization
and evaluation options.

## Evaluate Standardized Prediction Artifacts

The repository also supports model-independent evaluation from the shared
`predictions.csv` and `metadata.json` contract documented in
`模型输出格式指南.md`. This path does not load a model checkpoint:

```powershell
python scripts\evaluate_artifacts.py `
  --prediction-dir tests\fixtures\mock_small `
  --class-mapping tests\fixtures\mock_small\class_mapping.csv `
  --test-manifest tests\fixtures\mock_small\test.csv `
  --history tests\fixtures\mock_small\history.csv `
  --image-root tests\fixtures\mock_small `
  --output-dir outputs\runs\mock_evaluation\methods\mock_classifier `
  --examples-dir outputs\runs\mock_evaluation\examples\mock_classifier

python scripts\compare_models.py `
  --evaluation-root outputs\runs\mock_evaluation\methods `
  --output-dir outputs\runs\mock_evaluation\comparison
```

`scripts/evaluate.py` remains the compatibility entry point for checkpoint-based
deep-model evaluation. `scripts/evaluate_artifacts.py` is the generic evaluator
for saved outputs from traditional or deep-learning methods.

Run the evaluation contract and pipeline tests with:

```powershell
python -m pytest
```

## Plot Controlled Ablations

Ablation implementation lives in `evaluation/ablation.py`; the command-line
entry point is `scripts/plot_ablations.py`. The evaluator only reads saved run
artifacts and does not import or modify model-training code.

Each study directory contains `study.json` plus one directory per variant.
Repeated seeds are stored below each variant. See
`tests/fixtures/mock_ablation/` for a complete example.

```powershell
python scripts\plot_ablations.py `
  --study-dir tests\fixtures\mock_ablation `
  --output-dir outputs\runs\mock_evaluation\ablations\mock_augmentation
```

The command verifies that all variants use the same split, class count, seed
set, and controlled configuration. It rejects runs that change parameters not
declared in `allowed_config_differences`. It then writes aggregate metrics,
baseline deltas, validation details, error-bar charts, and training curves.

The completed deep-learning ablations are driven by
`configs/deep_ablations.json`:

```powershell
python scripts/run_deep_ablations.py
python scripts/export_deep_ablations.py
python scripts/write_experiment_docs.py
```

## Run The Advanced Class-Scaling Study

The primary benchmark remains the fixed 500-class split. The Advanced study
constructs nested 500-, 1,000-, and 2,500-class subsets while keeping the main
training-image budget at 20,000. Two additional 500-class controls isolate the
effect of reducing images per class.

```powershell
python scripts/prepare_scaling_splits.py
python scripts/run_class_scaling.py
python scripts/export_class_scaling.py
python scripts/write_experiment_docs.py
```

The complete specification is in `configs/class_scaling.json`. Report-ready
outputs are stored below
`outputs/final_results/inat500/advanced/class_scaling/`; they are not mixed into
the primary 500-class model comparison.

## Generated Files

Generated evaluation artifacts are grouped by split identity:

```text
outputs/runs/inat500_seed9517/
```

Each run is self-contained and may include standardized inputs, per-method
metrics, comparisons, examples, ablations, and `run_manifest.json`. Local runs,
datasets, model weights, and caches must not be committed to GitHub or included
in the final code ZIP.

Re-evaluate the current compact 500-class result package with:

```powershell
python scripts\evaluate_recorded_results.py
```

The script reads the class count and seed from `split_summary.json` and writes
to `outputs/runs/<split_id>/`. For another completed setup, pass its package
with `--final-results-root`.

## Shared Final Results

The compact, report-ready results package is tracked under:

```text
outputs/final_results/inat500/
|-- methods/                    Primary 500-class method records
|-- comparison/                 Primary 500-class comparisons
|-- ablations/
|   |-- deep_learning/          Controlled ResNet18 ablations
|   `-- <traditional studies>/
`-- advanced/
    `-- class_scaling/          Nested 500/1,000/2,500-class study
```

It contains standardized CSV metrics, training histories, per-class metrics,
confusion matrices, selected result figures, and the matching fixed split
manifests. The primary and Advanced results remain separated inside the package.
It excludes datasets, model weights, caches, probability arrays, and smoke-test
artifacts.

Regenerate the package from completed local experiments with:

```powershell
python scripts/export_final_results.py
```

See `outputs/final_results/README.md` for the experiment index. Each experiment
directory contains its own result guide, metric definitions, timing
limitations, and file descriptions.

## Third-Party Libraries

The implementation uses PyTorch and torchvision for deep models, scikit-learn
and scikit-image for classical methods and features, and NumPy, pandas,
Matplotlib, Seaborn, and Pillow for data handling and visualisation. Exact
dependencies are listed in `requirements.txt`. Pretrained model definitions and
weights are provided through torchvision; the project-specific training,
evaluation, ablation, and class-scaling orchestration is implemented in this
repository.

## Build The Code Submission ZIP

Do not submit a ZIP of the whole repository: the tracked report-ready results
contain images and make a full archive exceed the 40 MB code limit. Build the
source-only archive from the repository root:

```powershell
git archive --format=zip --output COMP9517_code.zip HEAD `
  README.md requirements.txt configs data data_splits demo evaluation models `
  notebooks scripts tests traditional training utils DATASET_SETUP_GUIDE.md `
  模型输出格式指南.md 评估模块文件结构指南.md
```

This intentionally excludes datasets, checkpoints, report files, and all
generated result images under `outputs/`.
