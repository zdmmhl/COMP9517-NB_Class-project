# COMP9517 iNaturalist Species Classification

This repository is the working codebase for the COMP9517 26T2 group project.
The project compares traditional computer vision and deep learning methods for
classifying species from an iNaturalist-2021 subset.

## Current status

The repository structure and evaluation pipeline are now implemented. Dataset
preparation, model training, and the demo will be implemented in later stages.

## Planned baselines

- Traditional computer vision: HOG features with an SVM classifier.
- Deep learning from scratch: a small CNN with randomly initialized weights.
- Transfer learning: an ImageNet-pretrained ResNet18 fine-tuned for 500 species.

All methods will use the same fixed class list and held-out test set. Planned
evaluation includes Top-1 accuracy, Top-5 accuracy, macro precision, macro
recall, macro F1, confusion matrices, and runtime.

## Repository structure

```text
COMP9517-NB_Class-project/
|-- configs/          # Reproducible experiment settings
|-- data/             # Dataset classes, transforms, and split logic
|-- data_splits/      # Generated class lists and split manifests
|-- demo/             # Software demonstration code
|-- evaluation/       # Classification metrics and error analysis
|-- models/           # Scratch and pretrained deep learning models
|-- notebooks/        # Exploration notebooks only
|-- outputs/          # Local figures, tables, and prediction files
|-- report/           # CVPR LaTeX report files
|-- scripts/          # Command-line entry points
|-- traditional/      # Handcrafted features and classical classifiers
|-- training/         # Shared deep learning training loop
|-- utils/            # Paths, random seeds, and common helpers
|-- requirements.txt  # Python dependencies
`-- README.md
```

## Data policy

The iNaturalist dataset is not stored in Git. Local dataset paths are configured
in `configs/baseline.yaml`. Generated split manifests should record the random
seed, selected species, image paths, and labels so that every experiment uses
exactly the same train, validation, and test data.

## Evaluation module

The evaluation module now provides:

- strict validation of prediction, metadata, class mapping, test manifest, and
  optional training history files;
- Top-1 and Top-5 accuracy, macro precision, macro recall, macro F1, balanced
  accuracy, per-class metrics, and confusion matrices;
- common-confusion, hardest-class, successful-prediction, Top-5-only, complete-
  failure, and high-confidence-error analysis;
- confusion matrix, training history, qualitative example, model comparison,
  per-class F1 distribution, and runtime-performance figures; and
- command-line pipelines for evaluating one model and comparing multiple models
  that used the same test split.

Run the complete automated test suite with:

```bash
python -m pytest
```

A small deterministic fixture is available under `tests/fixtures/mock_small/`.
Example report artifacts generated from this fixture are committed under
`outputs/evaluation/mock_classifier/`, `outputs/examples/mock_classifier/`, and
`outputs/comparison/`. They document the expected output layout; real experiment
artifacts remain ignored by default.

Evaluate the fixture and regenerate its figures with:

```bash
python scripts/evaluate.py \
  --prediction-dir tests/fixtures/mock_small \
  --class-mapping tests/fixtures/mock_small/class_mapping.csv \
  --test-manifest tests/fixtures/mock_small/test.csv \
  --history tests/fixtures/mock_small/history.csv \
  --image-root tests/fixtures/mock_small \
  --output-dir outputs/evaluation/mock_classifier \
  --examples-dir outputs/examples/mock_classifier

python scripts/compare_models.py \
  --evaluation-root outputs/evaluation \
  --output-dir outputs/comparison
```
