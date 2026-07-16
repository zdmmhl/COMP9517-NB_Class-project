# COMP9517 iNaturalist Species Classification

This repository is the working codebase for the COMP9517 26T2 group project.
The project compares traditional computer vision and deep learning methods for
classifying species from an iNaturalist-2021 subset.

## Current status

The repository currently contains the project structure only. Dataset
preparation, model training, evaluation, and the demo will be implemented in
later stages.

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

## Reuse note

The directory organization was inspired by
[ParzHe/CV9517_Group-Project](https://github.com/ParzHe/CV9517_Group-Project).
Its segmentation-specific implementation is not copied because this repository
targets a different classification task.
