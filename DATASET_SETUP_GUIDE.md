# Dataset Setup Guide

This guide explains which iNaturalist-2021 files are required for the COMP9517
Group Project, how to download and extract them, and how to configure data paths
on a local computer, a server, or Google Colab.

## 1. Required Files

The project uses the iNaturalist-2021 `train_mini` split and the official
validation split.

Download these four files:

| File | Purpose | Approximate size |
| --- | --- | ---: |
| `train_mini.tar.gz` | Training images, with 50 images per species | 42 GB |
| `train_mini.json.tar.gz` | Labels and category information for the training images | 45 MB |
| `val.tar.gz` | Official validation images, used as the test set in this project | 8.4 GB |
| `val.json.tar.gz` | Labels for the official validation images | 9.4 MB |

Official download page:

<https://github.com/visipedia/inat_comp/tree/master/2021#data>

Direct download URLs:

- <https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz>
- <https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz>
- <https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz>
- <https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz>

The following files are not required:

- The full `train.tar.gz`, which is approximately 224 GB.
- `public_test.tar.gz`, because its labels are not public and the project
  requires the official validation split to be used as the held-out test set.

## 2. Recommended Directory Structure

After downloading and extracting the files, organize them as follows:

```text
inat2021/
|-- archives/
|   |-- train_mini.tar.gz
|   |-- train_mini.json.tar.gz
|   |-- val.tar.gz
|   `-- val.json.tar.gz
|-- train_mini/
|-- val/
|-- train_mini.json
`-- val.json
```

The `archives` directory stores the original compressed files. If storage is
limited, the archives can be removed after their checksums have been verified
and extraction has completed successfully.

## 3. Downloading on Windows PowerShell

Create a data directory. The path below is only an example and should be
replaced with a location that has sufficient free space:

```powershell
New-Item -ItemType Directory -Force F:\COMP9517_data\inat2021\archives
cd F:\COMP9517_data\inat2021\archives
```

Download the four required files:

```powershell
curl.exe -L -C - -o train_mini.tar.gz https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz

curl.exe -L -C - -o train_mini.json.tar.gz https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz

curl.exe -L -C - -o val.tar.gz https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz

curl.exe -L -C - -o val.json.tar.gz https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz
```

The `-C -` option tells `curl` to attempt to resume an interrupted download when
the same command is run again.

## 4. Downloading on Linux, a Server, or Google Colab

Create and enter the download directory:

```bash
mkdir -p /path/to/inat2021/archives
cd /path/to/inat2021/archives
```

Download the files with `wget`:

```bash
wget -c https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
wget -c https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz
wget -c https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz
wget -c https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz
```

The `-c` option allows `wget` to continue an interrupted download.

When using Google Colab, first confirm that the runtime and Google Drive have
enough free space. Do not assume that the full dataset can be stored permanently
on the Colab runtime disk, because temporary files can be deleted when the
runtime is restarted.

## 5. Verifying Downloads

After downloading these large files, verify their MD5 checksums before
extraction. This avoids using incomplete or corrupted archives.

Official MD5 checksums:

```text
train_mini.tar.gz       db6ed8330e634445efc8fec83ae81442
train_mini.json.tar.gz  395a35be3651d86dc3b0d365b8ea5f92
val.tar.gz              f6f6e0e242e3d4c9569ba56400938afc
val.json.tar.gz         4d761e0f6a86cc63e8f7afc91f6a8f0b
```

On Windows PowerShell:

```powershell
Get-FileHash train_mini.tar.gz -Algorithm MD5
Get-FileHash train_mini.json.tar.gz -Algorithm MD5
Get-FileHash val.tar.gz -Algorithm MD5
Get-FileHash val.json.tar.gz -Algorithm MD5
```

On Linux or Colab:

```bash
md5sum train_mini.tar.gz
md5sum train_mini.json.tar.gz
md5sum val.tar.gz
md5sum val.json.tar.gz
```

Each calculated checksum must exactly match the corresponding official value.

## 6. Extracting the Dataset

Enter the `inat2021` root directory before extracting the files.

On Windows PowerShell:

```powershell
cd F:\COMP9517_data\inat2021

tar -xzf archives\train_mini.tar.gz
tar -xzf archives\train_mini.json.tar.gz
tar -xzf archives\val.tar.gz
tar -xzf archives\val.json.tar.gz
```

On Linux, a server, or Colab:

```bash
cd /path/to/inat2021

tar -xzf archives/train_mini.tar.gz
tar -xzf archives/train_mini.json.tar.gz
tar -xzf archives/val.tar.gz
tar -xzf archives/val.json.tar.gz
```

Extracting a large number of small image files can take a considerable amount
of time. Confirm that the destination disk has enough free space before
starting extraction.

## 7. Configuring the Dataset Path

The repository records the primary 500-class defaults in:

```text
configs/baseline.yaml
```

This YAML file is a human-readable reference for the agreed split. The current
Python entry points do **not** load it automatically. Changing `data.root` in
this file alone does not change a training or data-preparation run.

The recorded defaults are:

```yaml
data:
  root: datasets/inat2021
  train_annotations: train_mini.json
  test_annotations: val.json
  split_dir: data_splits
  seed: 9517
  num_classes: 500
  train_per_class: 40
  val_per_class: 10
  test_per_class: 10
  image_size: 224
```

The fields have the following meanings:

- `root`: root directory of the extracted iNaturalist-2021 data.
- `train_annotations`: annotation file for the training data.
- `test_annotations`: annotation file for the official validation data, which
  is used as the test data in this project.
- `split_dir`: directory containing class-selection and split manifests.
- `num_classes`: number of randomly selected species.
- `train_per_class`: number of training images per species.
- `val_per_class`: number of validation images per species.
- `test_per_class`: number of test images per species.

Pass the dataset location to the standard scripts with `--data-root`. For
example:

```powershell
python scripts\prepare_splits.py `
  --data-root F:\COMP9517_data\inat2021 `
  --output-dir data_splits `
  --seed 9517 `
  --num-classes 500 `
  --train-per-class 40 `
  --val-per-class 10 `
  --test-per-class 10

python scripts\train_deep.py `
  --model resnet18-pretrained `
  --data-root F:\COMP9517_data\inat2021 `
  --split-dir data_splits `
  --output-dir results\resnet18_pretrained_run
```

On Linux or a server, use the corresponding absolute path:

```bash
python scripts/prepare_splits.py \
  --data-root /data/COMP9517/inat2021 \
  --output-dir data_splits \
  --seed 9517 \
  --num-classes 500 \
  --train-per-class 40 \
  --val-per-class 10 \
  --test-per-class 10
```

For Colab, pass either the Drive path or the temporary runtime path:

```bash
python scripts/prepare_splits.py \
  --data-root /content/drive/MyDrive/COMP9517/inat2021 \
  --output-dir data_splits
```

The temporary runtime disk is usually faster, but the data must be copied or
downloaded again after the Colab runtime is reset.

The controlled deep-ablation and Advanced class-scaling workflows read
`configs/deep_ablations.json` and `configs/class_scaling.json`. Their
`data_root` values are repository-relative by default and may be changed in a
local, uncommitted configuration copy when the dataset is stored elsewhere.

## 8. Files Shared by the Team

The image files and large archives must not be uploaded to GitHub. Each team
member can store the dataset in a different location and pass that location
through `--data-root`, or through a local JSON configuration for the controlled
experiment runners.

To ensure that every experiment uses exactly the same data, the team should
share these small files:

- The random seed.
- The IDs and names of the 500 selected species.
- The train, validation, and test split manifests.
- The mapping from original category IDs to model class indices.

Image paths in manifests should be relative to the selected dataset root, for
example:

```text
train_mini/species_directory/image.jpg
```

Do not store an absolute path from one team member's computer, such as:

```text
F:\COMP9517_data\inat2021\train_mini\species_directory\image.jpg
```

Relative paths allow the same split manifests to work on different computers,
servers, and Colab environments.

## 9. Required Dataset Split

The minimum requirement is to randomly select at least 500 species. The
recommended split for each species is:

```text
50 images from train_mini:
40 images -> training
10 images -> validation

10 images from the official validation split:
10 images -> testing
```

The training, validation, and test sets must remain strictly separate. The
official validation split is the final test set for this project and must not be
used for training or hyperparameter tuning.

## 10. Recommended Hardware Assessment

Before downloading the dataset and running full experiments, the team member
responsible for training should check:

- GPU model and available VRAM.
- Number of CPU cores.
- Available system memory.
- Data-disk type and free space.
- Whether PyTorch can access CUDA.
- Planned image resolution, batch size, and number of training epochs.

Run a short experiment with a small number of classes and images first. Measure
the actual time per epoch and peak VRAM usage before selecting the batch size,
number of epochs, and model scale for the full experiments.
