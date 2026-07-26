# COMP9517 500 类物种分类正式实验结果

## 1. 目录用途

本目录保存已经完成并核验的真实实验结果，用于：

- 让组员无需重新训练即可检查指标、预测和图表；
- 为报告和 PPT 提供可直接引用的 CSV 与 PNG；
- 使用相同的 500 类和 train/validation/test 划分复现实验。

固定随机种子为 9517，数据划分为 20,000 张训练图、5,000 张验证图和
5,000 张测试图。图片本身不在仓库中。

## 2. 当前主要结果

原来只训练 3 个 epoch 的 SimpleCNN 和 ResNet18 结果已经被充分训练版本替换，
不再作为本目录的正式结果。

| 方法 | Epoch | 最佳 Epoch | Test Top-1 | Test Top-5 | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|
| 旧版 HOG + SGD linear SVM（20 次迭代） | - | - | 2.80% | 8.66% | 2.11% |
| SimpleCNN（从零训练） | 50 | 44 | **18.76%** | **40.04%** | **16.70%** |
| ImageNet 预训练 ResNet18 | 30 | 28 | **66.08%** | **85.78%** | **65.52%** |
| 优化后的 ImageNet 预训练 ResNet50 | - | - | 78.26% | 91.80% | 77.99% |
| ImageNet 预训练 ConvNeXt-Tiny + MixUp | - | - | 82.72% | 94.56% | 82.52% |
| ResNet50/ConvNeXt 验证集选择集成 | - | - | 83.78% | 94.90% | 83.63% |

完整机器可读结果位于：

```text
comparison/summary_metrics.csv
```

SimpleCNN 和 ResNet18 的重训原因、配置、曲线和复现命令见：

```text
CONVERGENCE_EXPERIMENTS.md
```

传统方法的 Color、LBP、HOG、SIFT-BoVW 与分类器对比见：

```text
TRADITIONAL_EXPERIMENTS.md
comparison/traditional_summary_metrics.csv
comparison/traditional_methods_comparison.png
```

ResNet18 的初始化、数据增强、Label Smoothing、MixUp 和 TTA
控制变量消融见：

```text
ablations/deep_learning/深度学习消融实验说明.md
```

## 3. 文件夹结构

```text
inat500/
|-- README.md
|-- CONVERGENCE_EXPERIMENTS.md
|-- TRADITIONAL_EXPERIMENTS.md
|-- artifact_manifest.csv
|-- comparison/
|   |-- summary_metrics.csv
|   |-- runtime_comparison.csv
|   |-- model_comparison.png
|   |-- per_class_f1_distribution.png
|   `-- runtime_vs_performance.png
|-- ablations/
|   |-- handcrafted_features/
|   |-- hog_classifiers/
|   |-- color_classifiers/
|   `-- deep_learning/
|-- examples/
|   `-- <传统方法>/
|-- methods/
|   |-- simple_cnn/
|   |-- resnet18_pretrained/
|   `-- <其他方法>/
`-- reproducibility/
    `-- data_splits/
```

`methods/simple_cnn/` 和 `methods/resnet18_pretrained/` 已经原地替换为充分训练
版本，没有额外创建容易混淆的 legacy 或 converged 方法目录。

## 4. 每个方法目录中的文件

| 文件 | 含义 |
|---|---|
| `metrics.csv` | 测试指标、训练时间和推理时间 |
| `configuration.csv` | 方法及训练超参数 |
| `history.csv` | 每个 epoch 的训练/验证指标、吞吐量和 GPU 峰值 |
| `training_curves.png` | Loss、Top-1、Top-5 和 Macro-F1 收敛曲线 |
| `learning_rate.png` | 学习率变化 |
| `test_predictions_top1.csv` | 5,000 张测试图片的真实类别和预测类别 |
| `per_class_metrics.csv` | 500 类各自的 Precision、Recall、F1 和 support |
| `confusion_matrix.csv` | 500 x 500 原始混淆计数 |
| `confusion_matrix_full.png` | 完整归一化混淆矩阵 |
| `confusion_matrix_subset.png` | 最低 F1 类别的局部混淆矩阵 |
| `top_confusions.csv` | 混淆次数最多的类别对 |
| `prediction_examples.png` | 测试集预测示例 |

不同方法可能没有训练历史、学习率或预测示例，因此对应文件可能不存在。

`examples/<传统方法>/` 保存从逐图片 Top-5 记录中筛选出的
`successes.csv`、`failures.csv`、`top5_only.csv` 和
`high_confidence_errors.csv`。当前仓库不包含原始图片，所以这些文件只保存
图片相对路径、标签和分类器分数，不生成新的案例图片网格。

## 5. 可复现数据

`reproducibility/data_splits/` 包含：

- `selected_classes.csv`：选中的 500 类；
- `class_mapping.json`：iNaturalist category ID 到连续类别编号的映射；
- `train.csv`：20,000 条训练样本相对路径；
- `val.csv`：5,000 条验证样本相对路径；
- `test.csv`：5,000 条测试样本相对路径；
- `split_summary.json`：随机种子、类别数和样本数。

## 6. 重新生成共享结果

在仓库根目录运行：

```powershell
python scripts/export_final_results.py
```

导出器从 `results/`、`analysis/error_analysis/` 和 `data_splits/` 读取结果，
并重新生成本目录中的结构化文件。SimpleCNN 和 ResNet18 默认读取新的
`simple_cnn_converged_full` 与 `resnet18_pretrained_converged_full` 原始运行。

## 7. 未上传内容

为控制仓库体积，以下内容不进入普通代码提交：

- `.pt` 和 `.joblib` 模型权重；
- iNaturalist 原始图片；
- 特征缓存和完整概率数组；
- smoke test、PID 和训练日志。

这些文件不影响组员阅读现有 CSV 和 PNG。若需要现场重新推理，应通过课程允许的
云盘或 GitHub Release 另行共享最终 checkpoint。
