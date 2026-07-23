# COMP9517 500 类物种分类实验结果

## 1. 这是什么

本目录保存当前已经完成的真实实验结果，用于：

- 让组员在不重新训练模型的情况下查看结果；
- 为报告和 PPT 绘制表格、曲线及比较图；
- 核对每个模型的预测、每类指标和混淆情况；
- 使用完全相同的 500 类及 train/validation/test 划分复现实验。

生成日期：2026-07-23

随机种子：9517

类别数量：500

训练/验证/测试图片数：20,000 / 5,000 / 5,000

## 2. 当前主要结果

| 方法 | Top-1 | Top-5 | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| 旧版 HOG + SGD linear SVM（20 次迭代） | 2.80% | 8.66% | 2.56% | 2.80% | 2.11% |
| SimpleCNN（从零训练） | 3.94% | 12.86% | 2.57% | 3.94% | 2.17% |
| ImageNet 预训练 ResNet18 | 37.26% | 64.16% | 44.33% | 37.26% | 36.09% |
| 优化后的 ImageNet 预训练 ResNet50 | 78.26% | 91.80% | 79.24% | 78.26% | 77.99% |
| ImageNet 预训练 ConvNeXt-Tiny + MixUp | 82.72% | 94.56% | 83.76% | 82.72% | 82.52% |
| 验证集选择的 ResNet50/ConvNeXt 集成 | **83.78%** | **94.90%** | **84.86%** | **83.78%** | **83.63%** |

完整数值请使用：

```text
comparison/summary_metrics.csv
```

补充的 Color / LBP / HOG / SIFT-BoVW 与分类器对比请阅读：

```text
TRADITIONAL_EXPERIMENTS.md
comparison/traditional_summary_metrics.csv
comparison/traditional_methods_comparison.png
```

## 3. 文件夹结构

```text
final_results/inat500/
|-- README.md
|-- TRADITIONAL_EXPERIMENTS.md
|-- artifact_manifest.csv
|-- comparison/
|   |-- summary_metrics.csv
|   |-- traditional_summary_metrics.csv
|   |-- traditional_methods_comparison.png
|   |-- runtime_comparison.csv
|   |-- model_comparison.png
|   |-- runtime_vs_performance.png
|   `-- per_class_f1_distribution.png
|-- methods/
|   |-- hog_svm/
|   |-- color_sgd_svm/
|   |-- lbp_sgd_svm/
|   |-- hog_sgd_svm/
|   |-- sift_bovw_sgd_svm/
|   |-- color_linear_svc/
|   |-- hog_random_forest/
|   |-- simple_cnn/
|   |-- resnet18_pretrained/
|   |-- resnet50_optimized/
|   |-- convnext_mixup/
|   `-- deep_ensemble/
`-- reproducibility/
    `-- data_splits/
```

## 4. 每个方法目录中的文件

| 文件 | 含义 |
|---|---|
| `metrics.csv` | 该模型的最终测试指标和运行时间 |
| `configuration.csv` | 模型及训练超参数 |
| `history.csv` | 每个 epoch 的训练和验证历史；无训练过程的方法没有此文件 |
| `test_predictions_top1.csv` | 5,000 张测试图片的真实类别和 Top-1 预测 |
| `test_predictions_top5.csv` | 新增传统实验的逐图片 Top-5 类别和分数 |
| `per_class_metrics.csv` | 500 类各自的 Precision、Recall、F1 和 support |
| `confusion_matrix.csv` | 500 x 500 原始混淆计数 |
| `confusion_matrix_full.png` | 完整 500 类归一化混淆矩阵 |
| `confusion_matrix_subset.png` | F1 最低的 25 类局部混淆矩阵 |
| `top_confusions.csv/png` | 最常见的物种混淆 |
| `worst_classes.csv` | 表现最差的类别 |
| `training_curves.png` | 训练与验证曲线；集成和 HOG 没有 epoch 曲线 |
| `learning_rate.png` | 学习率变化；仅适用于记录了调度器的模型 |
| `correct_examples.jpg` | 压缩后的代表性正确预测 |
| `failure_examples.jpg` | 压缩后的代表性失败案例 |

## 5. CSV 使用建议

报告总表和模型柱状图：

```text
comparison/summary_metrics.csv
```

运行时间与性能分析：

```text
comparison/runtime_comparison.csv
```

训练曲线：

```text
methods/<方法名>/history.csv
```

类别难度、F1 分布和最差类别：

```text
methods/<方法名>/per_class_metrics.csv
methods/<方法名>/worst_classes.csv
```

预测错误和混淆类别分析：

```text
methods/<方法名>/test_predictions_top1.csv
methods/<方法名>/top_confusions.csv
methods/<方法名>/confusion_matrix.csv
```

## 6. 重要口径说明

1. 本任务是单标签多分类，因此 `overall_accuracy` 与 `top1_accuracy` 数学上相同。
2. `balanced_accuracy` 等于 500 个类别 recall 的宏平均，即表中的 `macro_recall`。
3. 旧版 HOG 的 `training_time_seconds` 只包含分类器拟合；新增传统实验将首次特征提取时间单独记录在 `feature_extraction_time_seconds`。
4. ResNet50 原始运行没有保存每个 epoch 的时间，因此训练时间字段为空，未进行猜测或补造。
5. 集成模型本身不重新训练，因此训练时间为空；推理时间是两个组成模型测试推理时间之和。
6. 原始深度学习与旧版 HOG 实验只保存了逐图片 Top-1，因此这些方法只提供聚合 Top-5。新增的 6 个传统实验真实保存了各 5,000 行逐图片 Top-5 类别和分数。
7. SimpleCNN 当前只训练了 3 个 epoch，结果可以作为第一版 scratch baseline，但后续仍应增加训练轮数证明合理收敛。
8. `history.csv` 仅写入原实验真实记录的字段。ResNet50 未记录的 validation loss 和 epoch 时间保持为空。

## 7. 可复现数据

`reproducibility/data_splits/` 包含：

- `selected_classes.csv`：随机选择的 500 类；
- `class_mapping.json`：iNaturalist category ID 到连续类别编号的映射；
- `split_summary.json`：seed、类别数和样本数量；
- `train.csv`：20,000 张训练图片的相对路径；
- `val.csv`：5,000 张验证图片的相对路径；
- `test.csv`：5,000 张测试图片的相对路径。

这些文件不包含图片本身。组员仍需按项目 README 下载 iNaturalist-2021 数据集。

## 8. 如何重新生成

在仓库根目录运行：

```powershell
python scripts/export_final_results.py
```

默认读取：

```text
results/
analysis/error_analysis/
data_splits/
```

默认输出：

```text
outputs/final_results/inat500/
```

## 9. 没有上传的内容

为了控制仓库体积并符合最终代码 ZIP 要求，以下内容没有上传：

- `.pt` 和 `.joblib` 模型权重；
- iNaturalist 原始图片；
- HOG 特征缓存；
- 集成模型完整概率 `.npy`；
- smoke 测试；
- PID、stdout 和 stderr 日志。

如组员需要运行软件演示，应通过 Google Drive、OneDrive 或 GitHub Release 另外共享最终 checkpoint。
