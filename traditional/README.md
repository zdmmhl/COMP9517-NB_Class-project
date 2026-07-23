# 传统手工特征实验

本模块实现 COMP9517 项目中的传统图像分类方法。所有正式实验固定使用同一份
iNaturalist 500 类数据划分：20,000 张训练图、5,000 张验证图和 5,000 张测试图，
随机种子为 9517。

## 实验设计

特征对比固定使用 `SGDClassifier(loss="hinge")` 作为可扩展的线性 SVM：

| 特征 | 主要信息 | 维度 |
|---|---|---:|
| HSV colour histogram | 全局颜色分布 | 96 |
| Uniform LBP | 局部纹理 | 26 |
| HOG | 形状与边缘方向 | 1,764 |
| SIFT Bag-of-Visual-Words | 局部关键点视觉词频 | 128 |

分类器对比根据**验证集**选择表现最好的 HOG 特征，再比较 SGD linear SVM 与
300-tree Random Forest。另运行了 Color Histogram + `LinearSVC`，用于检查不同
线性 SVM 优化器的影响。模型选择不读取测试集结果。

## 500 类正式结果

| 方法 | Val Top-1 | Test Top-1 | Test Top-5 | Test Macro-F1 |
|---|---:|---:|---:|---:|
| Color + SGD linear SVM | 1.78% | 1.88% | 7.24% | 1.45% |
| LBP + SGD linear SVM | 0.28% | 0.46% | 1.78% | 0.26% |
| HOG + SGD linear SVM | **2.98%** | **2.98%** | **8.34%** | **2.35%** |
| SIFT-BoVW + SGD linear SVM | 1.28% | 1.22% | 4.82% | 0.66% |
| Color + LinearSVC | 2.98% | 3.06% | 10.12% | 1.84% |
| HOG + Random Forest | 2.14% | 2.48% | 6.72% | 1.92% |

结论：HOG 是统一 SGD-SVM 对比中最有效的手工特征。Random Forest 在同一 HOG
特征上训练约 1,531 秒，但测试指标低于 SGD-SVM，说明增加分类器复杂度没有带来
收益。Color + LinearSVC 的 Top-1 略高，但 Macro-F1 低于 HOG + SGD-SVM。

## 运行方法

单项实验：

```powershell
python scripts/run_traditional_experiment.py `
  --feature hog `
  --classifier sgd-svm `
  --data-root datasets/inat2021 `
  --output-dir results/traditional_hog_sgd_svm_full `
  --max-iter 1000
```

支持的特征为 `color`、`lbp`、`hog` 和 `sift-bovw`；支持的分类器为
`sgd-svm`、`linear-svc` 和 `random-forest`。

正式 Random Forest 结果使用 `--rf-estimators 300`。代码默认设置为 100 棵树，
便于在普通 32 GB 内存机器上复现；300 棵树运行时峰值内存接近 20 GB。

## 输出文件

每个结果目录包含：

- `metrics.json`：配置、特征提取时间、训练时间及 train/val/test 指标；
- `*_predictions.csv`：逐图片 Top-1 预测；
- `*_predictions_top5.csv`：逐图片 Top-5 类别和分类分数；
- `*_classification_report.json`：500 类分类报告；
- `*_confusion_matrix.npy`：500 x 500 混淆矩阵；
- `model.joblib`：本地模型文件，不上传 GitHub。

GitHub 中的精简结果位于 `outputs/final_results/`，不包含模型、特征缓存和原始图片。
