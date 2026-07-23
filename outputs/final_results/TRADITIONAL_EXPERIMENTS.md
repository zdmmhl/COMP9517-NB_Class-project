# 传统方法补充实验说明

## 为什么补充这些实验

原始正式结果实际使用的是 HOG 与
`SGDClassifier(loss="hinge", max_iter=20)`。它属于线性 SVM 方法，但不是
`LinearSVC`。本次补充实验纠正了旧文档中的简称，并按相同 500 类数据划分比较
Color、LBP、HOG、SIFT-BoVW 特征以及 SGD linear SVM、LinearSVC 和
Random Forest 分类器。

## 公平性设置

- 固定随机种子：9517；
- 固定 train/validation/test：20,000 / 5,000 / 5,000；
- 特征和视觉词典只使用训练集拟合；
- 四种特征的主要比较固定使用同一个 SGD linear SVM；
- Random Forest 使用验证集表现最好的 HOG 特征；
- 测试集不参与特征或分类器选择。

## 真实测试结果

| 方法 | Top-1 | Top-5 | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Color + SGD linear SVM | 1.88% | 7.24% | 2.33% | 1.88% | 1.45% |
| LBP + SGD linear SVM | 0.46% | 1.78% | 0.49% | 0.46% | 0.26% |
| HOG + SGD linear SVM | 2.98% | 8.34% | 2.56% | 2.98% | 2.35% |
| SIFT-BoVW + SGD linear SVM | 1.22% | 4.82% | 0.93% | 1.22% | 0.66% |
| Color + LinearSVC | 3.06% | 10.12% | 2.19% | 3.06% | 1.84% |
| HOG + Random Forest (300 trees) | 2.48% | 6.72% | 1.82% | 2.48% | 1.92% |

所有数值来自各实验的 `metrics.json`，并已使用 5,000 行测试预测 CSV 重新计算
Top-1 和 Macro-F1。逐图片 Top-5 文件也各有 5,000 行。

机器可直接读取的完整表位于：

```text
comparison/traditional_summary_metrics.csv
```

对应图表位于：

```text
comparison/traditional_methods_comparison.png
```

每个方法的详细文件位于：

```text
methods/<method_key>/
```

其中包含配置、指标、逐图片 Top-1/Top-5、每类指标和混淆矩阵。模型
`.joblib`、特征缓存和原始图片没有上传。

## 主要观察

1. 在固定 SGD linear SVM 下，HOG 的验证集和测试集表现均为四种特征中最佳。
2. 全局 LBP 直方图几乎无法区分 500 个细粒度物种，属于有解释价值的负结果。
3. SIFT-BoVW 比 LBP 好，但低于 HOG；128-word 全局词袋会丢失关键点空间关系。
4. Random Forest 使用同一 HOG 特征训练约 1,531 秒，仍低于 SGD-SVM，增加训练
   成本没有带来性能收益。
5. Color + LinearSVC 获得最高传统 Top-1/Top-5，但 Macro-F1 低于 HOG +
   SGD-SVM，说明它在类别间表现更不均衡。
