# 正式实验结果索引

本项目只使用固定的500类数据划分，不再预留其他类别规模的结果目录。

当前目录：

```text
final_results/
`-- inat500/  当前已完成的500类正式结果
```

准确的类别列表、数据划分和随机种子以该目录中的
`reproducibility/data_splits/` 为准。

本地评价过程保存在 `outputs/runs/<run_id>/`，不应直接混入本目录。只有经过检查、
准备用于报告的精简 CSV、JSON 和图片才导出到 `final_results/`。

当前 500 类结果说明见 [`inat500/README.md`](inat500/README.md)，SimpleCNN
与 ResNet18 的充分训练和收敛证据见
[`inat500/CONVERGENCE_EXPERIMENTS.md`](inat500/CONVERGENCE_EXPERIMENTS.md)。
