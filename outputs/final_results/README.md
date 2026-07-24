# 正式实验结果索引

每套类别选择分别保存在独立目录中，避免 500 类、800 类或其他规模的结果互相覆盖。

当前目录：

```text
final_results/
|-- inat500/                 当前已完成的 500 类正式结果
|-- inat800/                 800 类结果完成后放在这里
`-- cross_setup_comparison/  不同类别规模之间的对比，按需要创建
```

目录命名只表示实验规模。准确的类别列表、数据划分和随机种子以各目录中的
`reproducibility/data_splits/` 为准。

本地评价过程保存在 `outputs/runs/<run_id>/`，不应直接混入本目录。只有经过检查、
准备用于报告的精简 CSV、JSON 和图片才导出到 `final_results/`。

当前 500 类结果说明见 [`inat500/README.md`](inat500/README.md)，SimpleCNN
与 ResNet18 的充分训练和收敛证据见
[`inat500/CONVERGENCE_EXPERIMENTS.md`](inat500/CONVERGENCE_EXPERIMENTS.md)。
