# 正式实验结果索引

本项目使用固定500类划分作为主基准，并在其结果包内保存一个独立的类别规模
Advanced研究。Advanced研究使用嵌套的500、1,000和2,500类子集，不替代主结果。

当前目录：

```text
final_results/
`-- inat500/
    |-- methods/               主500类方法结果
    |-- comparison/            主500类总体比较
    |-- ablations/             传统与深度学习消融
    `-- advanced/
        `-- class_scaling/     类别规模Advanced研究
```

主基准准确的类别列表、数据划分和随机种子以
`inat500/reproducibility/data_splits/` 为准；Advanced各子集使用各自
`advanced/class_scaling/reproducibility/<split_key>/` 中的清单。

本地评价过程保存在 `outputs/runs/<run_id>/`，不应直接混入本目录。只有经过检查、
准备用于报告的精简 CSV、JSON 和图片才导出到 `final_results/`。

当前 500 类结果说明见 [`inat500/README.md`](inat500/README.md)，SimpleCNN
与 ResNet18 的充分训练和收敛证据见
[`inat500/CONVERGENCE_EXPERIMENTS.md`](inat500/CONVERGENCE_EXPERIMENTS.md)。
深度学习消融与类别规模研究也由该README统一索引。

深度模型的逐图片 Top-5 类别、概率、独立复算记录和
“Top-1 错但 Top-5 对”案例均保存在各自方法或实验运行目录中。
