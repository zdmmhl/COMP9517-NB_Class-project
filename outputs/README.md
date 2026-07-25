# 实验输出目录

本目录将“可重复生成的本地运行”和“准备共享的正式结果”分开：

```text
outputs/
|-- runs/
|   `-- inat500_seed9517/
`-- final_results/
    `-- inat500/
```

- `runs/<run_id>/`：评价过程产生的完整中间结果，不提交到 Git。
- `final_results/inat500/`：经过检查、用于报告和组内共享的500类正式结果。

当前 500 类传统手工特征与分类器补充实验请查看
[`final_results/inat500/TRADITIONAL_EXPERIMENTS.md`](final_results/inat500/TRADITIONAL_EXPERIMENTS.md)，
机器可读汇总位于
`final_results/inat500/comparison/traditional_summary_metrics.csv`。

该目录只跟踪体积较小的 CSV、JSON 和可视化图片，不包含：

- iNaturalist 原始图片；
- 模型权重；
- 特征缓存；
- 完整概率矩阵；
- smoke 测试结果；
- PID 和原始运行日志。

正式结果索引见 [`final_results/README.md`](final_results/README.md)。

重新整理当前实验结果：

```powershell
python scripts/export_final_results.py
```

当前固定划分包含500类，因此脚本默认导出到
`outputs/final_results/inat500/`。
