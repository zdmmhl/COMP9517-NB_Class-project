# 实验输出目录

本目录将“可重复生成的本地运行”和“准备共享的正式结果”分开：

```text
outputs/
|-- runs/
|   `-- inat500_seed9517/
`-- final_results/
    `-- inat500/
        |-- methods/
        |-- comparison/
        |-- ablations/
        |   |-- deep_learning/
        |   `-- <traditional studies>/
        `-- advanced/
            `-- class_scaling/
```

- `runs/<run_id>/`：评价过程产生的完整中间结果，不提交到 Git。
- `final_results/inat500/methods/` 和 `comparison/`：主500类基准结果。
- `final_results/inat500/ablations/`：传统方法与ResNet18控制变量消融。
- `final_results/inat500/advanced/class_scaling/`：独立的500/1,000/2,500类
  Advanced规模实验，不并入主模型排名。

当前 500 类传统手工特征与分类器补充实验请查看
[`final_results/inat500/TRADITIONAL_EXPERIMENTS.md`](final_results/inat500/TRADITIONAL_EXPERIMENTS.md)，
机器可读汇总位于
`final_results/inat500/comparison/traditional_summary_metrics.csv`。

深度学习消融说明位于
`final_results/inat500/ablations/deep_learning/深度学习消融实验说明.md`；
类别规模Advanced实验说明位于
`final_results/inat500/advanced/class_scaling/类别规模Advanced实验说明.md`。

深度模型目录同时保存逐图 Top-5 类别和概率、独立复算结果，以及
“Top-1 错但 Top-5 对”的 CSV 与案例图。

该目录跟踪精简 CSV、JSON 和报告用可视化图片。部分案例图可能达到数 MB，因此
整个 `final_results/` 只用于组内共享，不放入最终 code ZIP。该目录不包含：

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
python scripts/export_deep_ablations.py
python scripts/export_class_scaling.py
python scripts/write_experiment_docs.py
python scripts/refresh_final_comparison.py
```

主结果、消融和Advanced结果分别由上述导出脚本维护，但统一收纳在
`outputs/final_results/inat500/` 项目结果包中。
`refresh_final_comparison.py` 只读取已跟踪的 `methods/*/metrics.csv` 和
`per_class_metrics.csv` 生成主对比结果，同时从正式消融、类别规模指标和各实验
当前的 `test_predictions_top1.csv` 重画汇总图及紧凑混淆矩阵，保证正式图片与
最新 CSV 一致。
