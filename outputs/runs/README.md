# 本地评价运行

本目录用于保存可重复生成的完整评价过程。GitHub 只跟踪这份说明，不跟踪各
run 的预测副本、中间表格和图片。

run ID 使用下面的格式：

```text
inat<类别数>_seed<数据划分随机种子>
```

例如：

```text
outputs/runs/
|-- inat500_seed9517/
|-- inat800_seed9517/
`-- mock_evaluation/
```

每个正式 run 可以包含：

```text
run_manifest.json
standardized_inputs/
methods/
comparison/
examples/
ablation_inputs/
ablations/
```

当前本地 500 类评价由以下命令重新生成：

```powershell
python scripts/evaluate_recorded_results.py
```

经过检查、需要在 GitHub 上共享或用于报告的精简结果，应导出到对应的
`outputs/final_results/inat<类别数>/`，不要直接提交整个 run。
