# 实验结果目录

`final_results/` 保存已经完成的真实实验结果，供组员查看、制作图表和撰写报告。

该目录只跟踪体积较小的 CSV、JSON 和可视化图片，不包含：

- iNaturalist 原始图片；
- 模型权重；
- 特征缓存；
- 完整概率矩阵；
- smoke 测试结果；
- PID 和原始运行日志。

详细说明请阅读 [`final_results/README.md`](final_results/README.md)。

重新整理当前实验结果：

```powershell
python scripts/export_final_results.py
```
