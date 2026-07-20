# 小型评估测试数据

这套fixture用于在真实模型结果到达前开发和测试评估模块。数据完全是人工生成的，不代表任何模型的真实性能。

## 数据规模

```text
类别数：6
每类test样本：5
总样本数：30
随机种子：42
```

类别被固定分成三组容易相互混淆的pair：

```text
class 0 <-> class 1
class 2 <-> class 3
class 4 <-> class 5
```

每个类别的5个样本中：

```text
2个Top-1正确
2个Top-1错误但Top-5正确
1个Top-5完全错误
```

因此预期结果是：

```text
Top-1 accuracy：0.40
Top-5 accuracy：0.80
Overall accuracy：0.40
Macro precision：0.40
Macro recall：0.40
Macro F1：0.40
Balanced accuracy：0.40
```

## 文件说明

```text
mock_small/
|-- images/                         # 30张可视化占位图片
|-- class_mapping.csv               # 6个模拟类别
|-- test.csv                        # Test manifest
|-- predictions.csv                 # 符合模型输出规约的Top-5预测
|-- metadata.json                   # 模拟运行信息
|-- history.csv                     # 包含轻微过拟合趋势的训练历史
|-- expected_metrics.json           # 评估代码应得到的指标
|-- expected_confusion_matrix.csv   # 评估代码应得到的混淆矩阵
`-- generate_fixture.py             # 确定性生成脚本
```

`image_path`以当前fixture目录为数据根目录，例如：

```text
images/class_0/sample_0.png
```

## 重新生成

在仓库根目录运行：

```bash
python tests/fixtures/mock_small/generate_fixture.py
```

重新生成后，所有CSV、JSON和图片应保持相同内容。

## 主要测试场景

这套数据可以验证：

- Top-1和Top-5计算。
- Macro和per-class指标。
- 6 x 6混淆矩阵。
- 最常混淆类别对提取。
- `sample_id`与test manifest对齐。
- Metadata和history读取。
- 训练曲线中的后期过拟合趋势。
- 成功和失败案例图片展示。
