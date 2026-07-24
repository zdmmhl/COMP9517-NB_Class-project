# SimpleCNN 与 ResNet18 充分训练和收敛实验

## 1. 为什么替换原结果

原 SimpleCNN 和 ResNet18 均只训练了 3 个 epoch。两者在第 3 轮时的验证
Macro-F1 仍在明显上升，不能证明模型已经充分训练或收敛：

| 模型 | Epoch 1 验证 Macro-F1 | Epoch 3 验证 Macro-F1 |
|---|---:|---:|
| SimpleCNN 原运行 | 0.66% | 1.99% |
| ResNet18 原运行 | 25.37% | 36.06% |

因此正式结果目录中的这两个方法已经原地替换为充分训练版本。旧 3-epoch 文件
不再保留在 `methods/simple_cnn/` 和 `methods/resnet18_pretrained/` 中。

## 2. 实验控制

两组实验使用完全相同的固定数据划分：

- 500 个类别；
- train/validation/test = 20,000 / 5,000 / 5,000；
- 随机种子 9517；
- 训练过程中不读取测试集；
- 根据验证集 Macro-F1 选择 checkpoint；
- 最终只在训练结束后用选中的 checkpoint 评估测试集。

训练代码新增了：

- 每轮保存 `last_checkpoint.pt`，包含模型、优化器、调度器和 AMP scaler；
- 支持 `--resume-checkpoint` 断点续训；
- 基于验证 Macro-F1 的早停和 `min_delta`；
- 每轮记录 loss、Top-1、Top-5、Macro-F1、学习率、吞吐量和 GPU 峰值；
- AMP、channels-last 和 cuDNN benchmark；
- 在 Windows 上训练结束后关闭 worker，再进行最终评估。

## 3. 训练配置

| 配置 | SimpleCNN | ResNet18 |
|---|---:|---:|
| 初始化 | 随机初始化 | ImageNet 预训练 |
| 输入大小 | 128 x 128 | 224 x 224 |
| 最大 Epoch | 50 | 30 |
| Train / Eval Batch | 512 / 1024 | 512 / 1024 |
| 学习率 | 0.002 | 0.001 |
| Backbone 学习率倍数 | 1.0 | 0.1 |
| Weight Decay | 0.0001 | 0.0001 |
| 调度器 | Cosine | Cosine |
| Label Smoothing | 0.1 | 0.1 |
| 数据增强 | Basic | Strong |
| Test-Time Augmentation | 否 | 是 |
| Early-stop Patience | 8 | 6 |
| Min Delta | 0.0005 | 0.0005 |

## 4. 替换前后真实测试结果

| 模型 | 版本 | Test Top-1 | Test Top-5 | Macro Precision | Macro Recall | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| SimpleCNN | 原 3 epoch | 3.94% | 12.86% | 2.57% | 3.94% | 2.17% |
| SimpleCNN | 新 50 epoch | **18.76%** | **40.04%** | **17.36%** | **18.76%** | **16.70%** |
| ResNet18 | 原 3 epoch | 37.26% | 64.16% | 44.33% | 37.26% | 36.09% |
| ResNet18 | 新 30 epoch | **66.08%** | **85.78%** | **67.04%** | **66.08%** | **65.52%** |

相对原结果，SimpleCNN 的 Test Top-1 提升 14.82 个百分点，ResNet18 提升
28.82 个百分点。新指标均由各自 5,000 行测试预测 CSV 重新计算核对，保存值
与反算值一致。

## 5. 收敛证据

### SimpleCNN

- 训练 loss：6.0485 降至 4.1852；
- 训练 Top-1：0.84% 升至 23.55%；
- 验证 Macro-F1：0.37% 升至 17.95%；
- 最佳 checkpoint：epoch 44，验证 Macro-F1 18.10%；
- epoch 44 至 50 的验证结果只小幅波动，已经进入平台期。

### ResNet18

- 训练 loss：5.5882 降至 1.3934；
- 训练 Top-1：7.71% 升至 95.48%；
- 验证 Macro-F1：20.22% 升至 66.78%；
- 按 `min_delta=0.0005` 选择 epoch 28 checkpoint，验证 Macro-F1 66.83%；
- 原始最高验证 Macro-F1 为 epoch 29 的 66.85%，只比 epoch 28 高
  0.02 个百分点，低于最小有效提升阈值；
- 最后数轮验证 Macro-F1 稳定在约 66%–67%。

这两组曲线证明：3 epoch 时模型尚未收敛，而延长训练后 loss 持续下降、
验证性能大幅提升并最终稳定。

## 6. 时间与 GPU 使用

| 模型 | 训练阶段总时间 | 验证阶段总时间 | PyTorch 峰值显存 |
|---|---:|---:|---:|
| SimpleCNN | 21 分 36 秒 | 2 分 38 秒 | 4,326 MiB |
| ResNet18 | 36 分 07 秒 | 8 分 25 秒 | 9,704 MiB |

显存来自每个 epoch 的 `torch.cuda.max_memory_allocated()`，不是人工估算。
正式训练前测试过更大的 batch，最终选择 512 以兼顾 GPU 利用率、每轮优化步数、
泛化能力和 Windows 页面文件稳定性。ResNet18 训练时 GPU 利用率可达到 100%。

## 7. 复现命令

下面命令假设数据位于 `datasets/inat2021`：

```powershell
python scripts/train_deep.py `
  --model simple-cnn `
  --data-root datasets/inat2021 `
  --output-dir results/simple_cnn_converged_full `
  --image-size 128 --epochs 50 `
  --batch-size 512 --eval-batch-size 1024 `
  --lr 0.002 --weight-decay 0.0001 `
  --scheduler cosine --min-lr-ratio 0.01 `
  --label-smoothing 0.1 --augmentation basic `
  --grad-clip 1.0 `
  --early-stopping-patience 8 `
  --early-stopping-min-delta 0.0005
```

```powershell
python scripts/train_deep.py `
  --model resnet18-pretrained `
  --data-root datasets/inat2021 `
  --output-dir results/resnet18_pretrained_converged_full `
  --image-size 224 --epochs 30 `
  --batch-size 512 --eval-batch-size 1024 `
  --lr 0.001 --backbone-lr-multiplier 0.1 `
  --weight-decay 0.0001 `
  --scheduler cosine --min-lr-ratio 0.01 `
  --label-smoothing 0.1 --augmentation strong `
  --grad-clip 1.0 --tta `
  --early-stopping-patience 6 `
  --early-stopping-min-delta 0.0005 `
  --num-workers 4 --eval-num-workers 0
```

中断后可以增加：

```powershell
--resume-checkpoint results/<方法目录>/last_checkpoint.pt
```

## 8. GitHub 中的结果位置

```text
methods/simple_cnn/
methods/resnet18_pretrained/
```

重点文件：

- `history.csv`：逐 epoch 原始数据；
- `training_curves.png`：收敛曲线；
- `metrics.csv`：最终测试指标；
- `configuration.csv`：完整实验配置；
- `test_predictions_top1.csv`：5,000 条测试预测；
- `per_class_metrics.csv`：500 类逐类指标；
- `confusion_matrix.csv/png`：混淆矩阵；
- `prediction_examples.png`：预测样例。

模型 checkpoint 保留在本机 `results/`，不进入普通 Git 提交。
