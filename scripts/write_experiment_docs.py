"""Write Chinese, result-backed documentation for DL ablations and scaling."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = PROJECT_ROOT / "outputs" / "final_results" / "inat500"
ABLATION_ROOT = FINAL_ROOT / "ablations" / "deep_learning"
SCALING_ROOT = FINAL_ROOT / "advanced" / "class_scaling"


def percent(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def delta_points(value: float) -> str:
    return f"{100 * float(value):+.2f}"


def write_deep_ablation_doc() -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "deep_ablations.json").read_text(
            encoding="utf-8"
        )
    )
    lines = [
        "# 深度学习控制变量消融实验",
        "",
        "## 实验目的",
        "",
        "本实验严格遵循一次只改变一个变量的原则，使用固定的500类数据划分和"
        "ResNet18训练预算，分别研究初始化、数据增强、Label Smoothing、MixUp"
        "和测试时增强（TTA）。所有结果均由实际测试集预测CSV重新计算验证。",
        "",
        "## 固定设置",
        "",
        "| 设置 | 数值 |",
        "|---|---:|",
    ]
    for key in [
        "image_size",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "lr",
        "weight_decay",
        "scheduler",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "cuda_memory_fraction",
    ]:
        lines.append(f"| `{key}` | `{config['common'][key]}` |")
    lines.extend(
        [
            f"| 随机种子 | `{config['seed']}` |",
            "| 数据划分 | 500类；20,000训练、5,000验证、5,000测试 |",
            "",
            "## 实验结果",
            "",
        ]
    )

    interpretations = {
        "initialization": (
            "结论：ImageNet预训练带来最大的性能提升，并显著加快收敛；"
            "在每类仅40张训练图片时，从随机初始化学习完整视觉表征明显受限。"
        ),
        "augmentation": (
            "结论：Basic augmentation优于当前Strong方案，说明RandAugment与"
            "Random Erasing的组合对这个小样本细粒度任务偏强，可能破坏了"
            "区分相近物种所需的局部纹理。"
        ),
        "label_smoothing": (
            "结论：Label Smoothing=0.1小幅提高Top-1和Macro-F1，但Top-5下降；"
            "它改善了最佳类别判断，却没有同时改善候选类别排序。"
        ),
        "mixup": (
            "结论：MixUp alpha=0.2在三个主要指标上均有提升，是有效但幅度"
            "较温和的训练正则化。"
        ),
        "tta": (
            "结论：对完全相同的最佳checkpoint使用水平翻转TTA，无需重新训练"
            "即可获得本组最大的额外增益；代价是测试时需要两次前向计算。"
        ),
    }
    for study in config["studies"]:
        study_dir = ABLATION_ROOT / study["key"]
        summary = pd.read_csv(study_dir / "ablation_summary.csv")
        deltas = pd.read_csv(study_dir / "ablation_deltas.csv")
        lines.extend(
            [
                f"### {study['study_name']}",
                "",
                f"唯一变化变量：`{study['factor']}`。",
                "",
                "| 方案 | Top-1 | Top-5 | Macro-F1 | 运行次数 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['display_name']} | "
                f"{percent(row['top1_accuracy_mean'])} | "
                f"{percent(row['top5_accuracy_mean'])} | "
                f"{percent(row['macro_f1_mean'])} | "
                f"{int(row['n_runs'])} |"
            )
        candidate = deltas.iloc[-1]
        lines.extend(
            [
                "",
                "相对表中基线，最后一个方案的变化为："
                f"Top-1 `{delta_points(candidate['delta_top1_accuracy'])}` 个百分点，"
                f"Top-5 `{delta_points(candidate['delta_top5_accuracy'])}` 个百分点，"
                f"Macro-F1 `{delta_points(candidate['delta_macro_f1'])}` 个百分点。",
                "",
                interpretations[study["key"]],
                "",
                f"![{study['study_name']}指标](./{study['key']}/ablation_metrics.png)",
                "",
                f"![{study['study_name']}训练曲线](./{study['key']}/training_curves.png)",
                "",
            ]
        )

    lines.extend(
        [
            "## 文件说明",
            "",
            "- 每个子目录的 `study.json` 定义唯一允许变化的参数。",
            "- `validation_report.json` 记录控制变量检查结果。",
            "- `ablation_runs.csv` 保存每次真实运行的指标和路径。",
            "- `ablation_summary.csv` 与 `ablation_deltas.csv` 用于报告制表。",
            "- 每个方案目录保存配置、训练历史、预测、每类指标和必要图表。",
            "",
            "## 限制",
            "",
            "受计算预算限制，每个配置当前使用一个固定随机种子，因此这里展示的是"
            "严格可复现的单次受控比较，不能将差异解释为多随机种子统计显著性结论。",
            "",
        ]
    )
    (ABLATION_ROOT / "深度学习消融实验说明.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_scaling_doc() -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "class_scaling.json").read_text(
            encoding="utf-8"
        )
    )
    summary = pd.read_csv(SCALING_ROOT / "scaling_summary.csv")
    indexed = summary.set_index("run_key")

    def change(left: str, right: str, metric: str) -> str:
        return delta_points(
            indexed.loc[right, metric] - indexed.loc[left, metric]
        )
    lines = [
        "# Advanced：类别数量扩展实验",
        "",
        "## 对应作业要求",
        "",
        "本实验对应项目说明中的 Advanced Method Development 第3项"
        "“Effect of the number of classes”。基础结果仍然报告在原500类测试集上；"
        "1,000类和2,500类数据仅用于规模扩展研究。",
        "",
        "## 数据构造",
        "",
        "随机种子固定为 `9517`。1,000类完整包含基础500类，2,500类完整包含"
        "前1,000类。主实验保持训练、验证和测试图片总量分别约为20,000、5,000"
        "和5,000，并随类别增加减少每类图片数。",
        "",
        "| 实验 | 类型 | 类别数 | 训练/类 | 验证/类 | 测试/类 | 训练总数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['run_key']}` | {row['kind']} | "
            f"{int(row['num_classes'])} | {int(row['train_per_class'])} | "
            f"{int(row['val_per_class'])} | {int(row['test_per_class'])} | "
            f"{int(row['total_train_images'])} |"
        )
    lines.extend(
        [
            "",
            "## 固定模型与训练设置",
            "",
            "所有规模使用同一个ImageNet预训练ResNet18配置并独立从相同预训练权重"
            "开始，不从500类checkpoint继续训练。除数据划分和输出类别数外，优化器、"
            "学习率、增强、epoch上限、early-stop规则和随机种子保持一致。",
            "",
            "## 实际结果",
            "",
            "| 实验 | 最佳/完成epoch | Top-1 | Top-5 | Macro-F1 | 训练时间 | 峰值显存 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['run_key']}` | {int(row['best_epoch'])}/"
            f"{int(row['completed_epochs'])} | {percent(row['top1_accuracy'])} | "
            f"{percent(row['top5_accuracy'])} | {percent(row['macro_f1'])} | "
            f"{float(row['training_time_seconds']) / 60:.1f}分钟 | "
            f"{float(row['peak_gpu_memory_mb']):.0f} MiB |"
        )
    lines.extend(
        [
            "",
            "![类别规模与性能](./class_scaling_metrics.png)",
            "",
            "![固定500类的单类样本量控制](./sample_size_control.png)",
            "",
            "![运行时间](./runtime_comparison.png)",
            "",
            "## 如何解读",
            "",
            "- `classes_500 → classes_1000 → classes_2500` 同时体现类别更细和"
            "单类样本减少的综合影响，同时总训练图片量基本固定。",
            "- `classes_500 → control_500x20 → control_500x8` 固定500类，"
            "用于单独观察每类训练样本减少造成的影响。",
            "- 对比 `control_500x20` 与 `classes_1000`，以及"
            "`control_500x8` 与 `classes_2500`，可以辅助判断类别数量增加的影响。",
            "",
            "## 结果分析",
            "",
            f"- 主规模实验从500类增加到1,000类时，Top-1变化"
            f"`{change('classes_500', 'classes_1000', 'top1_accuracy')}`个百分点，"
            f"Macro-F1变化"
            f"`{change('classes_500', 'classes_1000', 'macro_f1')}`个百分点。",
            f"- 从1,000类继续增加到2,500类时，Top-1变化"
            f"`{change('classes_1000', 'classes_2500', 'top1_accuracy')}`个百分点，"
            f"Macro-F1变化"
            f"`{change('classes_1000', 'classes_2500', 'macro_f1')}`个百分点，"
            "下降幅度进一步扩大。",
            f"- 固定500类，仅把每类训练图片从40减到20时，Top-1变化"
            f"`{change('classes_500', 'control_500x20', 'top1_accuracy')}`个百分点；"
            f"减到8张时变化"
            f"`{change('classes_500', 'control_500x8', 'top1_accuracy')}`个百分点。"
            "这证明单类样本减少本身就是主要性能瓶颈。",
            f"- 固定每类20张，对比500类和1,000类，类别翻倍额外造成Top-1"
            f"`{change('control_500x20', 'classes_1000', 'top1_accuracy')}`个百分点、"
            f"Macro-F1"
            f"`{change('control_500x20', 'classes_1000', 'macro_f1')}`个百分点变化。",
            f"- 固定每类8张，对比500类和2,500类，类别增加额外造成Top-1"
            f"`{change('control_500x8', 'classes_2500', 'top1_accuracy')}`个百分点、"
            f"Macro-F1"
            f"`{change('control_500x8', 'classes_2500', 'macro_f1')}`个百分点变化。",
            "",
            "综合来看，类别增多和单类样本减少都会降低性能；在本实验中，"
            "单类样本降到8张造成的影响尤其明显，而类别扩展又进一步加剧了"
            "细粒度物种之间的混淆。",
            "",
            "## 可复现文件",
            "",
            "每个子集都保存 `train.csv`、`val.csv`、`test.csv`、"
            "`selected_classes.csv`、`class_mapping.json` 和 `split_summary.json`。"
            "`split_summary.json` 记录数量、随机种子和SHA-256；"
            "`reproducibility/validation_report.json` 记录嵌套与数据交叉检查。",
            "",
            "所有测试指标均与 `test_predictions_top1.csv` 重新计算的Top-1和"
            "Macro-F1核对，结果见各运行目录的 `prediction_validation.json`。",
            "",
            "本实验使用一个固定随机种子，结论适合作为严格可复现的受控趋势，"
            "但不等同于多随机种子的统计显著性分析。",
            "",
        ]
    )
    (SCALING_ROOT / "类别规模Advanced实验说明.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    write_deep_ablation_doc()
    write_scaling_doc()
    print("Wrote Chinese experiment documentation.")


if __name__ == "__main__":
    main()
