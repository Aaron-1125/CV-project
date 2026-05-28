#!/usr/bin/env python3
"""Write Stage2 task 6.x reports and the week 2 Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def metric_row(name: str, metrics: dict[str, Any], size_mb: Any) -> str:
    speed = metrics.get("embedding_speed", {})
    return (
        f"| {name} | {pct(metrics.get('accuracy'))} | {num(metrics.get('roc_auc'))} | "
        f"{num(speed.get('latency_ms_per_image'), 3)} | {num(speed.get('throughput_images_per_second'), 2)} | "
        f"{num(size_mb, 2)} |"
    )


def write_optimization_survey(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Task 6.1 模型优化技术调研

## 动态量化

动态量化在推理时把部分权重转换为 int8，并在运行中动态处理激活值。它适合 `Linear`、RNN 等矩阵乘法占比较高的模块，部署简单，不需要重新训练或校准数据。Task6 使用 PyTorch `quantize_dynamic` 对 ArcFace backbone 里的 `Linear` 层进行量化。由于 IResNet50 的主要计算来自卷积层，动态量化的体积和速度收益预计有限，但可以作为最小侵入的模型压缩 baseline。

## 剪枝

剪枝通过删除冗余通道、卷积核或权重连接来减少计算量。非结构化剪枝能直接制造稀疏权重，但通用硬件未必能获得明显速度收益；结构化通道剪枝更适合端侧部署，但通常需要剪枝后微调以恢复精度。对人脸识别模型而言，剪枝要重点监控 embedding 角度分布和 LFW/业务验证集准确率，避免压缩后类间间隔变小。

## 蒸馏

知识蒸馏使用高精度 teacher 模型指导较小 student 模型学习。人脸识别中常见做法是让 student 同时学习分类损失、ArcFace margin 约束，以及 teacher embedding 的余弦相似度或特征距离。蒸馏通常比单纯剪枝更稳，但需要额外 teacher checkpoint 和重新训练成本。

## ONNX 部署

ONNX 将 PyTorch 模型导出为跨框架计算图，便于使用 ONNX Runtime 做 CPU/GPU 推理，也方便后续接入 TensorRT、OpenVINO 等推理引擎。Task6 导出的是 ArcFace embedding backbone，输入为 `N x 3 x 112 x 112`，输出为 `N x 512` 归一化 embedding。
""",
        encoding="utf-8",
    )


def write_task6_report(path: Path, source: dict[str, Any], quant: dict[str, Any], onnx: dict[str, Any]) -> None:
    fp32 = quant.get("fp32", {})
    q = quant.get("dynamic_quantized", {})
    onnx_metrics = onnx.get("onnx", {})
    pytorch_ref = onnx.get("pytorch_reference", {})
    rows = [
        metric_row("FP32 backbone (Task5 first version)", fp32.get("metrics", {}), fp32.get("model_size_mb")),
        metric_row("Dynamic quantized INT8", q.get("metrics", {}), q.get("model_size_mb")),
        metric_row("ONNX Runtime", onnx_metrics, onnx.get("onnx_size_mb")),
    ]
    source_acc = source.get("source_lfw_accuracy", source.get("source_best_lfw_accuracy"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Stage2 Task 6.x 人脸识别模型压缩与 ONNX 推理报告

## 1. 任务目标

本任务完成 6.1、6.2、6.3，不包含可选 6.4。实验模型固定为 Task5 第一版自实现 `IResNet50 + ArcFace`，来源于云端结果包 `reports/task5/task5_cloud_results_8167.tar.gz` 中的 `best.pth`。该模型在 LFW 6000-pair 10-fold protocol 上的云端基线准确率为 `{pct(source_acc)}`。

## 2. 量化、剪枝与蒸馏调研

量化用于降低权重精度和模型体积；剪枝用于删除冗余结构；蒸馏用于把大模型的 embedding 判别能力迁移到小模型。Task6 的实现重点是 PyTorch 动态量化和 ONNX 推理，调研正文见 `task6_1_optimization_methods.md`。

## 3. 动态量化实验

动态量化通过 `torch.quantization.quantize_dynamic(model, {{torch.nn.Linear}}, dtype=torch.qint8)` 完成，只作用于 `Linear` 层。IResNet50 的计算主体是卷积层，所以该实验重点观察是否能在不明显损失 LFW 精度的前提下降低全连接层相关存储与 CPU 推理成本。

![动态量化对比图](assets/evaluation/task6_quantization_comparison.png)

## 4. ONNX 导出与推理

ONNX 导出使用动态 batch 维度，输入为 `N x 3 x 112 x 112`，输出为 `N x 512` embedding。ONNX Runtime 推理后重新做 L2 normalization，并使用同一 LFW pair protocol 计算 accuracy、ROC AUC 和推理速度。

![ONNX 对比图](assets/evaluation/task6_onnx_comparison.png)

## 5. 性能对比

| 模型 | LFW accuracy | ROC AUC | latency ms/image | throughput img/s | model size MB |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

ONNX 数值一致性：mean cosine `{num(onnx.get('consistency', {}).get('mean_cosine'), 6)}`，max abs diff `{num(onnx.get('consistency', {}).get('max_abs_diff'), 6)}`。

## 6. 结论

Task6 已完成第一版 ArcFace 模型的动态量化、量化前后性能对比、ONNX 导出和 ONNX Runtime 推理验证。由于源模型本身 LFW baseline 为 `{pct(source_acc)}`，Task6 的重点不是继续提升识别精度，而是验证压缩和部署链路是否保持与源模型可比的 embedding 行为。

## 7. 复现实验命令

```bash
python code/task6/stage2_task6_prepare_source_model.py
python code/task6/stage2_task6_2_quantize_arcface.py
python code/task6/stage2_task6_3_export_onnx.py
python code/task6/stage2_task6_write_reports.py
python code/task6/stage2_task6_export_weekly_pdf.py \\
  --source reports/weekly/week2_report_2026-05-28.md \\
  --output reports/weekly/week2_report_2026-05-28.pdf
```
""",
        encoding="utf-8",
    )


def write_weekly_report(path: Path, source: dict[str, Any], quant: dict[str, Any], onnx: dict[str, Any]) -> None:
    fp32_metrics = quant.get("fp32", {}).get("metrics", {})
    q_metrics = quant.get("dynamic_quantized", {}).get("metrics", {})
    onnx_metrics = onnx.get("onnx", {})
    source_acc = source.get("source_lfw_accuracy", source.get("source_best_lfw_accuracy"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# 第二周周报：阶段二人脸识别训练、压缩与部署

周期：第二周，2026-05-22 至 2026-05-28

## 1. 本周已完成

- 完成 Stage2 Task3.x WIDER FACE 人脸检测训练链路整理，并将检测任务报告、训练曲线、检测可视化保留在 Task3 独立目录。
- 完成 Stage2 Task4.x 300W 人脸关键点检测与仿射对齐交付，输出关键点 overlay、对齐图和任务报告。
- 完成 Stage2 Task5.x 第一版 `IResNet50 + ArcFace` 训练和 LFW 评估。云端第一版 checkpoint 的 LFW accuracy 为 `{pct(source_acc)}`，作为 Task6 模型压缩基线。
- 完成 Stage2 Task6.1/6.2/6.3：模型优化技术调研、PyTorch 动态量化、ONNX 导出与 ONNX Runtime 推理验证。
- 按任务隔离原则整理交付物：Task6 代码在 `code/task6/`，报告在 `reports/task6/`，模型和 ONNX artifact 在 ignored 的 `work_dirs/task6/`。

## 2. 运行截图

![Task5 云端训练曲线](../task6/source_task5/assets/ms1mv3_dense_loss_acc_curve.png)

![Task5 云端 LFW ROC](../task6/source_task5/assets/lfw_roc_curve.png)

![Task6 动态量化对比](../task6/assets/evaluation/task6_quantization_comparison.png)

![Task6 ONNX 推理对比](../task6/assets/evaluation/task6_onnx_comparison.png)

---PAGEBREAK---

## 3. 实验结果与图表

### 3.1 Task5 第一版 ArcFace 基线

Task6 使用的源模型来自 `reports/task5/task5_cloud_results_8167.tar.gz`，训练集为 dense MS1MV3 子集，云端训练记录显示最终使用 `800000` 张图片、`20000` 个 identities、`60` 个 epoch。

### 3.2 动态量化结果

| 模型 | LFW accuracy | latency ms/image | model size MB |
|---|---:|---:|---:|
| FP32 | {pct(fp32_metrics.get('accuracy'))} | {num(fp32_metrics.get('embedding_speed', {}).get('latency_ms_per_image'), 3)} | {num(quant.get('fp32', {}).get('model_size_mb'), 2)} |
| Dynamic INT8 | {pct(q_metrics.get('accuracy'))} | {num(q_metrics.get('embedding_speed', {}).get('latency_ms_per_image'), 3)} | {num(quant.get('dynamic_quantized', {}).get('model_size_mb'), 2)} |

### 3.3 ONNX 推理结果

| 模型 | LFW accuracy | latency ms/image | model size MB |
|---|---:|---:|---:|
| ONNX Runtime | {pct(onnx_metrics.get('accuracy'))} | {num(onnx_metrics.get('embedding_speed', {}).get('latency_ms_per_image'), 3)} | {num(onnx.get('onnx_size_mb'), 2)} |

ONNX 与 PyTorch embedding 对比：mean cosine `{num(onnx.get('consistency', {}).get('mean_cosine'), 6)}`，max abs diff `{num(onnx.get('consistency', {}).get('max_abs_diff'), 6)}`。

---PAGEBREAK---

## 4. 关键代码段与解释

### 4.1 Task6 源模型隔离

文件：`code/task6/stage2_task6_prepare_source_model.py`

```python
extract_member(tar, TASK5_BEST, best_out)
extract_member(tar, TASK5_LFW_SUMMARY, lfw_summary_out)
extract_member(tar, TASK5_TRAIN_SUMMARY, train_summary_out)
```

解释：Task6 不直接覆盖 Task5 的 `work_dirs/`，而是从云端 tar 包中只提取第一版 `best.pth` 和 summary 到 Task6 专用目录，保证后续量化和 ONNX 实验不会误用本地旧 checkpoint。

### 4.2 动态量化

文件：`code/task6/stage2_task6_2_quantize_arcface.py`

```python
quantized = torch.quantization.quantize_dynamic(
    fp32_backbone.cpu(), {{torch.nn.Linear}}, dtype=torch.qint8
)
```

解释：PyTorch 动态量化仅量化 `Linear` 层，适合快速得到 CPU 推理 baseline。由于 IResNet50 以卷积为主，报告中同时记录精度、速度和模型体积，避免只看单一指标。

### 4.3 ONNX 导出

文件：`code/task6/stage2_task6_3_export_onnx.py`

```python
torch.onnx.export(
    backbone,
    dummy,
    onnx_out,
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={{"input": {{0: "batch"}}, "embedding": {{0: "batch"}}}},
)
```

解释：导出的 ONNX 模型保留动态 batch 维度，便于在 ONNX Runtime 中按不同 batch size 做推理。输出 embedding 继续使用 LFW 6000-pair 10-fold protocol 评估。

## 5. 下周待办

- 若需要继续提升 Task5 精度，优先切换到官方 InsightFace full MS1MV3 RecordIO 路线，而不是继续扩大第一版自实现训练。
- 在 Task6 基础上尝试静态量化、结构化剪枝或 TensorRT/ONNX Runtime GPU provider，加速效果会比仅动态量化 `Linear` 层更明显。
- 整理最终提交目录，确认 `data/`、`work_dirs/`、`.pth`、`.onnx` 继续被 Git 忽略，只提交代码、报告、summary、PDF 和小体积图表。
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, default=Path("reports/task6/summaries/source_model_summary.json"))
    parser.add_argument("--quant-summary", type=Path, default=Path("reports/task6/summaries/quantization_summary.json"))
    parser.add_argument("--onnx-summary", type=Path, default=Path("reports/task6/summaries/onnx_summary.json"))
    parser.add_argument("--survey-out", type=Path, default=Path("reports/task6/task6_1_optimization_methods.md"))
    parser.add_argument("--task6-report-out", type=Path, default=Path("reports/task6/stage2_task6_model_optimization_report.md"))
    parser.add_argument("--weekly-out", type=Path, default=Path("reports/weekly/week2_report_2026-05-28.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = read_json_optional(args.source_summary)
    quant = read_json_optional(args.quant_summary)
    onnx = read_json_optional(args.onnx_summary)
    write_optimization_survey(args.survey_out)
    write_task6_report(args.task6_report_out, source, quant, onnx)
    write_weekly_report(args.weekly_out, source, quant, onnx)
    print(f"Wrote {args.survey_out}")
    print(f"Wrote {args.task6_report_out}")
    print(f"Wrote {args.weekly_out}")


if __name__ == "__main__":
    main()
