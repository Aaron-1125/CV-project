#!/usr/bin/env python3
"""Write Stage2 Task6 reports and the week 2 Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent
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
        f"{num(speed.get('latency_ms_per_image'), 3)} | "
        f"{num(speed.get('throughput_images_per_second'), 2)} | {num(size_mb, 2)} |"
    )


def best_lfw_epoch(task5_train: dict[str, Any]) -> tuple[Any, Any]:
    history = task5_train.get("history") or []
    if not history:
        return None, task5_train.get("best_lfw_accuracy")
    best = max(history, key=lambda row: row.get("lfw_accuracy", -1))
    return best.get("epoch"), best.get("lfw_accuracy")


def write_optimization_survey(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        dedent(
            """\
            # Task 6.1 模型优化技术调研

            ## 动态量化

            动态量化在推理时把部分权重转换为 int8，并在运行过程中动态处理激活值。它适合 `Linear`、RNN 等矩阵乘法占比较高的模块，部署简单，不需要重新训练或额外校准数据。Task6 使用 PyTorch `quantize_dynamic` 对 ArcFace backbone 中的 `Linear` 层进行动态量化。由于 IResNet50 的主要计算来自卷积层，动态量化的体积和速度收益有限，但它可以作为最小侵入的压缩 baseline。

            ## 剪枝

            剪枝通过删除冗余通道、卷积核或权重连接来减少计算量。非结构化剪枝能制造稀疏权重，但通用硬件未必能直接获得速度收益；结构化通道剪枝更适合端侧部署，但通常需要剪枝后微调来恢复精度。人脸识别模型剪枝时要重点监控 embedding 角度分布和 LFW/业务验证集准确率，避免压缩后类间间隔变小。

            ## 蒸馏

            知识蒸馏使用高精度 teacher 模型指导较小 student 模型学习。人脸识别中常见做法是让 student 同时学习分类损失、ArcFace margin 约束，以及 teacher embedding 的余弦相似度或特征距离。蒸馏通常比单纯剪枝更稳，但需要额外 teacher checkpoint 和重新训练成本。

            ## ONNX 部署

            ONNX 将 PyTorch 模型导出为跨框架计算图，便于使用 ONNX Runtime 做 CPU/GPU 推理，也方便后续接入 TensorRT、OpenVINO 等推理引擎。Task6 导出的是 ArcFace embedding backbone，输入为 `N x 3 x 112 x 112`，输出为 `N x 512` 归一化 embedding。
            """
        ),
        encoding="utf-8",
    )


def write_task6_report(path: Path, source: dict[str, Any], quant: dict[str, Any], onnx: dict[str, Any]) -> None:
    fp32 = quant.get("fp32", {})
    q = quant.get("dynamic_quantized", {})
    onnx_metrics = onnx.get("onnx", {})
    source_acc = source.get("source_lfw_accuracy", source.get("source_best_lfw_accuracy", fp32.get("metrics", {}).get("accuracy")))
    rows = [
        metric_row("FP32 backbone (Task5 first version)", fp32.get("metrics", {}), fp32.get("model_size_mb")),
        metric_row("Dynamic quantized INT8", q.get("metrics", {}), q.get("model_size_mb")),
        metric_row("ONNX Runtime", onnx_metrics, onnx.get("onnx_size_mb")),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        dedent(
            f"""\
            # Stage2 Task 6.x 人脸识别模型压缩与 ONNX 推理报告

            ## 1. 任务目标

            本任务完成 6.1、6.2、6.3，不包含可选 6.4。实验模型固定为 Task5 第一版自实现 `IResNet50 + ArcFace`，来源于云端结果包 `reports/task5/task5_cloud_results_8167.tar.gz` 中的 `best.pth`。该模型在 LFW 6000-pair 10-fold protocol 上的云端基线准确率约为 `{pct(source_acc)}`。

            ## 2. 优化方法调研

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

            Task6 已完成第一版 ArcFace 模型的动态量化、量化前后性能对比、ONNX 导出和 ONNX Runtime 推理验证。由于源模型本身的 LFW baseline 约为 `{pct(source_acc)}`，Task6 的重点不是继续提升识别精度，而是验证压缩和部署链路是否保持与源模型可比较的 embedding 行为。
            """
        ),
        encoding="utf-8",
    )


def write_weekly_report(
    path: Path,
    task5_train: dict[str, Any],
    task5_eval: dict[str, Any],
    source: dict[str, Any],
    quant: dict[str, Any],
    onnx: dict[str, Any],
) -> None:
    fp32_metrics = quant.get("fp32", {}).get("metrics", {})
    q_metrics = quant.get("dynamic_quantized", {}).get("metrics", {})
    onnx_metrics = onnx.get("onnx", {})
    source_acc = task5_train.get(
        "best_lfw_accuracy",
        source.get("source_lfw_accuracy", source.get("source_best_lfw_accuracy", fp32_metrics.get("accuracy"))),
    )
    best_epoch, best_acc = best_lfw_epoch(task5_train)
    eval_metrics = task5_eval.get("metrics", {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        dedent(
            f"""\
            # 第二周周报：Stage2 人脸检测、关键点、识别与部署

            周期：第二周，2026-05-22 至 2026-05-28

            ## 1. 本周已完成

            - 完成 Stage2 Task3.x WIDER FACE 人脸检测交付，整理 GPU Docker、数据准备、全量训练、全量评估、检测可视化和独立报告目录。
            - 完成 Stage2 Task4.x 300W 人脸关键点检测与仿射对齐交付，输出 HRNet 训练/评估结果、关键点 overlay、aligned face 和 before/after grid。
            - 完成 Stage2 Task5.x 第一版 `IResNet50 + ArcFace` 云端训练与 LFW 验证同步。当前展示层已统一为云端 dense 结果：`{task5_train.get('num_images', 'N/A')}` 张图、`{task5_train.get('num_identities', 'N/A')}` 个 identities、`{task5_train.get('epochs_completed', 'N/A')}` 个 epoch，best LFW accuracy `{pct(source_acc)}`。
            - 完成 Stage2 Task6.1/6.2/6.3：模型优化方法调研、PyTorch 动态量化、ONNX 导出、ONNX Runtime 推理和 LFW protocol 对比。
            - 按任务隔离原则整理交付物：Task5 展示层在 `reports/task5/`，Task6 代码在 `code/task6/`，Task6 报告在 `reports/task6/`，模型权重与 ONNX 文件保留在 ignored 的 `work_dirs/task6/`。

            ## 2. 运行截图

            以下截图均为本机 PowerShell/Windows Terminal 真实运行命令后的界面；本节只放终端运行画面，不放实验结果图表。

            ![运行截图：GPU Docker 环境验证，显示 CUDA、MMDetection、MMPose、Task5/Task6 依赖可用。](../assets/weekly/week2/terminal_environment_check.png)

            ![运行截图：Task5 云端 800k/60 epoch/0.8167 summary 已同步到 reports/task5。](../assets/weekly/week2/terminal_task5_cloud_sync.png)

            ![运行截图：Task6 动态量化与 ONNX summary 检查，显示 FP32、INT8、ONNX 的 LFW 精度和模型体积。](../assets/weekly/week2/terminal_task6_eval_summary.png)

            ![运行截图：短 demo 训练命令真实启动并跑完，用于证明训练链路可运行，不作为正式指标来源。](../assets/weekly/week2/terminal_demo_training.png)

            ---PAGEBREAK---

            ## 3. 实验结果与图表

            ### 3.1 Task3 WIDER FACE 人脸检测

            Task3 目标是跑通 WIDER FACE 人脸检测训练与评估链路，并把检测图、训练曲线、评估指标放入 `reports/task3/`，不与后续关键点或识别任务混在一起。

            ![Task3 全量训练 loss 曲线。](../task3/assets/training/full_loss_curve.png)

            ![Task3 WIDER FACE 评估指标图。](../task3/assets/evaluation/widerface_full_eval_metrics.png)

            ![Task3 检测可视化示例。](../task3/assets/detection/detection_00_0_Parade_Parade_0_102.jpg)

            ### 3.2 Task4 300W 关键点检测与对齐

            Task4 使用 300W + HRNetv2-W18 完成人脸 68 点关键点检测，并基于眼睛、鼻尖和嘴角估计仿射矩阵，把人脸对齐到 112x112 ArcFace 模板。

            ![Task4 HRNet 训练 loss 曲线。](../task4/assets/training/300w_full_loss_curve.png)

            ![Task4 300W NME 评估指标图。](../task4/assets/evaluation/300w_nme_metrics.png)

            ![Task4 对齐 before/after 示例。](../task4/assets/alignment/04_ibug_image_103_before_after.jpg)

            ---PAGEBREAK---

            ### 3.3 Task5 ArcFace 识别训练与 LFW 验证

            Task5 展示层已同步为云端 800k dense 结果。训练曲线改为三个分面：训练 loss、训练 top-1、LFW accuracy，避免三条曲线挤在同一坐标系中。最佳 LFW 出现在 epoch `{best_epoch}`，accuracy 为 `{pct(best_acc)}`；最终独立评估 summary 中 accuracy 为 `{pct(eval_metrics.get('accuracy'))}`，ROC AUC 为 `{num(eval_metrics.get('roc_auc'))}`。

            LFW accuracy 从 epoch 1 就在 0.75-0.80 附近，是因为 epoch 1 已经完整看过 800k 张训练图，LFW 又是 aligned 1:1 verification protocol，并且每折会在训练折上选择阈值；这会让早期 embedding 已能在相对容易的 LFW 上得到中等准确率。后续 accuracy 波动和停滞，说明第一版自实现模型逐渐把 closed-set 分类身份拟合得更好，但 open-set embedding 泛化没有继续提升，所以不是单纯“epoch 太少”或“学习率太低”能解释。

            ![Task5 云端训练曲线，分面展示 train loss、train top-1 与 LFW accuracy。](../task5/assets/training/ms1mv3_dense_loss_acc_curve.png)

            ![Task5 LFW ROC 曲线。](../task5/assets/evaluation/lfw_roc_curve.png)

            ![Task5 LFW 相似度分布。](../task5/assets/evaluation/lfw_similarity_histogram.png)

            ### 3.4 Task6 模型压缩与 ONNX 推理

            | 模型 | LFW accuracy | latency ms/image | model size MB |
            |---|---:|---:|---:|
            | FP32 | {pct(fp32_metrics.get('accuracy'))} | {num(fp32_metrics.get('embedding_speed', {}).get('latency_ms_per_image'), 3)} | {num(quant.get('fp32', {}).get('model_size_mb'), 2)} |
            | Dynamic INT8 | {pct(q_metrics.get('accuracy'))} | {num(q_metrics.get('embedding_speed', {}).get('latency_ms_per_image'), 3)} | {num(quant.get('dynamic_quantized', {}).get('model_size_mb'), 2)} |
            | ONNX Runtime | {pct(onnx_metrics.get('accuracy'))} | {num(onnx_metrics.get('embedding_speed', {}).get('latency_ms_per_image'), 3)} | {num(onnx.get('onnx_size_mb'), 2)} |

            ![Task6 动态量化对比图。](../task6/assets/evaluation/task6_quantization_comparison.png)

            ![Task6 ONNX 推理对比图。](../task6/assets/evaluation/task6_onnx_comparison.png)

            ---PAGEBREAK---

            ## 4. 关键代码段与解释

            ### 4.1 Task3 WIDER FACE 数据与检测链路

            文件：`code/prepare/stage2_task3_2_prepare_widerface.py`、`code/evaluate/stage2_task3_3_evaluate_widerface.py`

            ```python
            writer.writerow(["image_path", "x1", "y1", "w", "h", "blur", "expression", "illumination", "invalid", "occlusion", "pose"])
            detections = nms_detections(detections, iou_thr=args.iou_thr)
            draw_detections(image, detections[: args.vis_top_k], out_path)
            ```

            解释：Task3 将 WIDER FACE 标注转换为训练可读的结构化索引，评估阶段再按 score threshold 和 IoU NMS 得到检测框，并输出固定数量的可视化样例，保证训练、评估和报告图可以追溯。

            ### 4.2 Task4 关键点驱动的人脸对齐

            文件：`code/task4/stage2_task4_3_align_faces.py`

            ```python
            src = np.float32([left_eye, right_eye, nose_tip, left_mouth, right_mouth])
            dst = np.float32(ARCFACE_TEMPLATE_112)
            matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
            aligned = cv2.warpAffine(image, matrix, (112, 112), flags=cv2.INTER_LINEAR)
            ```

            解释：Task4 不只画关键点，还把左眼、右眼、鼻尖和嘴角映射到 ArcFace 112x112 模板。这样输出的 aligned face 可以直接服务后续人脸识别模型输入。

            ### 4.3 Task5 ArcFace margin 与 LFW 10-fold 验证

            文件：`code/task5/stage2_task5_run_arcface.py`

            ```python
            cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
            theta = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            target_logits = torch.cos(theta + self.margin)
            logits = cosine.scatter(1, labels.view(-1, 1), target_logits.gather(1, labels.view(-1, 1))) * self.scale
            ```

            解释：ArcFace 在目标类别角度上加入 margin，使同一身份的 embedding 更紧、不同身份之间角度间隔更大。LFW 评估则使用 6000 pairs 的 10-fold protocol，每一折只在训练折选阈值，再在测试折计算 accuracy。

            ### 4.4 Task6 动态量化与 ONNX 导出

            文件：`code/task6/stage2_task6_2_quantize_arcface.py`、`code/task6/stage2_task6_3_export_onnx.py`

            ```python
            quantized = torch.quantization.quantize_dynamic(
                fp32_backbone.cpu(), {{torch.nn.Linear}}, dtype=torch.qint8
            )
            torch.onnx.export(
                backbone, dummy, onnx_out,
                input_names=["input"], output_names=["embedding"],
                dynamic_axes={{"input": {{0: "batch"}}, "embedding": {{0: "batch"}}}},
            )
            ```

            解释：动态量化只压缩 `Linear` 层，因此对 IResNet50 这种卷积为主的模型提升有限；ONNX 导出保留动态 batch 维度，并用同一 LFW protocol 验证 embedding 数值一致性和推理指标。

            ## 5. 下周待办

            - 如果继续追求 Task5 的 98.5% LFW 目标，优先采用官方 InsightFace full MS1MV3 RecordIO 训练路线，而不是继续扩大第一版自实现训练。
            - 在 Task6 基础上尝试静态量化、结构化剪枝或 TensorRT/ONNX Runtime GPU provider，预期比仅动态量化 `Linear` 层更有部署收益。
            - 整理最终提交目录，确认 `data/`、`work_dirs/`、`.pth`、`.onnx`、云端 tar 包继续被 Git 忽略，只提交代码、报告、summary、PDF 和小体积图表。
            """
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task5-train-summary", type=Path, default=Path("reports/task5/summaries/ms1mv3_dense_train_summary.json"))
    parser.add_argument("--task5-eval-summary", type=Path, default=Path("reports/task5/summaries/lfw_eval_summary.json"))
    parser.add_argument("--source-summary", type=Path, default=Path("reports/task6/summaries/source_model_summary.json"))
    parser.add_argument("--quant-summary", type=Path, default=Path("reports/task6/summaries/quantization_summary.json"))
    parser.add_argument("--onnx-summary", type=Path, default=Path("reports/task6/summaries/onnx_summary.json"))
    parser.add_argument("--survey-out", type=Path, default=Path("reports/task6/task6_1_optimization_methods.md"))
    parser.add_argument("--task6-report-out", type=Path, default=Path("reports/task6/stage2_task6_model_optimization_report.md"))
    parser.add_argument("--weekly-out", type=Path, default=Path("reports/weekly/week2_report_2026-05-28.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task5_train = read_json_optional(args.task5_train_summary)
    task5_eval = read_json_optional(args.task5_eval_summary)
    source = read_json_optional(args.source_summary)
    quant = read_json_optional(args.quant_summary)
    onnx = read_json_optional(args.onnx_summary)
    write_optimization_survey(args.survey_out)
    write_task6_report(args.task6_report_out, source, quant, onnx)
    write_weekly_report(args.weekly_out, task5_train, task5_eval, source, quant, onnx)
    print(f"Wrote {args.survey_out}")
    print(f"Wrote {args.task6_report_out}")
    print(f"Wrote {args.weekly_out}")


if __name__ == "__main__":
    main()
