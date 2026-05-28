#!/usr/bin/env python3
"""Dynamically quantize the Task5 first-version ArcFace model and benchmark LFW."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import torch

from stage2_task6_utils import (
    evaluate_torch_lfw,
    file_size_mb,
    load_first_version_backbone,
    plot_task6_comparison,
    set_torch_threads,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path("work_dirs/task6/source_arcface_8167/best.pth"))
    parser.add_argument("--lfw-dir", type=Path, default=Path("data/task5_lfw"))
    parser.add_argument("--work-dir", type=Path, default=Path("work_dirs/task6/quantization"))
    parser.add_argument("--summary-out", type=Path, default=Path("reports/task6/summaries/quantization_summary.json"))
    parser.add_argument("--comparison-plot-out", type=Path, default=Path("reports/task6/assets/evaluation/task6_quantization_comparison.png"))
    parser.add_argument("--fp32-roc-out", type=Path, default=Path("reports/task6/assets/evaluation/task6_fp32_lfw_roc.png"))
    parser.add_argument("--quantized-roc-out", type=Path, default=Path("reports/task6/assets/evaluation/task6_dynamic_quant_lfw_roc.png"))
    parser.add_argument("--device", default="cpu", help="Dynamic quantization is CPU-only; keep this as cpu for the comparable benchmark.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional smoke limit; omit for full 6000-pair LFW.")
    parser.add_argument("--quantized-engine", default="fbgemm", choices=["fbgemm", "qnnpack", "x86"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_torch_threads(args.threads)
    device = torch.device(args.device)
    if device.type != "cpu":
        raise ValueError("PyTorch dynamic quantized Linear modules are CPU inference modules; use --device cpu.")
    if args.quantized_engine in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = args.quantized_engine

    args.work_dir.mkdir(parents=True, exist_ok=True)
    fp32_backbone, cfg, checkpoint = load_first_version_backbone(args.config, args.checkpoint, device)
    fp32_backbone.eval()

    fp32_state_path = args.work_dir / "arcface_fp32_backbone_state_dict.pth"
    torch.save(fp32_backbone.state_dict(), fp32_state_path)

    fp32_metrics = evaluate_torch_lfw(
        fp32_backbone,
        cfg,
        args.lfw_dir,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_pairs=args.max_pairs,
        roc_plot_out=args.fp32_roc_out,
        hist_plot_out=args.fp32_roc_out.with_name("task6_fp32_similarity_histogram.png"),
        title_prefix="FP32 ArcFace",
    )

    quantized = torch.quantization.quantize_dynamic(fp32_backbone.cpu(), {torch.nn.Linear}, dtype=torch.qint8)
    quantized.eval()
    quantized_path = args.work_dir / "arcface_dynamic_quantized.pt"
    torch.save(quantized, quantized_path)

    quantized_metrics = evaluate_torch_lfw(
        quantized,
        cfg,
        args.lfw_dir,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_pairs=args.max_pairs,
        roc_plot_out=args.quantized_roc_out,
        hist_plot_out=args.quantized_roc_out.with_name("task6_dynamic_quant_similarity_histogram.png"),
        title_prefix="Dynamic Quantized ArcFace",
    )

    rows = [
        {
            "name": "FP32",
            "accuracy": fp32_metrics["accuracy"],
            "latency_ms_per_image": fp32_metrics["embedding_speed"]["latency_ms_per_image"],
            "model_size_mb": file_size_mb(fp32_state_path),
        },
        {
            "name": "Dynamic INT8",
            "accuracy": quantized_metrics["accuracy"],
            "latency_ms_per_image": quantized_metrics["embedding_speed"]["latency_ms_per_image"],
            "model_size_mb": file_size_mb(quantized_path),
        },
    ]
    plot_task6_comparison(rows, args.comparison_plot_out, "Task6 Dynamic Quantization Comparison")

    summary: dict[str, Any] = {
        "task": "Stage2 Task 6.2 dynamic quantization",
        "config": str(args.config),
        "source_checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_lfw_accuracy": checkpoint.get("best_lfw_accuracy"),
        "lfw_dir": str(args.lfw_dir),
        "full_lfw_protocol": args.max_pairs is None,
        "max_pairs": args.max_pairs,
        "artifacts": {
            "fp32_backbone_state_dict": str(fp32_state_path),
            "dynamic_quantized_model": str(quantized_path),
            "comparison_plot": str(args.comparison_plot_out),
            "fp32_roc_plot": str(args.fp32_roc_out),
            "quantized_roc_plot": str(args.quantized_roc_out),
        },
        "fp32": {
            "model_size_mb": file_size_mb(fp32_state_path),
            "metrics": fp32_metrics,
        },
        "dynamic_quantized": {
            "model_size_mb": file_size_mb(quantized_path),
            "metrics": quantized_metrics,
        },
        "comparison": {
            "accuracy_delta": float(quantized_metrics["accuracy"] - fp32_metrics["accuracy"]),
            "latency_speedup": float(
                fp32_metrics["embedding_speed"]["latency_ms_per_image"]
                / max(1e-12, quantized_metrics["embedding_speed"]["latency_ms_per_image"])
            ),
            "size_ratio": float(file_size_mb(quantized_path) / max(1e-12, file_size_mb(fp32_state_path))),
        },
        "note": "Dynamic quantization only affects Linear layers, so the convolution-heavy IResNet50 backbone may show modest gains.",
    }
    write_json(args.summary_out, summary)


if __name__ == "__main__":
    main()
