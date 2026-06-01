#!/usr/bin/env python3
"""Diagnose why dynamic quantization did not reduce ArcFace latency."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from stage2_task6_utils import (
    file_size_mb,
    load_first_version_backbone,
    set_torch_threads,
    write_json,
)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def module_param_count(module: torch.nn.Module, module_type: type[torch.nn.Module]) -> int:
    total = 0
    for child in module.modules():
        if isinstance(child, module_type):
            total += sum(param.numel() for param in child.parameters(recurse=False))
    return total


def module_type_counts(module: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for child in module.modules():
        name = child.__class__.__name__
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def static_model_profile(model: torch.nn.Module) -> dict[str, Any]:
    total_params = sum(param.numel() for param in model.parameters())
    conv_params = module_param_count(model, torch.nn.Conv2d)
    linear_params = module_param_count(model, torch.nn.Linear)
    bn_params = module_param_count(model, torch.nn.BatchNorm2d)
    return {
        "total_params": int(total_params),
        "conv2d_params": int(conv_params),
        "linear_params": int(linear_params),
        "batchnorm2d_params": int(bn_params),
        "conv2d_param_ratio": conv_params / max(total_params, 1),
        "linear_param_ratio": linear_params / max(total_params, 1),
        "batchnorm2d_param_ratio": bn_params / max(total_params, 1),
        "module_counts": module_type_counts(model),
    }


@torch.no_grad()
def benchmark_torch(
    model: torch.nn.Module,
    image_size: int,
    batch_size: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    model.to(device)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    images = batch_size * iters
    return {
        "batch_size": float(batch_size),
        "elapsed_seconds": elapsed,
        "latency_ms_per_batch": elapsed / max(iters, 1) * 1000.0,
        "latency_ms_per_image": elapsed / max(images, 1) * 1000.0,
        "throughput_images_per_second": images / elapsed if elapsed > 0 else 0.0,
    }


def benchmark_onnx(
    onnx_path: Path,
    image_size: int,
    batch_size: int,
    warmup: int,
    iters: int,
    provider: str,
) -> dict[str, Any] | None:
    if not onnx_path.exists():
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    providers = ["CPUExecutionProvider"]
    if provider == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    x = np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32)
    for _ in range(warmup):
        session.run([output_name], {input_name: x})
    start = time.perf_counter()
    for _ in range(iters):
        session.run([output_name], {input_name: x})
    elapsed = time.perf_counter() - start
    images = batch_size * iters
    return {
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "latency_ms_per_batch": elapsed / max(iters, 1) * 1000.0,
        "latency_ms_per_image": elapsed / max(images, 1) * 1000.0,
        "throughput_images_per_second": images / elapsed if elapsed > 0 else 0.0,
        "providers": session.get_providers(),
    }


def plot_latency(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for backend in ("fp32", "dynamic_quantized", "onnx"):
        for row in summary.get("microbench", {}).get(backend, []):
            rows.append((backend, int(row["batch_size"]), float(row["latency_ms_per_image"])))
    if not rows:
        return
    batch_sizes = sorted({batch for _, batch, _ in rows})
    labels = ["fp32", "dynamic_quantized", "onnx"]
    x = np.arange(len(batch_sizes))
    width = 0.25
    plt.figure(figsize=(8.4, 4.6))
    for idx, label in enumerate(labels):
        values = []
        for batch in batch_sizes:
            match = [value for backend, b, value in rows if backend == label and b == batch]
            values.append(match[0] if match else 0.0)
        plt.bar(x + (idx - 1) * width, values, width=width, label=label)
    plt.xticks(x, [str(item) for item in batch_sizes])
    plt.xlabel("batch size")
    plt.ylabel("model-only latency ms/image")
    plt.title("Task6 Latency Diagnosis")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path("work_dirs/task6/source_arcface_8167/best.pth"))
    parser.add_argument("--dynamic-model", type=Path, default=Path("work_dirs/task6/quantization/arcface_dynamic_quantized.pt"))
    parser.add_argument("--onnx-model", type=Path, default=Path("work_dirs/task6/onnx/arcface_iresnet50_8167.onnx"))
    parser.add_argument("--summary-out", type=Path, default=Path("reports/task6/summaries/latency_diagnosis_summary.json"))
    parser.add_argument("--plot-out", type=Path, default=Path("reports/task6/assets/evaluation/task6_latency_diagnosis.png"))
    parser.add_argument("--batch-sizes", default="1,16,64")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--onnx-provider", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_torch_threads(args.threads)
    device = torch.device("cpu")
    fp32_backbone, cfg, checkpoint = load_first_version_backbone(args.config, args.checkpoint, device)
    image_size = int(cfg.data.image_size)
    quantized = torch.quantization.quantize_dynamic(fp32_backbone.cpu(), {torch.nn.Linear}, dtype=torch.qint8)
    batch_sizes = parse_int_list(args.batch_sizes)

    microbench = {"fp32": [], "dynamic_quantized": [], "onnx": []}
    for batch_size in batch_sizes:
        microbench["fp32"].append(benchmark_torch(fp32_backbone, image_size, batch_size, args.warmup, args.iters, device))
        microbench["dynamic_quantized"].append(
            benchmark_torch(quantized, image_size, batch_size, args.warmup, args.iters, device)
        )
        onnx_row = benchmark_onnx(args.onnx_model, image_size, batch_size, args.warmup, args.iters, args.onnx_provider)
        if onnx_row is not None:
            microbench["onnx"].append(onnx_row)

    fp32_profile = static_model_profile(fp32_backbone)
    summary: dict[str, Any] = {
        "task": "Stage2 Task6 latency diagnosis",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_lfw_accuracy": checkpoint.get("best_lfw_accuracy"),
        "image_size": image_size,
        "threads": args.threads,
        "profile": fp32_profile,
        "quantization_scope": {
            "method": "torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)",
            "quantized_layer_types": ["Linear"],
            "linear_param_ratio": fp32_profile["linear_param_ratio"],
            "conv2d_param_ratio": fp32_profile["conv2d_param_ratio"],
            "interpretation": "Dynamic quantization leaves Conv2d layers in FP32, so most IResNet50 compute is unchanged.",
        },
        "file_sizes_mb": {
            "source_checkpoint": file_size_mb(args.checkpoint),
            "dynamic_quantized_model": file_size_mb(args.dynamic_model),
            "onnx_model": file_size_mb(args.onnx_model),
        },
        "microbench": microbench,
        "conclusion": {
            "dynamic_quant_latency_expected": "small_or_none",
            "primary_reason": "The ArcFace IResNet50 backbone is convolution-heavy; dynamic quantization only affects Linear layers.",
            "secondary_reasons": [
                "The LFW benchmark includes image decoding, resizing, normalization, and dataloader overhead.",
                "Quantized Linear layers can add quant/dequant overhead while reducing only a small part of total compute.",
                "No Conv2d INT8 kernel path is used by PyTorch dynamic quantization.",
                "ONNX Runtime is faster because it uses graph-level/runtime optimizations, not because the model was compressed.",
            ],
        },
    }
    plot_latency(summary, args.plot_out)
    summary["plot"] = str(args.plot_out) if args.plot_out.exists() else ""
    write_json(args.summary_out, summary)


if __name__ == "__main__":
    main()
