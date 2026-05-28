#!/usr/bin/env python3
"""Export the Task5 first-version ArcFace backbone to ONNX and benchmark LFW."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from stage2_task6_utils import (
    compute_lfw_metrics,
    evaluate_torch_lfw,
    file_size_mb,
    load_first_version_backbone,
    make_lfw_loader,
    plot_task6_comparison,
    read_lfw_pairs,
    set_torch_threads,
    unique_lfw_paths,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path("work_dirs/task6/source_arcface_8167/best.pth"))
    parser.add_argument("--lfw-dir", type=Path, default=Path("data/task5_lfw"))
    parser.add_argument("--onnx-out", type=Path, default=Path("work_dirs/task6/onnx/arcface_iresnet50_8167.onnx"))
    parser.add_argument("--summary-out", type=Path, default=Path("reports/task6/summaries/onnx_summary.json"))
    parser.add_argument("--comparison-plot-out", type=Path, default=Path("reports/task6/assets/evaluation/task6_onnx_comparison.png"))
    parser.add_argument("--onnx-roc-out", type=Path, default=Path("reports/task6/assets/evaluation/task6_onnx_lfw_roc.png"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify-count", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional smoke limit; omit for full 6000-pair LFW.")
    return parser.parse_args()


def select_providers(provider: str) -> list[str]:
    available = ort.get_available_providers()
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider is not available. Available providers: {available}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def export_onnx(backbone: torch.nn.Module, onnx_out: Path, image_size: int, device: torch.device, opset: int) -> None:
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    backbone.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        backbone,
        dummy,
        onnx_out,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )


def run_onnx_embeddings(
    session: ort.InferenceSession,
    paths: list[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import time

    loader = make_lfw_loader(paths, image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    embeddings: dict[str, np.ndarray] = {}
    start = time.perf_counter()
    for images, batch_paths in loader:
        feats = session.run([output_name], {input_name: images.numpy().astype(np.float32)})[0].astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-12)
        feats = feats / norms
        for path, feat in zip(batch_paths, feats):
            embeddings[str(path)] = feat
    elapsed = time.perf_counter() - start
    stats = {
        "images": len(paths),
        "elapsed_seconds": elapsed,
        "latency_ms_per_image": (elapsed / max(1, len(paths))) * 1000.0,
        "throughput_images_per_second": len(paths) / elapsed if elapsed > 0 else 0.0,
        "batch_size": int(batch_size),
        "providers": session.get_providers(),
    }
    return embeddings, stats


@torch.no_grad()
def compare_pytorch_onnx(
    backbone: torch.nn.Module,
    session: ort.InferenceSession,
    paths: list[str],
    image_size: int,
    device: torch.device,
    count: int,
) -> dict[str, float]:
    sample_paths = paths[: max(1, min(count, len(paths)))]
    loader = make_lfw_loader(sample_paths, image_size=image_size, batch_size=len(sample_paths), num_workers=0)
    images, _ = next(iter(loader))
    backbone.eval()
    torch_feats = backbone(images.to(device)).detach().cpu().numpy().astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_feats = session.run([output_name], {input_name: images.numpy().astype(np.float32)})[0].astype(np.float32)
    torch_feats = torch_feats / np.linalg.norm(torch_feats, axis=1, keepdims=True).clip(min=1e-12)
    onnx_feats = onnx_feats / np.linalg.norm(onnx_feats, axis=1, keepdims=True).clip(min=1e-12)
    abs_diff = np.abs(torch_feats - onnx_feats)
    cosine = np.sum(torch_feats * onnx_feats, axis=1)
    return {
        "samples": float(len(sample_paths)),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "mean_cosine": float(cosine.mean()),
        "min_cosine": float(cosine.min()),
    }


def main() -> None:
    args = parse_args()
    set_torch_threads(args.threads)
    requested_device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    backbone, cfg, checkpoint = load_first_version_backbone(args.config, args.checkpoint, requested_device)
    image_size = int(cfg.data.image_size)

    export_onnx(backbone, args.onnx_out, image_size=image_size, device=requested_device, opset=args.opset)
    providers = select_providers(args.provider)
    session = ort.InferenceSession(str(args.onnx_out), providers=providers)

    pairs = read_lfw_pairs(args.lfw_dir, max_pairs=args.max_pairs)
    paths = unique_lfw_paths(pairs)
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"LFW has {len(missing)} missing images; first missing path: {missing[0]}")

    torch_metrics = evaluate_torch_lfw(
        backbone,
        cfg,
        args.lfw_dir,
        requested_device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_pairs=args.max_pairs,
        roc_plot_out=args.onnx_roc_out.with_name("task6_pytorch_reference_lfw_roc.png"),
        hist_plot_out=None,
        title_prefix="PyTorch Reference ArcFace",
    )
    onnx_embeddings, onnx_speed = run_onnx_embeddings(
        session,
        paths,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    onnx_metrics = compute_lfw_metrics(
        pairs,
        onnx_embeddings,
        roc_plot_out=args.onnx_roc_out,
        hist_plot_out=args.onnx_roc_out.with_name("task6_onnx_similarity_histogram.png"),
        plot_title_prefix="ONNX Runtime ArcFace",
    )
    onnx_metrics["embedding_speed"] = onnx_speed
    onnx_metrics["full_protocol"] = args.max_pairs is None

    consistency = compare_pytorch_onnx(
        backbone,
        session,
        paths,
        image_size=image_size,
        device=requested_device,
        count=args.verify_count,
    )
    rows = [
        {
            "name": "PyTorch",
            "accuracy": torch_metrics["accuracy"],
            "latency_ms_per_image": torch_metrics["embedding_speed"]["latency_ms_per_image"],
            "model_size_mb": file_size_mb(args.checkpoint),
        },
        {
            "name": "ONNX",
            "accuracy": onnx_metrics["accuracy"],
            "latency_ms_per_image": onnx_metrics["embedding_speed"]["latency_ms_per_image"],
            "model_size_mb": file_size_mb(args.onnx_out),
        },
    ]
    plot_task6_comparison(rows, args.comparison_plot_out, "Task6 ONNX Runtime Comparison")

    summary: dict[str, Any] = {
        "task": "Stage2 Task 6.3 ONNX export and inference",
        "config": str(args.config),
        "source_checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_lfw_accuracy": checkpoint.get("best_lfw_accuracy"),
        "onnx_model": str(args.onnx_out),
        "onnx_size_mb": file_size_mb(args.onnx_out),
        "opset": args.opset,
        "providers": session.get_providers(),
        "full_lfw_protocol": args.max_pairs is None,
        "max_pairs": args.max_pairs,
        "pytorch_reference": torch_metrics,
        "onnx": onnx_metrics,
        "consistency": consistency,
        "comparison": {
            "accuracy_delta": float(onnx_metrics["accuracy"] - torch_metrics["accuracy"]),
            "latency_speedup": float(
                torch_metrics["embedding_speed"]["latency_ms_per_image"]
                / max(1e-12, onnx_metrics["embedding_speed"]["latency_ms_per_image"])
            ),
            "size_ratio_to_checkpoint": float(file_size_mb(args.onnx_out) / max(1e-12, file_size_mb(args.checkpoint))),
        },
        "artifacts": {
            "comparison_plot": str(args.comparison_plot_out),
            "onnx_roc_plot": str(args.onnx_roc_out),
        },
    }
    write_json(args.summary_out, summary)


if __name__ == "__main__":
    main()
