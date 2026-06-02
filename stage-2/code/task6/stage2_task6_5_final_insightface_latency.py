#!/usr/bin/env python3
"""Benchmark the final InsightFace R50 checkpoint for Task6 deployment latency."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import pickle
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np


STAGE2_ROOT = Path(__file__).resolve().parents[2]
TASK5_INSIGHTFACE_MODULE = "stage2_task5_official_insightface"


def sanitize_thread_env(default: str = "4") -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        value = os.environ.get(key)
        if value is None:
            continue
        stripped = value.strip()
        if not stripped.isdigit() or int(stripped) <= 0:
            os.environ[key] = default


sanitize_thread_env()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn.preprocessing
import torch
from PIL import Image
from sklearn.metrics import roc_curve

from stage2_task6_utils import file_size_mb, jsonable, set_torch_threads, write_json


def load_config(path: Path) -> Any:
    from mmengine.config import Config

    return Config.fromfile(path)


def load_task5_insightface_module() -> Any:
    if TASK5_INSIGHTFACE_MODULE in sys.modules:
        return sys.modules[TASK5_INSIGHTFACE_MODULE]
    module_path = STAGE2_ROOT / "code" / "task5" / "stage2_task5_5_run_insightface.py"
    spec = importlib.util.spec_from_file_location(TASK5_INSIGHTFACE_MODULE, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Task5 InsightFace wrapper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[TASK5_INSIGHTFACE_MODULE] = module
    spec.loader.exec_module(module)
    return module


def prepare_insightface(cfg: Any, insightface_ref: str | None) -> dict[str, Any]:
    task5 = load_task5_insightface_module()
    repo = task5.ensure_repo(cfg.insightface.repo_url, insightface_ref or cfg.insightface.ref, Path(cfg.insightface.external_dir))
    arcface_root = task5.arcface_dir(cfg).resolve()
    patch = task5.patch_verification_interp(arcface_root)
    if str(arcface_root) not in sys.path:
        sys.path.insert(0, str(arcface_root))
    return {"repo": repo, "verification_patch": patch, "arcface_root": str(arcface_root)}


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def load_backbone(cfg: Any, checkpoint: Path, device: torch.device, precision: str = "fp32") -> torch.nn.Module:
    from backbones import get_model  # type: ignore

    model = get_model(
        cfg.official.network,
        dropout=0.0,
        fp16=False,
        num_features=int(cfg.official.embedding_size),
    )
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict_backbone" in state:
        state = state["state_dict_backbone"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload in {checkpoint}")
    model.load_state_dict(strip_module_prefix(state), strict=True)
    model.eval()
    model.to(device)
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 PyTorch/ONNX export is only supported on CUDA for this script.")
        model.half()
    return model


def load_lfw_bin(path: Path) -> tuple[list[bytes], list[bool]]:
    with path.open("rb") as handle:
        try:
            bins, issame = pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            bins, issame = pickle.load(handle, encoding="bytes")
    return list(bins), [bool(item) for item in issame]


def image_bytes_to_array(raw: bytes, image_size: tuple[int, int], flip: bool = False, dtype: np.dtype = np.float32) -> np.ndarray:
    with Image.open(BytesIO(raw)) as image:
        image = image.convert("RGB")
        if image.size != image_size:
            image = image.resize(image_size, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    if flip:
        array = array[:, ::-1, :].copy()
    array = np.transpose(array, (2, 0, 1))
    array = ((array / 255.0) - 0.5) / 0.5
    return array.astype(dtype, copy=False)


def _accuracy_at_distance_threshold(threshold: float, distances: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    predict_issame = distances < threshold
    tp = np.sum(np.logical_and(predict_issame, labels))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(labels)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(labels)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), labels))
    tpr = 0.0 if (tp + fn) == 0 else float(tp) / float(tp + fn)
    fpr = 0.0 if (fp + tn) == 0 else float(fp) / float(fp + tn)
    acc = float(tp + tn) / max(1, distances.size)
    return tpr, fpr, acc


def _val_far_at_distance_threshold(threshold: float, distances: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    predict_issame = distances < threshold
    true_accept = np.sum(np.logical_and(predict_issame, labels))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(labels)))
    n_same = np.sum(labels)
    n_diff = np.sum(np.logical_not(labels))
    val = 0.0 if n_same == 0 else float(true_accept) / float(n_same)
    far = 0.0 if n_diff == 0 else float(false_accept) / float(n_diff)
    return val, far


def evaluate_embeddings(embeddings: np.ndarray, issame: list[bool]) -> dict[str, Any]:
    labels = np.array(issame, dtype=bool)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    distances = np.sum(np.square(embeddings1 - embeddings2), axis=1)
    thresholds = np.arange(0.0, 4.0, 0.01)
    folds = np.array_split(np.arange(len(labels)), 10)
    accuracies: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_indices in folds:
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[fold_indices] = False
        train_distances = distances[train_mask]
        train_labels = labels[train_mask]
        train_acc = [
            _accuracy_at_distance_threshold(threshold, train_distances, train_labels)[2]
            for threshold in thresholds
        ]
        best_threshold = float(thresholds[int(np.argmax(train_acc))])
        _, _, test_acc = _accuracy_at_distance_threshold(best_threshold, distances[fold_indices], labels[fold_indices])
        accuracies.append(float(test_acc))
        fold_rows.append(
            {
                "best_threshold": best_threshold,
                "train_accuracy": float(np.max(train_acc)),
                "test_accuracy": float(test_acc),
            }
        )

    val_thresholds = np.arange(0.0, 4.0, 0.001)
    far_target = 0.001
    vals: list[float] = []
    fars: list[float] = []
    for fold_indices in folds:
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[fold_indices] = False
        far_train = np.array(
            [
                _val_far_at_distance_threshold(threshold, distances[train_mask], labels[train_mask])[1]
                for threshold in val_thresholds
            ]
        )
        if np.max(far_train) >= far_target:
            unique_far, unique_indices = np.unique(far_train, return_index=True)
            unique_thresholds = val_thresholds[unique_indices]
            order = np.argsort(unique_far)
            unique_far = unique_far[order]
            unique_thresholds = unique_thresholds[order]
            if unique_far.size < 2:
                threshold = float(unique_thresholds[0])
            else:
                threshold = float(np.interp(far_target, unique_far, unique_thresholds))
        else:
            threshold = 0.0
        fold_val, fold_far = _val_far_at_distance_threshold(threshold, distances[fold_indices], labels[fold_indices])
        vals.append(float(fold_val))
        fars.append(float(fold_far))

    fpr, tpr, _ = roc_curve(labels.astype(int), -distances)
    return {
        "pairs": len(issame),
        "images": int(embeddings.shape[0]),
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "fold_accuracies": accuracies,
        "folds": fold_rows,
        "val_at_far_1e-3": float(np.mean(vals)),
        "val_std": float(np.std(vals)),
        "far": float(np.mean(fars)),
        "target_far": far_target,
        "roc_auc": float(np.trapz(tpr, fpr)),
        "protocol": "InsightFace-compatible 6000-pair 10-fold squared Euclidean threshold search",
    }


@torch.no_grad()
def run_lfw_torch(
    model: torch.nn.Module,
    bins: list[bytes],
    issame: list[bool],
    image_size: tuple[int, int],
    batch_size: int,
    device: torch.device,
    precision: str,
    label: str,
) -> dict[str, Any]:
    dtype = torch.float16 if precision == "fp16" else torch.float32
    embeddings_by_flip: list[np.ndarray] = []
    print(f"[LFW][{label}] start images={len(bins)} batch={batch_size} precision={precision}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for flip in (False, True):
        print(f"[LFW][{label}] flip={flip}", flush=True)
        chunks: list[np.ndarray] = []
        for start in range(0, len(bins), batch_size):
            batch = bins[start : start + batch_size]
            arrays = [image_bytes_to_array(raw, image_size, flip=flip, dtype=np.float32) for raw in batch]
            tensor = torch.from_numpy(np.stack(arrays)).to(device=device, dtype=dtype)
            output = model(tensor).detach().float().cpu().numpy()
            chunks.append(output)
        embeddings_by_flip.append(np.concatenate(chunks, axis=0))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[LFW][{label}] finished in {elapsed:.2f}s", flush=True)
    xnorm = float(np.mean([np.linalg.norm(row) for embeddings in embeddings_by_flip for row in embeddings]))
    combined = embeddings_by_flip[0] + embeddings_by_flip[1]
    metrics = evaluate_embeddings(sklearn.preprocessing.normalize(combined), issame)
    metrics["xnorm"] = xnorm
    metrics["embedding_speed"] = {
        "backend": label,
        "images": len(bins),
        "effective_forward_images": len(bins) * 2,
        "elapsed_seconds": elapsed,
        "latency_ms_per_image": elapsed / max(1, len(bins)) * 1000.0,
        "latency_ms_per_forward_image": elapsed / max(1, len(bins) * 2) * 1000.0,
        "throughput_images_per_second": len(bins) / elapsed if elapsed > 0 else 0.0,
        "batch_size": int(batch_size),
        "device": str(device),
        "precision": precision,
    }
    return metrics


def export_onnx(
    model: torch.nn.Module,
    onnx_out: Path,
    image_size: tuple[int, int],
    device: torch.device,
    precision: str,
    opset: int,
) -> None:
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if precision == "fp16" else torch.float32
    dummy = torch.randn(1, 3, image_size[0], image_size[1], device=device, dtype=dtype)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_out,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
        )


def make_onnx_session(path: Path, provider: str):
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider is not available. Available providers: {available}")
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    # Avoid cuDNN exhaustive algorithm search failures on some
                    # laptop/WSL CUDA stacks while keeping CUDA execution.
                    "cudnn_conv_algo_search": "DEFAULT",
                },
            ),
            "CPUExecutionProvider",
        ]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    session = ort.InferenceSession(str(path), providers=providers)
    actual = session.get_providers()
    if provider == "cuda" and "CUDAExecutionProvider" not in actual:
        raise RuntimeError(
            "CUDAExecutionProvider was requested but the ONNX session did not use it. "
            f"Available providers: {available}; session providers: {actual}"
        )
    return session, available


def onnx_input_dtype(session: Any) -> np.dtype:
    input_type = session.get_inputs()[0].type
    if input_type == "tensor(float16)":
        return np.float16
    return np.float32


def run_lfw_onnx(
    session: Any,
    bins: list[bytes],
    issame: list[bool],
    image_size: tuple[int, int],
    batch_size: int,
    label: str,
) -> dict[str, Any]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dtype = onnx_input_dtype(session)
    embeddings_by_flip: list[np.ndarray] = []
    print(f"[LFW][{label}] start images={len(bins)} batch={batch_size}", flush=True)
    started = time.perf_counter()
    for flip in (False, True):
        print(f"[LFW][{label}] flip={flip}", flush=True)
        chunks: list[np.ndarray] = []
        for start in range(0, len(bins), batch_size):
            batch = bins[start : start + batch_size]
            inputs = np.stack([image_bytes_to_array(raw, image_size, flip=flip, dtype=dtype) for raw in batch])
            output = session.run([output_name], {input_name: inputs})[0].astype(np.float32)
            chunks.append(output)
        embeddings_by_flip.append(np.concatenate(chunks, axis=0))
    elapsed = time.perf_counter() - started
    print(f"[LFW][{label}] finished in {elapsed:.2f}s", flush=True)
    xnorm = float(np.mean([np.linalg.norm(row) for embeddings in embeddings_by_flip for row in embeddings]))
    combined = embeddings_by_flip[0] + embeddings_by_flip[1]
    metrics = evaluate_embeddings(sklearn.preprocessing.normalize(combined), issame)
    metrics["xnorm"] = xnorm
    metrics["embedding_speed"] = {
        "backend": label,
        "images": len(bins),
        "effective_forward_images": len(bins) * 2,
        "elapsed_seconds": elapsed,
        "latency_ms_per_image": elapsed / max(1, len(bins)) * 1000.0,
        "latency_ms_per_forward_image": elapsed / max(1, len(bins) * 2) * 1000.0,
        "throughput_images_per_second": len(bins) / elapsed if elapsed > 0 else 0.0,
        "batch_size": int(batch_size),
        "providers": session.get_providers(),
        "input_dtype": str(dtype),
    }
    return metrics


@torch.no_grad()
def benchmark_torch_forward(
    model: torch.nn.Module,
    image_size: tuple[int, int],
    batch_size: int,
    warmup: int,
    iters: int,
    device: torch.device,
    precision: str,
    label: str,
) -> dict[str, Any]:
    dtype = torch.float16 if precision == "fp16" else torch.float32
    x = torch.randn(batch_size, 3, image_size[0], image_size[1], device=device, dtype=dtype)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "backend": label,
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "iters": int(iters),
        "elapsed_seconds": elapsed,
        "latency_ms_per_batch": elapsed / max(1, iters) * 1000.0,
        "latency_ms_per_image": elapsed / max(1, iters * batch_size) * 1000.0,
        "throughput_images_per_second": (iters * batch_size) / elapsed if elapsed > 0 else 0.0,
        "device": str(device),
        "precision": precision,
    }


def benchmark_onnx_forward(
    session: Any,
    image_size: tuple[int, int],
    batch_size: int,
    warmup: int,
    iters: int,
    label: str,
) -> dict[str, Any]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dtype = onnx_input_dtype(session)
    x = np.random.randn(batch_size, 3, image_size[0], image_size[1]).astype(dtype)
    for _ in range(warmup):
        _ = session.run([output_name], {input_name: x})
    started = time.perf_counter()
    for _ in range(iters):
        _ = session.run([output_name], {input_name: x})
    elapsed = time.perf_counter() - started
    return {
        "backend": label,
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "iters": int(iters),
        "elapsed_seconds": elapsed,
        "latency_ms_per_batch": elapsed / max(1, iters) * 1000.0,
        "latency_ms_per_image": elapsed / max(1, iters * batch_size) * 1000.0,
        "throughput_images_per_second": (iters * batch_size) / elapsed if elapsed > 0 else 0.0,
        "providers": session.get_providers(),
        "input_dtype": str(dtype),
    }


@torch.no_grad()
def compare_onnx_to_torch(
    torch_model: torch.nn.Module,
    session: Any,
    image_size: tuple[int, int],
    device: torch.device,
    samples: int,
) -> dict[str, Any]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dtype = onnx_input_dtype(session)
    x = torch.randn(samples, 3, image_size[0], image_size[1], device=device, dtype=torch.float32)
    torch_out = torch_model(x).detach().float().cpu().numpy()
    onnx_in = x.detach().cpu().numpy().astype(dtype)
    onnx_out = session.run([output_name], {input_name: onnx_in})[0].astype(np.float32)
    torch_norm = torch_out / np.linalg.norm(torch_out, axis=1, keepdims=True).clip(min=1e-12)
    onnx_norm = onnx_out / np.linalg.norm(onnx_out, axis=1, keepdims=True).clip(min=1e-12)
    cosine = np.sum(torch_norm * onnx_norm, axis=1)
    abs_diff = np.abs(torch_norm - onnx_norm)
    return {
        "samples": int(samples),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "mean_cosine": float(np.mean(cosine)),
        "min_cosine": float(np.min(cosine)),
        "onnx_input_dtype": str(dtype),
    }


def plot_latency(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = summary.get("model_only_latency", [])
    if not rows:
        return
    labels = sorted({row["backend"] for row in rows})
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    x = np.arange(len(batch_sizes))
    width = 0.8 / max(1, len(labels))
    plt.figure(figsize=(11.5, 5.2))
    for idx, label in enumerate(labels):
        values = []
        for batch in batch_sizes:
            match = next((row for row in rows if row["backend"] == label and int(row["batch_size"]) == batch), None)
            values.append(float(match.get("latency_ms_per_image", 0.0)) if match else 0.0)
        plt.bar(x + idx * width, values, width=width, label=label)
    plt.xticks(x + width * (len(labels) - 1) / 2, [str(item) for item in batch_sizes])
    plt.ylabel("model-only latency ms/image")
    plt.xlabel("batch size")
    plt.title("Task6 Final InsightFace R50 Latency")
    plt.grid(axis="y", alpha=0.22)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lfw = summary.get("lfw_metrics", {})
    model_only = summary.get("model_only_latency", [])
    lines = [
        "# Task6 Final InsightFace R50 Latency Report",
        "",
        "## Source",
        "",
        f"- Checkpoint: `{summary.get('checkpoint')}`",
        f"- Source/cloud LFW accuracy: `{summary.get('source_lfw_accuracy')}`",
        f"- Local latency-bin LFW accuracy: `{summary.get('local_lfw_recheck_accuracy')}`",
        f"- Device: `{summary.get('device')}`",
        f"- ONNX Runtime providers available: `{summary.get('onnxruntime_available_providers')}`",
        f"- LFW note: {summary.get('lfw_evaluation_note', '')}",
        "",
        "## Full LFW End-to-End Benchmark",
        "",
        "| Backend | Accuracy | Latency ms/image | Throughput img/s | Size MB |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in lfw.items():
        speed = metrics.get("embedding_speed", {})
        size_mb = summary.get("artifact_sizes_mb", {}).get(name, 0.0)
        lines.append(
            f"| {name} | {metrics.get('accuracy', 0.0):.4f} | "
            f"{speed.get('latency_ms_per_image', 0.0):.3f} | "
            f"{speed.get('throughput_images_per_second', 0.0):.2f} | {size_mb:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Model-Only Latency",
            "",
            "| Backend | Batch | Latency ms/image | Throughput img/s |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in model_only:
        lines.append(
            f"| {row.get('backend')} | {row.get('batch_size')} | "
            f"{row.get('latency_ms_per_image', 0.0):.4f} | "
            f"{row.get('throughput_images_per_second', 0.0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            summary.get("conclusion", ""),
            "",
            "Dynamic quantization remains a CPU Linear-layer control and is not the main acceleration route for this convolution-heavy ArcFace R50 model.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


def parse_batch_sizes(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_providers(text: str) -> list[str]:
    providers = [item.strip().lower() for item in text.split(",") if item.strip()]
    valid = {"cuda", "cpu"}
    invalid = sorted(set(providers) - valid)
    if invalid:
        raise ValueError(f"Unsupported providers: {invalid}")
    return providers


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_torch_threads(args.threads)
    cfg = load_config(args.config)
    print("[Task6 final] preparing InsightFace runtime", flush=True)
    setup = prepare_insightface(cfg, args.insightface_ref)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing final Task5 checkpoint: {checkpoint}")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    image_size = tuple(int(item) for item in cfg.data.expected_image_size)
    lfw_bin = Path(args.lfw_bin) if args.lfw_bin else Path(cfg.data.rec) / "lfw.bin"
    if not lfw_bin.exists():
        raise FileNotFoundError(f"Missing InsightFace-format LFW bin: {lfw_bin}")

    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    requested_providers = parse_providers(args.providers)
    if "cuda" in requested_providers and "CUDAExecutionProvider" not in available_providers:
        diagnostic = {
            "task": "Stage2 Task6 final InsightFace latency",
            "status": "failed",
            "reason": "CUDAExecutionProvider is not available",
            "onnxruntime_available_providers": available_providers,
            "requested_providers": requested_providers,
            "checkpoint": str(checkpoint),
        }
        write_json(args.summary_out, diagnostic)
        raise RuntimeError(f"CUDAExecutionProvider is not available. Available providers: {available_providers}")

    bins, issame = load_lfw_bin(lfw_bin)
    print(f"[Task6 final] loaded LFW bin: images={len(bins)} pairs={len(issame)}", flush=True)
    if args.max_images:
        limit = max(2, int(args.max_images))
        bins = bins[:limit]
        issame = issame[: max(1, limit // 2)]

    print("[Task6 final] loading PyTorch backbones", flush=True)
    fp32_model = load_backbone(cfg, checkpoint, device=device, precision="fp32")
    fp16_model = load_backbone(cfg, checkpoint, device=device, precision="fp16") if device.type == "cuda" else None

    work_dir = args.work_dir
    fp32_onnx = work_dir / "insightface_r50_final_fp32.onnx"
    fp16_onnx = work_dir / "insightface_r50_final_fp16.onnx"
    print(f"[Task6 final] exporting ONNX FP32 -> {fp32_onnx}", flush=True)
    export_onnx(fp32_model, fp32_onnx, image_size, device=device, precision="fp32", opset=args.opset)
    if fp16_model is not None:
        print(f"[Task6 final] exporting ONNX FP16 -> {fp16_onnx}", flush=True)
        export_onnx(fp16_model, fp16_onnx, image_size, device=device, precision="fp16", opset=args.opset)

    onnx_sessions: dict[str, Any] = {}
    for provider in requested_providers:
        session, _ = make_onnx_session(fp32_onnx, provider)
        onnx_sessions[f"onnx_fp32_{provider}"] = session
    if fp16_onnx.exists() and "cuda" in requested_providers:
        session, _ = make_onnx_session(fp16_onnx, "cuda")
        onnx_sessions["onnx_fp16_cuda"] = session

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    model_only: list[dict[str, Any]] = []
    print(f"[Task6 final] model-only latency batches={batch_sizes}", flush=True)
    for batch_size in batch_sizes:
        model_only.append(
            benchmark_torch_forward(
                fp32_model,
                image_size,
                batch_size,
                args.warmup,
                args.iters,
                device=device,
                precision="fp32",
                label="pytorch_fp32_cuda" if device.type == "cuda" else "pytorch_fp32_cpu",
            )
        )
        if fp16_model is not None:
            model_only.append(
                benchmark_torch_forward(
                    fp16_model,
                    image_size,
                    batch_size,
                    args.warmup,
                    args.iters,
                    device=device,
                    precision="fp16",
                    label="pytorch_fp16_cuda",
                )
            )
        for name, session in onnx_sessions.items():
            if name.endswith("_cuda") and batch_size > args.onnx_cuda_max_batch:
                model_only.append(
                    {
                        "backend": name,
                        "batch_size": int(batch_size),
                        "skipped": True,
                        "reason": (
                            "Skipped by default on 8GB laptop GPUs to avoid cuDNN algorithm-search failures; "
                            f"set --onnx-cuda-max-batch >= {batch_size} to force it."
                        ),
                    }
                )
            else:
                model_only.append(benchmark_onnx_forward(session, image_size, batch_size, args.warmup, args.iters, name))

    dynamic_control = None
    if args.include_dynamic_control:
        print("[Task6 final] dynamic quantization CPU control", flush=True)
        cpu_model = load_backbone(cfg, checkpoint, device=torch.device("cpu"), precision="fp32")
        quantized = torch.quantization.quantize_dynamic(cpu_model, {torch.nn.Linear}, dtype=torch.qint8)
        dynamic_rows = []
        for batch_size in batch_sizes:
            dynamic_rows.append(
                benchmark_torch_forward(
                    quantized,
                    image_size,
                    batch_size,
                    max(1, args.warmup // 2),
                    max(1, args.iters // 2),
                    device=torch.device("cpu"),
                    precision="fp32",
                    label="dynamic_int8_linear_cpu",
                )
            )
        dynamic_control = {
            "method": "torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)",
            "note": "Control only: this quantizes Linear layers on CPU and does not accelerate Conv2d-heavy ArcFace R50.",
            "model_only_latency": dynamic_rows,
        }

    lfw_metrics: dict[str, Any] = {}
    if not args.skip_lfw:
        print("[Task6 final] full LFW end-to-end benchmark", flush=True)
        lfw_metrics["pytorch_fp32_cuda" if device.type == "cuda" else "pytorch_fp32_cpu"] = run_lfw_torch(
            fp32_model,
            bins,
            issame,
            image_size,
            args.lfw_batch_size,
            device,
            precision="fp32",
            label="pytorch_fp32_cuda" if device.type == "cuda" else "pytorch_fp32_cpu",
        )
        if fp16_model is not None:
            lfw_metrics["pytorch_fp16_cuda"] = run_lfw_torch(
                fp16_model,
                bins,
                issame,
                image_size,
                args.lfw_batch_size,
                device,
                precision="fp16",
                label="pytorch_fp16_cuda",
            )
        for name, session in onnx_sessions.items():
            onnx_batch_size = args.onnx_cuda_lfw_batch_size if name.endswith("_cuda") else args.lfw_batch_size
            lfw_metrics[name] = run_lfw_onnx(session, bins, issame, image_size, onnx_batch_size, label=name)

    consistency: dict[str, Any] = {}
    print("[Task6 final] ONNX/PyTorch consistency checks", flush=True)
    for name, session in onnx_sessions.items():
        consistency[name] = compare_onnx_to_torch(fp32_model, session, image_size, device, args.consistency_samples)

    source_eval = None
    source_eval_path = Path(args.source_eval_summary)
    if source_eval_path.exists():
        source_eval = json.loads(source_eval_path.read_text(encoding="utf-8"))

    sizes = {
        "checkpoint": file_size_mb(checkpoint),
        "onnx_fp32_cuda": file_size_mb(fp32_onnx),
        "onnx_fp32_cpu": file_size_mb(fp32_onnx),
        "onnx_fp16_cuda": file_size_mb(fp16_onnx),
    }
    claimed_source_lfw_accuracy = None
    if isinstance(source_eval, dict):
        claimed_source_lfw_accuracy = source_eval.get("accuracy") or source_eval.get("metrics", {}).get("accuracy")

    best_backend = None
    best_speedup = None
    reference_key = "pytorch_fp32_cuda" if device.type == "cuda" else "pytorch_fp32_cpu"
    if lfw_metrics:
        reference_latency = lfw_metrics[reference_key]["embedding_speed"]["latency_ms_per_image"]
        candidates = []
        for name, metrics in lfw_metrics.items():
            latency = metrics["embedding_speed"]["latency_ms_per_image"]
            speedup = reference_latency / max(1e-12, latency)
            metrics["speedup_vs_pytorch_fp32"] = float(speedup)
            candidates.append((speedup, name))
        best_speedup, best_backend = max(candidates)

    target_lfw_accuracy = None
    if isinstance(source_eval, dict):
        target_lfw_accuracy = source_eval.get("target_lfw_accuracy")
    if target_lfw_accuracy is None:
        target_lfw_accuracy = float(cfg.train.target_lfw_accuracy)
    local_lfw_recheck_accuracy = None
    if lfw_metrics and reference_key in lfw_metrics:
        local_lfw_recheck_accuracy = lfw_metrics[reference_key].get("accuracy")
    source_lfw_accuracy = claimed_source_lfw_accuracy
    if source_lfw_accuracy is None:
        source_lfw_accuracy = local_lfw_recheck_accuracy
    lfw_evaluation_note = (
        "Task5 acceptance uses the cloud 112x112 LFW validation set recorded in source_eval_summary. "
        "The local LFW bin in this Task6 run is retained only for same-input latency/accuracy comparison "
        "across PyTorch and ONNX backends."
    )

    conclusion = (
        "ONNX Runtime GPU/FP16 is the preferred deployment path if it beats the PyTorch CUDA baseline. "
        "Dynamic quantization is retained only as a CPU Linear-layer control because it does not target Conv2d."
    )

    summary = {
        "task": "Stage2 Task6 final InsightFace R50 latency",
        "status": "ok",
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_size_mb": file_size_mb(checkpoint),
        "source_lfw_accuracy": source_lfw_accuracy,
        "claimed_source_lfw_accuracy": claimed_source_lfw_accuracy,
        "local_lfw_recheck_accuracy": local_lfw_recheck_accuracy,
        "source_lfw_accuracy_metric": "cloud_112x112_lfw",
        "local_lfw_is_acceptance_metric": False,
        "lfw_evaluation_note": lfw_evaluation_note,
        "target_lfw_accuracy": target_lfw_accuracy,
        "target_met": bool(source_lfw_accuracy is not None and source_lfw_accuracy >= target_lfw_accuracy),
        "source_eval_summary": str(source_eval_path),
        "config": str(args.config),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "onnxruntime_available_providers": available_providers,
        "requested_providers": requested_providers,
        "setup": setup,
        "lfw_bin": str(lfw_bin),
        "lfw_pairs": len(issame),
        "lfw_images": len(bins),
        "batch_sizes": batch_sizes,
        "model_only_latency": model_only,
        "lfw_metrics": lfw_metrics,
        "consistency": consistency,
        "dynamic_quantization_control": dynamic_control,
        "artifacts": {
            "onnx_fp32": str(fp32_onnx),
            "onnx_fp16": str(fp16_onnx) if fp16_onnx.exists() else None,
            "latency_plot": str(args.plot_out),
            "report": str(args.report_out),
        },
        "artifact_sizes_mb": sizes,
        "best_backend": best_backend,
        "best_speedup_vs_pytorch_fp32": float(best_speedup) if best_speedup is not None else None,
        "conclusion": conclusion,
    }
    write_json(args.summary_out, summary)
    plot_latency(summary, args.plot_out)
    write_markdown_report(summary, args.report_out)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path("work_dirs/task5/insightface_ms1mv3_r50_full/model.pt"))
    parser.add_argument("--lfw-bin", type=Path, default=None)
    parser.add_argument("--source-eval-summary", type=Path, default=Path("reports/task5/summaries/insightface_full_lfw_eval_summary.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("work_dirs/task6/final_insightface_r50"))
    parser.add_argument("--summary-out", type=Path, default=Path("reports/task6/final/summaries/final_latency_summary.json"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/task6/final/stage2_task6_final_latency_report.md"))
    parser.add_argument("--plot-out", type=Path, default=Path("reports/task6/final/assets/evaluation/final_latency_comparison.png"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--providers", default="cuda,cpu")
    parser.add_argument("--insightface-ref", default=None)
    parser.add_argument("--batch-sizes", default="1,16,64,256")
    parser.add_argument("--lfw-batch-size", type=int, default=256)
    parser.add_argument("--onnx-cuda-lfw-batch-size", type=int, default=64)
    parser.add_argument("--onnx-cuda-max-batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--consistency-samples", type=int, default=32)
    parser.add_argument("--include-dynamic-control", action="store_true", default=True)
    parser.add_argument("--no-dynamic-control", dest="include_dynamic_control", action="store_false")
    parser.add_argument("--skip-lfw", action="store_true")
    parser.add_argument("--max-images", type=int, default=0, help="Debug only; keep 0 for full LFW.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 - write a useful failure artifact for reproducibility.
        failure = {
            "task": "Stage2 Task6 final InsightFace R50 latency",
            "status": "failed",
            "error": f"{exc.__class__.__name__}: {exc}",
            "checkpoint": str(args.checkpoint),
            "summary_out": str(args.summary_out),
        }
        try:
            write_json(args.summary_out, failure)
        finally:
            raise


if __name__ == "__main__":
    main()
