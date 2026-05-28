#!/usr/bin/env python3
"""Shared helpers for Stage2 task 6.x model optimization experiments."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional


def sanitize_thread_env(default: str = "4") -> None:
    """Avoid libgomp warnings from empty or malformed thread env variables."""
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
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


STAGE2_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = STAGE2_ROOT.parent
TASK5_MODULE_NAME = "stage2_task5_first_arcface"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_task5_module() -> Any:
    if TASK5_MODULE_NAME in sys.modules:
        return sys.modules[TASK5_MODULE_NAME]
    module_path = STAGE2_ROOT / "code" / "task5" / "stage2_task5_run_arcface.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing Task5 first-version ArcFace wrapper: {module_path}")
    spec = importlib.util.spec_from_file_location(TASK5_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Task5 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[TASK5_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def load_config(config_path: Path) -> Any:
    task5 = load_task5_module()
    return task5.load_config(str(config_path))


def load_first_version_backbone(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    task5 = load_task5_module()
    cfg = task5.load_config(str(config_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = int(checkpoint.get("num_classes", 1))
    backbone, _margin = task5.build_model(cfg, num_classes, device)
    task5.load_checkpoint(checkpoint_path, backbone, margin=None, map_location=device)
    backbone.eval()
    return backbone, cfg, checkpoint


def cfg_get(container: Any, key: str, default: Any = None) -> Any:
    if hasattr(container, "get"):
        return container.get(key, default)
    return getattr(container, key, default)


def resolve_lfw_image_path(path_text: str, lfw_dir: Path) -> str:
    normalized = path_text.replace("\\", "/")
    raw = Path(normalized)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                STAGE2_ROOT / raw,
                PROJECT_ROOT / raw,
                Path.cwd() / raw,
                lfw_dir / raw,
            ]
        )
        if normalized.startswith("data/task5_lfw/"):
            candidates.append(STAGE2_ROOT / normalized)
        else:
            candidates.append(lfw_dir / normalized)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve() if candidates else raw)


def read_lfw_pairs(lfw_dir: Path, max_pairs: Optional[int] = None) -> list[dict[str, Any]]:
    pairs_csv = lfw_dir / "pairs.csv"
    if not pairs_csv.exists():
        raise FileNotFoundError(f"Missing LFW pairs.csv. Run Task5 LFW preparation first: {pairs_csv}")
    with pairs_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if max_pairs is not None:
        rows = rows[: max(1, int(max_pairs))]
    pairs: list[dict[str, Any]] = []
    for row in rows:
        pairs.append(
            {
                "fold": int(row["fold"]),
                "path1": resolve_lfw_image_path(row["path1"], lfw_dir),
                "path2": resolve_lfw_image_path(row["path2"], lfw_dir),
                "same": bool(int(row["same"])),
            }
        )
    return pairs


class LFWTensorDataset(Dataset):
    def __init__(self, paths: list[str], image_size: int) -> None:
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.paths[idx]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        return self.transform(image), path


def unique_lfw_paths(pairs: list[dict[str, Any]]) -> list[str]:
    return sorted({item["path1"] for item in pairs} | {item["path2"] for item in pairs})


def make_lfw_loader(paths: list[str], image_size: int, batch_size: int, num_workers: int = 0) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": False,
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = False
    return DataLoader(LFWTensorDataset(paths, image_size), **kwargs)


def set_torch_threads(threads: Optional[int]) -> None:
    if threads is None or int(threads) <= 0:
        return
    threads = int(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        torch.set_num_threads(threads)
    except RuntimeError:
        pass


@torch.no_grad()
def extract_torch_embeddings(
    model: torch.nn.Module,
    paths: list[str],
    image_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    loader = make_lfw_loader(paths, image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    model.to(device)
    embeddings: dict[str, np.ndarray] = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for images, batch_paths in loader:
        images = images.to(device, non_blocking=False)
        feats = model(images).detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-12)
        feats = feats / norms
        for path, feat in zip(batch_paths, feats):
            embeddings[str(path)] = feat
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    stats = {
        "images": len(paths),
        "elapsed_seconds": elapsed,
        "latency_ms_per_image": (elapsed / max(1, len(paths))) * 1000.0,
        "throughput_images_per_second": len(paths) / elapsed if elapsed > 0 else 0.0,
        "batch_size": int(batch_size),
        "device": str(device),
    }
    return embeddings, stats


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(-1.0, 1.0, 2001)
    best_acc = -1.0
    best_thr = 0.0
    for threshold in thresholds:
        preds = scores >= threshold
        acc = float(np.mean(preds == labels))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(threshold)
    return best_thr, best_acc


def tpr_at_far(labels: np.ndarray, scores: np.ndarray, fars: tuple[float, ...] = (0.001, 0.01, 0.1)) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels.astype(int), scores)
    result: dict[str, float] = {}
    for far in fars:
        valid = np.where(fpr <= far)[0]
        result[f"tpr@far={far:g}"] = float(np.max(tpr[valid])) if len(valid) else 0.0
    return result


def plot_roc(labels: np.ndarray, scores: np.ndarray, output_path: Path, title: str) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels.astype(int), scores)
    roc_auc = float(auc(fpr, tpr))
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"AUC {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlim(1e-4, 1.0)
    plt.ylim(0.0, 1.01)
    plt.xlabel("false accept rate")
    plt.ylabel("true accept rate")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return roc_auc


def plot_similarity_hist(labels: np.ndarray, scores: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.4))
    plt.hist(scores[labels], bins=42, alpha=0.72, label="same", color="#16a34a")
    plt.hist(scores[~labels], bins=42, alpha=0.72, label="different", color="#dc2626")
    plt.xlabel("cosine similarity")
    plt.ylabel("pair count")
    plt.title(title)
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def compute_lfw_metrics(
    pairs: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    roc_plot_out: Optional[Path] = None,
    hist_plot_out: Optional[Path] = None,
    plot_title_prefix: str = "LFW",
) -> dict[str, Any]:
    labels = np.array([item["same"] for item in pairs], dtype=bool)
    folds = np.array([item["fold"] for item in pairs], dtype=int)
    scores = np.array(
        [float(np.dot(embeddings[item["path1"]], embeddings[item["path2"]])) for item in pairs],
        dtype=np.float32,
    )
    fold_results = []
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        test_mask = folds == fold
        threshold, train_acc = best_threshold(scores[train_mask], labels[train_mask])
        test_preds = scores[test_mask] >= threshold
        test_acc = float(np.mean(test_preds == labels[test_mask]))
        fold_results.append(
            {
                "fold": int(fold),
                "threshold": threshold,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
            }
        )
    roc_auc = plot_roc(labels, scores, roc_plot_out, f"{plot_title_prefix} ROC") if roc_plot_out else float(auc(*roc_curve(labels.astype(int), scores)[:2]))
    if hist_plot_out:
        plot_similarity_hist(labels, scores, hist_plot_out, f"{plot_title_prefix} Similarity Distribution")
    return {
        "pairs": len(pairs),
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int((~labels).sum()),
        "accuracy": float(np.mean([item["test_accuracy"] for item in fold_results])),
        "accuracy_std": float(np.std([item["test_accuracy"] for item in fold_results])),
        "mean_threshold": float(np.mean([item["threshold"] for item in fold_results])),
        "roc_auc": roc_auc,
        "folds": fold_results,
        "tpr_at_far": tpr_at_far(labels, scores),
        "score_mean_same": float(np.mean(scores[labels])),
        "score_mean_different": float(np.mean(scores[~labels])),
    }


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0) if path.exists() else 0.0


def evaluate_torch_lfw(
    model: torch.nn.Module,
    cfg: Any,
    lfw_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_pairs: Optional[int],
    roc_plot_out: Optional[Path],
    hist_plot_out: Optional[Path],
    title_prefix: str,
) -> dict[str, Any]:
    pairs = read_lfw_pairs(lfw_dir, max_pairs=max_pairs)
    paths = unique_lfw_paths(pairs)
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"LFW has {len(missing)} missing images; first missing path: {missing[0]}")
    image_size = int(cfg.data.image_size)
    embeddings, speed = extract_torch_embeddings(
        model,
        paths,
        image_size=image_size,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    metrics = compute_lfw_metrics(
        pairs,
        embeddings,
        roc_plot_out=roc_plot_out,
        hist_plot_out=hist_plot_out,
        plot_title_prefix=title_prefix,
    )
    metrics["embedding_speed"] = speed
    metrics["full_protocol"] = max_pairs is None
    return metrics


def plot_task6_comparison(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["name"] for row in rows]
    accuracies = [float(row.get("accuracy", 0.0)) for row in rows]
    latencies = [float(row.get("latency_ms_per_image", 0.0)) for row in rows]
    sizes = [float(row.get("model_size_mb", 0.0)) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    colors = ["#2563eb", "#16a34a", "#f59e0b"][: len(rows)]
    axes[0].bar(names, accuracies, color=colors)
    axes[0].set_title("LFW accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(names, latencies, color=colors)
    axes[1].set_title("Latency ms/image")
    axes[1].grid(axis="y", alpha=0.2)
    axes[2].bar(names, sizes, color=colors)
    axes[2].set_title("Model size MB")
    axes[2].grid(axis="y", alpha=0.2)
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
