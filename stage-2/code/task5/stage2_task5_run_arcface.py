#!/usr/bin/env python3
"""Train ResNet50/IResNet50 + ArcFace and evaluate on LFW for Stage2 task 5.x."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def load_config(path: str):
    from mmengine.config import Config

    return Config.fromfile(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def load_train_rows(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        raise FileNotFoundError(f"Missing MS1MV3 train index: {index_path}")
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty MS1MV3 train index: {index_path}")
    for row in rows:
        row["label"] = int(row["label"])
    return rows


class ArcFaceImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], image_size: int, train: bool) -> None:
        self.rows = rows
        ops: list[Any] = [transforms.Resize((image_size, image_size))]
        if train:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        with Image.open(row["path"]) as handle:
            image = handle.convert("RGB")
        return self.transform(image), torch.tensor(int(row["label"]), dtype=torch.long)


class LFWImageDataset(Dataset):
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


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-5),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    def __init__(self, layers: tuple[int, int, int, int], embedding_size: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.inplanes = 64
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.PReLU(64),
        )
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512, eps=1e-5),
            nn.Dropout(p=dropout),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_size),
            nn.BatchNorm1d(embedding_size, eps=1e-5),
        )
        self._init_weights()
        self.output_layer[-1].weight.requires_grad = False

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [IBasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes, 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return F.normalize(x)


def iresnet50(embedding_size: int = 512, dropout: float = 0.0) -> IResNet:
    return IResNet((3, 4, 14, 3), embedding_size=embedding_size, dropout=dropout)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 64.0, margin: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.scale


def build_model(cfg: Any, num_classes: int, device: torch.device) -> tuple[IResNet, ArcMarginProduct]:
    if cfg.model.backbone != "iresnet50":
        raise ValueError(f"Unsupported backbone: {cfg.model.backbone}")
    backbone = iresnet50(cfg.model.embedding_size, cfg.model.get("dropout", 0.0)).to(device)
    margin = ArcMarginProduct(
        cfg.model.embedding_size,
        num_classes,
        scale=float(cfg.loss.scale),
        margin=float(cfg.loss.margin),
    ).to(device)
    return backbone, margin


def make_optimizer(cfg: Any, backbone: nn.Module, margin: nn.Module) -> torch.optim.Optimizer:
    params = list(backbone.parameters()) + list(margin.parameters())
    if cfg.train.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    return torch.optim.SGD(
        params,
        lr=float(cfg.train.lr),
        momentum=float(cfg.train.get("momentum", 0.9)),
        weight_decay=float(cfg.train.weight_decay),
    )


def is_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def dataloader_kwargs(cfg: Any) -> dict[str, Any]:
    num_workers = int(cfg.data.num_workers)
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.data.get("persistent_workers", False))
        if cfg.data.get("prefetch_factor", None) is not None:
            kwargs["prefetch_factor"] = int(cfg.data.prefetch_factor)
    return kwargs


def build_train_loader(rows: list[dict[str, Any]], cfg: Any, batch_size: int) -> DataLoader:
    dataset = ArcFaceImageDataset(rows, cfg.data.image_size, train=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs(cfg),
    )


def probe_batch(
    backbone: nn.Module,
    margin: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> None:
    images, labels = next(iter(loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    backbone.train()
    margin.train()
    with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
        embeddings = backbone(images)
        logits = margin(embeddings, labels)
        loss = F.cross_entropy(logits, labels)
    loss.backward()
    backbone.zero_grad(set_to_none=True)
    margin.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def resolve_batch_size(
    rows: list[dict[str, Any]],
    cfg: Any,
    backbone: nn.Module,
    margin: nn.Module,
    device: torch.device,
) -> tuple[DataLoader, int, int]:
    batch_size = int(cfg.train.batch_size)
    amp = bool(cfg.train.amp)
    while batch_size >= 8:
        loader = build_train_loader(rows, cfg, batch_size)
        try:
            probe_batch(backbone, margin, loader, device, amp)
            accum_steps = max(1, math.ceil(int(cfg.train.effective_batch_size) / batch_size))
            return loader, batch_size, accum_steps
        except RuntimeError as exc:
            if not is_oom(exc) or batch_size <= 8:
                raise
            batch_size //= 2
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"CUDA OOM during batch probe; retrying with batch_size={batch_size}")
    raise RuntimeError("Could not find a viable batch size.")


def checkpoint_state(
    epoch: int,
    global_step: int,
    backbone: nn.Module,
    margin: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    best_lfw_accuracy: float,
    history: list[dict[str, Any]],
    cfg: Any,
    num_classes: int,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "backbone": backbone.state_dict(),
        "margin": margin.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict(),
        "best_lfw_accuracy": best_lfw_accuracy,
        "history": history,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
        "num_classes": num_classes,
    }


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    backbone: nn.Module,
    margin: nn.Module | None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    backbone.load_state_dict(checkpoint["backbone"])
    if margin is not None and "margin" in checkpoint:
        margin.load_state_dict(checkpoint["margin"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def read_lfw_pairs(lfw_dir: Path) -> list[dict[str, Any]]:
    pairs_csv = lfw_dir / "pairs.csv"
    if not pairs_csv.exists():
        raise FileNotFoundError(f"Missing LFW pairs.csv. Run stage2_task5_3_prepare_lfw.py first: {pairs_csv}")
    with pairs_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs: list[dict[str, Any]] = []
    for row in rows:
        pairs.append(
            {
                "fold": int(row["fold"]),
                "path1": row["path1"],
                "path2": row["path2"],
                "same": bool(int(row["same"])),
            }
        )
    return pairs


@torch.no_grad()
def compute_lfw_embeddings(
    backbone: nn.Module,
    paths: list[str],
    cfg: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    dataset = LFWImageDataset(paths, cfg.data.image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs(cfg),
    )
    backbone.eval()
    embeddings: dict[str, np.ndarray] = {}
    for images, batch_paths in tqdm(loader, desc="embed LFW"):
        images = images.to(device, non_blocking=True)
        feats = backbone(images).detach().cpu().numpy()
        for path, feat in zip(batch_paths, feats):
            embeddings[str(path)] = feat.astype(np.float32)
    return embeddings


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


def plot_roc(labels: np.ndarray, scores: np.ndarray, output_path: Path) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels.astype(int), scores)
    roc_auc = float(auc(fpr, tpr))
    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"AUC {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlim(1e-4, 1.0)
    plt.ylim(0.0, 1.01)
    plt.xlabel("false accept rate")
    plt.ylabel("true accept rate")
    plt.title("LFW 6000-pair ROC")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return roc_auc


def plot_similarity_hist(labels: np.ndarray, scores: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.5))
    plt.hist(scores[labels], bins=40, alpha=0.72, label="same", color="#16a34a")
    plt.hist(scores[~labels], bins=40, alpha=0.72, label="different", color="#dc2626")
    plt.xlabel("cosine similarity")
    plt.ylabel("pair count")
    plt.title("LFW Pair Similarity Distribution")
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def tpr_at_far(labels: np.ndarray, scores: np.ndarray, fars: tuple[float, ...] = (0.001, 0.01, 0.1)) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels.astype(int), scores)
    result: dict[str, float] = {}
    for far in fars:
        valid = np.where(fpr <= far)[0]
        result[f"tpr@far={far:g}"] = float(np.max(tpr[valid])) if len(valid) else 0.0
    return result


def evaluate_lfw(
    backbone: nn.Module,
    cfg: Any,
    lfw_dir: Path,
    device: torch.device,
    roc_plot_out: Path | None = None,
    dist_plot_out: Path | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    pairs = read_lfw_pairs(lfw_dir)
    paths = sorted({item["path1"] for item in pairs} | {item["path2"] for item in pairs})
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"LFW has {len(missing)} missing images; first missing path: {missing[0]}")
    embeddings = compute_lfw_embeddings(backbone, paths, cfg, device, batch_size=batch_size)
    labels = np.array([item["same"] for item in pairs], dtype=bool)
    folds = np.array([item["fold"] for item in pairs], dtype=int)
    scores = np.array(
        [
            float(np.dot(embeddings[item["path1"]], embeddings[item["path2"]]))
            for item in pairs
        ],
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

    roc_auc = plot_roc(labels, scores, roc_plot_out) if roc_plot_out else None
    if dist_plot_out:
        plot_similarity_hist(labels, scores, dist_plot_out)
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


def plot_training_history(history: list[dict[str, Any]], output_path: Path) -> None:
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    lfw_acc = [row.get("lfw_accuracy") for row in history]

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(epochs, losses, color="#dc2626", linewidth=1.8, label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross entropy loss")
    ax1.grid(alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_acc, color="#2563eb", linewidth=1.6, label="train top-1")
    if any(value is not None for value in lfw_acc):
        ax2.plot(epochs, [np.nan if value is None else value for value in lfw_acc], color="#16a34a", linewidth=1.8, label="LFW acc")
    ax2.set_ylabel("accuracy")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    plt.title("MS1MV3 ArcFace Training")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg.train.get("cudnn_benchmark", True))
    train_rows = load_train_rows(Path(cfg.data.train_index))
    num_classes = max(int(row["label"]) for row in train_rows) + 1
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    backbone, margin = build_model(cfg, num_classes, device)
    optimizer = make_optimizer(cfg, backbone, margin)
    train_loader, actual_batch_size, accum_steps = resolve_batch_size(train_rows, cfg, backbone, margin, device)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg.train.epochs) * updates_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp) and device.type == "cuda")

    start_epoch = 1
    global_step = 0
    best_lfw_accuracy = 0.0
    history: list[dict[str, Any]] = []
    last_checkpoint = work_dir / "last.pth"
    should_resume = bool(args.resume or cfg.train.get("resume", False))
    if should_resume and last_checkpoint.exists():
        checkpoint = load_checkpoint(last_checkpoint, backbone, margin, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_lfw_accuracy = float(checkpoint.get("best_lfw_accuracy", 0.0))
        history = list(checkpoint.get("history", []))
        print(f"Resumed from {last_checkpoint} at epoch {start_epoch}")

    started = time.time()
    target_met = best_lfw_accuracy >= float(cfg.train.target_lfw_accuracy)
    for epoch in range(start_epoch, int(cfg.train.epochs) + 1):
        backbone.train()
        margin.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        for step, (images, labels) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.train.amp) and device.type == "cuda"):
                embeddings = backbone(images)
                logits = margin(embeddings, labels)
                loss = F.cross_entropy(logits, labels) / accum_steps
            scaler.scale(loss).backward()
            batch_loss = float(loss.detach().item()) * accum_steps
            preds = logits.detach().argmax(dim=1)
            running_correct += int((preds == labels).sum().item())
            running_seen += int(labels.numel())
            running_loss += batch_loss * int(labels.numel())

            if step % accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            if step % int(cfg.train.log_interval) == 0:
                progress.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{running_correct / max(1, running_seen):.4f}")

        train_loss = running_loss / max(1, running_seen)
        train_acc = running_correct / max(1, running_seen)
        epoch_row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "batch_size": actual_batch_size,
            "accum_steps": accum_steps,
        }

        if (lfw_dir := Path(cfg.data.lfw_root)).exists() and epoch % int(cfg.train.lfw_eval_interval) == 0:
            eval_metrics = evaluate_lfw(
                backbone,
                cfg,
                lfw_dir,
                device,
                roc_plot_out=None,
                dist_plot_out=None,
                batch_size=max(actual_batch_size, 128),
            )
            epoch_row["lfw_accuracy"] = eval_metrics["accuracy"]
            epoch_row["lfw_accuracy_std"] = eval_metrics["accuracy_std"]
            epoch_row["lfw_mean_threshold"] = eval_metrics["mean_threshold"]
            if eval_metrics["accuracy"] > best_lfw_accuracy:
                best_lfw_accuracy = float(eval_metrics["accuracy"])
                save_checkpoint(
                    work_dir / "best.pth",
                    checkpoint_state(
                        epoch,
                        global_step,
                        backbone,
                        margin,
                        optimizer,
                        scheduler,
                        scaler,
                        best_lfw_accuracy,
                        history + [epoch_row],
                        cfg,
                        num_classes,
                    ),
                )
            target_met = best_lfw_accuracy >= float(cfg.train.target_lfw_accuracy)
        history.append(epoch_row)

        state = checkpoint_state(
            epoch,
            global_step,
            backbone,
            margin,
            optimizer,
            scheduler,
            scaler,
            best_lfw_accuracy,
            history,
            cfg,
            num_classes,
        )
        save_checkpoint(last_checkpoint, state)
        if bool(cfg.train.save_every_epoch):
            save_checkpoint(work_dir / f"epoch_{epoch}.pth", state)
        if target_met and bool(cfg.train.stop_on_target):
            print(f"Reached LFW target {cfg.train.target_lfw_accuracy:.4f}; stopping after epoch {epoch}.")
            break
        elapsed_hours = (time.time() - started) / 3600
        if elapsed_hours > float(cfg.train.max_hours) and not target_met:
            print(
                f"Training time exceeded {cfg.train.max_hours}h and LFW target is not met; "
                "saving resume checkpoint for expansion/continuation."
            )
            break

    if not (work_dir / "best.pth").exists():
        save_checkpoint(work_dir / "best.pth", checkpoint_state(epoch, global_step, backbone, margin, optimizer, scheduler, scaler, best_lfw_accuracy, history, cfg, num_classes))
    plot_training_history(history, Path(args.loss_plot_out))
    summary = {
        "config": args.config,
        "work_dir": str(work_dir),
        "train_index": cfg.data.train_index,
        "num_images": len(train_rows),
        "num_identities": num_classes,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "actual_batch_size": actual_batch_size,
        "gradient_accumulation_steps": accum_steps,
        "effective_batch_size": actual_batch_size * accum_steps,
        "best_checkpoint": str(work_dir / "best.pth"),
        "last_checkpoint": str(last_checkpoint),
        "best_lfw_accuracy": best_lfw_accuracy,
        "target_lfw_accuracy": float(cfg.train.target_lfw_accuracy),
        "target_met": best_lfw_accuracy >= float(cfg.train.target_lfw_accuracy),
        "final_loss": history[-1]["train_loss"] if history else None,
        "history": history,
        "loss_plot": args.loss_plot_out if Path(args.loss_plot_out).exists() else "",
        "note": (
            "If target_met is false, rerun MS1MV3 preparation with a larger subset or mode=full, "
            "then resume this command. The checkpoint is not treated as final until LFW >= 98.5%."
        ),
    }
    write_json(Path(args.summary_out), summary)


def run_eval_lfw(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = int(checkpoint.get("num_classes", 1))
    backbone, margin = build_model(cfg, num_classes, device)
    load_checkpoint(Path(args.checkpoint), backbone, margin=None, map_location=device)
    dist_plot_out = Path(args.roc_plot_out).with_name("lfw_similarity_histogram.png")
    metrics = evaluate_lfw(
        backbone,
        cfg,
        Path(args.lfw_dir),
        device,
        roc_plot_out=Path(args.roc_plot_out),
        dist_plot_out=dist_plot_out,
        batch_size=max(int(cfg.train.batch_size), 128),
    )
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "lfw_dir": args.lfw_dir,
        "metrics": metrics,
        "accuracy": metrics["accuracy"],
        "target_lfw_accuracy": float(cfg.train.target_lfw_accuracy),
        "target_met": metrics["accuracy"] >= float(cfg.train.target_lfw_accuracy),
        "roc_plot": args.roc_plot_out if Path(args.roc_plot_out).exists() else "",
        "similarity_histogram": str(dist_plot_out) if dist_plot_out.exists() else "",
    }
    write_json(Path(args.summary_out), summary)
    if not summary["target_met"]:
        raise SystemExit(
            f"LFW accuracy {metrics['accuracy']:.4f} is below target {cfg.train.target_lfw_accuracy:.4f}; "
            "continue training or expand the MS1MV3 subset."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--work-dir", default="work_dirs/task5/resnet50_arcface_ms1mv3")
    train.add_argument("--summary-out", default="reports/task5/summaries/ms1mv3_train_summary.json")
    train.add_argument("--loss-plot-out", default="reports/task5/assets/training/ms1mv3_loss_acc_curve.png")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--resume", action="store_true")

    eval_lfw = subparsers.add_parser("eval-lfw")
    eval_lfw.add_argument("--config", required=True)
    eval_lfw.add_argument("--checkpoint", required=True)
    eval_lfw.add_argument("--lfw-dir", default="data/task5_lfw")
    eval_lfw.add_argument("--summary-out", default="reports/task5/summaries/lfw_eval_summary.json")
    eval_lfw.add_argument("--roc-plot-out", default="reports/task5/assets/evaluation/lfw_roc_curve.png")
    eval_lfw.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval-lfw":
        run_eval_lfw(args)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
