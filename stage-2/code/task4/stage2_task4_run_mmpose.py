#!/usr/bin/env python3
"""Run Stage2 task 4.x MMPose training and evaluation.

This wrapper mirrors MMPose's tools/train.py and tools/test.py behavior through
MMEngine Runner so the project does not need an external MMPose source checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.runner import Runner


EVAL_SPLITS = {
    "valid": "annotations/face_landmarks_300w_valid.json",
    "common": "annotations/face_landmarks_300w_valid_common.json",
    "challenge": "annotations/face_landmarks_300w_valid_challenge.json",
    "full": "annotations/face_landmarks_300w_valid.json",
    "test": "annotations/face_landmarks_300w_test.json",
}


def register_mmpose() -> None:
    try:
        from mmpose.utils import register_all_modules
    except ImportError as exc:
        raise SystemExit("Missing mmpose. Rebuild the stage2-gpu Docker image with mmpose==1.3.2.") from exc
    register_all_modules(init_default_scope=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def parse_scalar_logs(work_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scalar_logs = sorted(
        work_dir.glob("*/vis_data/scalars.json"),
        key=lambda path: path.stat().st_mtime,
    )
    paths = [scalar_logs[-1]] if scalar_logs else sorted(work_dir.rglob("*.json"))
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(item)
    return rows


def plot_loss(rows: list[dict[str, Any]], output_path: Path) -> None:
    loss_rows = [row for row in rows if "loss" in row]
    if not loss_rows:
        return
    xs = [int(row.get("step", row.get("iter", idx))) for idx, row in enumerate(loss_rows)]
    ys = [float(row["loss"]) for row in loss_rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(xs, ys, color="#2563eb", linewidth=1.8)
    plt.title("300W HRNet Landmark Training Loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def find_metric(metrics: dict[str, Any], name: str = "NME") -> float | None:
    for key, value in metrics.items():
        if key.lower() == name.lower() or key.lower().endswith("/" + name.lower()):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def plot_nme_metrics(results: dict[str, dict[str, Any]], output_path: Path) -> None:
    values = [(split, find_metric(metrics)) for split, metrics in results.items()]
    values = [(split, value) for split, value in values if value is not None]
    if not values:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [split for split, _ in values]
    nmes = [value for _, value in values]
    plt.figure(figsize=(7, 4.2))
    bars = plt.bar(labels, nmes, color=["#2563eb", "#16a34a", "#f97316", "#7c3aed"][: len(labels)])
    plt.title("300W Landmark NME")
    plt.ylabel("NME")
    plt.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, nmes):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def count_missing_images_for_cfg(cfg: Config, ann_file: str) -> int:
    dataset = cfg.test_dataloader.dataset
    data_root = Path(dataset.get("data_root", cfg.get("data_root", "")))
    data_prefix = dataset.get("data_prefix", {}).get("img", "")
    ann_path = data_root / ann_file
    image_root = data_root / data_prefix
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    missing = 0
    for image in data.get("images", []):
        if not (image_root / image["file_name"]).exists():
            missing += 1
    return missing


def copy_best_checkpoint(work_dir: Path) -> str:
    target = work_dir / "best.pth"
    candidates = sorted(work_dir.glob("best*.pth"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        candidates = sorted(work_dir.glob("epoch_*.pth"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        candidates = sorted(work_dir.rglob("*.pth"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return ""
    source = candidates[-1]
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return str(target)


def build_runner(config: str, work_dir: str = "", checkpoint: str = "", resume: bool = False) -> Runner:
    cfg = Config.fromfile(config)
    if work_dir:
        cfg.work_dir = work_dir
    if checkpoint:
        cfg.load_from = checkpoint
    if resume:
        cfg.resume = True
    cfg.launcher = "none"
    cfg.default_hooks.setdefault("logger", {})["interval"] = 20
    return Runner.from_cfg(cfg)


def run_train(args: argparse.Namespace) -> None:
    runner = build_runner(args.config, work_dir=args.work_dir, resume=args.resume)
    runner.train()
    work_dir = Path(runner.work_dir)
    scalar_rows = parse_scalar_logs(work_dir)
    plot_loss(scalar_rows, Path(args.loss_plot_out))
    loss_rows = [row for row in scalar_rows if "loss" in row]
    nme_rows = [row for row in scalar_rows if find_metric(row) is not None]
    best_checkpoint = copy_best_checkpoint(work_dir)
    summary = {
        "config": args.config,
        "work_dir": str(work_dir),
        "best_checkpoint": best_checkpoint,
        "num_logged_train_steps": len(loss_rows),
        "first_loss": float(loss_rows[0]["loss"]) if loss_rows else None,
        "last_loss": float(loss_rows[-1]["loss"]) if loss_rows else None,
        "min_loss": float(min(row["loss"] for row in loss_rows)) if loss_rows else None,
        "last_epoch": int(loss_rows[-1].get("epoch", 0)) if loss_rows else None,
        "best_logged_nme": min((find_metric(row) for row in nme_rows), default=None),
        "last_validation": nme_rows[-1] if nme_rows else {},
        "loss_curve": args.loss_plot_out if Path(args.loss_plot_out).exists() else "",
    }
    write_json(Path(args.summary_out), summary)


def run_test(args: argparse.Namespace) -> None:
    results: dict[str, dict[str, Any]] = {}
    for split, ann_file in EVAL_SPLITS.items():
        cfg = Config.fromfile(args.config)
        if args.work_dir:
            cfg.work_dir = str(Path(args.work_dir) / split)
        cfg.load_from = args.checkpoint
        cfg.launcher = "none"
        cfg.test_dataloader.dataset.ann_file = ann_file
        missing_images = count_missing_images_for_cfg(cfg, ann_file)
        if missing_images:
            if split == "test":
                results[split] = {
                    "skipped": True,
                    "ann_file": ann_file,
                    "missing_images": missing_images,
                    "reason": "Official 300W Test/ images are not present in the prepared data root.",
                }
                continue
            raise SystemExit(f"{split} split has {missing_images} missing images; rerun data preparation.")
        runner = Runner.from_cfg(cfg)
        metrics = runner.test()
        results[split] = jsonable(metrics)

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "metrics": results,
        "nme": {split: find_metric(metrics) for split, metrics in results.items()},
    }
    plot_nme_metrics(results, Path(args.metrics_plot_out))
    summary["metrics_plot"] = args.metrics_plot_out if Path(args.metrics_plot_out).exists() else ""
    write_json(Path(args.summary_out), summary)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", default="work_dirs/task4/hrnetv2_w18_300w_full")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary-out", default="reports/task4/summaries/300w_full_train_summary.json")
    parser.add_argument("--loss-plot-out", default="reports/task4/assets/training/300w_full_loss_curve.png")
    parser.add_argument("--metrics-plot-out", default="reports/task4/assets/evaluation/300w_nme_metrics.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    register_mmpose()
    if args.mode == "train":
        run_train(args)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for test mode.")
        run_test(args)


if __name__ == "__main__":
    main()
