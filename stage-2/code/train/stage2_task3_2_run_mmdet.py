#!/usr/bin/env python3
"""Run MMDetection training or testing from a local config.

This wrapper avoids depending on an external mmdetection source checkout with
tools/train.py. It uses MMEngine Runner directly, so the pip-installed mmdet
package is enough once mmcv/mmengine/mmdet are installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.runner import Runner


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
    rows = [row for row in rows if "loss" in row]
    if not rows:
        return
    xs = [int(row.get("step", row.get("iter", idx))) for idx, row in enumerate(rows)]
    ys = [float(row["loss"]) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(xs, ys, color="#2563eb", linewidth=1.8)
    plt.title("WIDER FACE Training Loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def latest_checkpoint(work_dir: Path) -> str:
    marker = work_dir / "last_checkpoint"
    if marker.exists():
        candidate = Path(marker.read_text(encoding="utf-8").strip())
        if candidate.exists():
            return str(candidate)
    checkpoints = sorted(work_dir.rglob("*.pth"), key=lambda path: path.stat().st_mtime)
    return str(checkpoints[-1]) if checkpoints else ""


def write_summary(work_dir: Path, summary_path: Path, plot_path: Path | None) -> dict[str, Any]:
    scalar_rows = parse_scalar_logs(work_dir)
    rows = [row for row in scalar_rows if "loss" in row]
    val_rows = [row for row in scalar_rows if "pascal_voc/mAP" in row]
    if plot_path:
        plot_loss(rows, plot_path)
    summary = {
        "work_dir": str(work_dir),
        "latest_checkpoint": latest_checkpoint(work_dir),
        "num_logged_train_steps": len(rows),
        "first_loss": float(rows[0]["loss"]) if rows else None,
        "last_loss": float(rows[-1]["loss"]) if rows else None,
        "min_loss": float(min(row["loss"] for row in rows)) if rows else None,
        "validation": val_rows[-1] if val_rows else {},
        "loss_curve": str(plot_path) if plot_path and plot_path.exists() else "",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return summary


def build_runner(args: argparse.Namespace) -> Runner:
    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.resume:
        cfg.resume = True
    if args.load_from:
        cfg.load_from = args.load_from
    if args.device:
        cfg.default_hooks.setdefault("logger", {}).setdefault("interval", 10)
        cfg.launcher = "none"
    runner = Runner.from_cfg(cfg)
    patch_ssd_head_predict_compat(runner.model)
    return runner


def patch_ssd_head_predict_compat(model: Any) -> None:
    """Patch MMDetection 3.3 SSDHead prediction compatibility.

    SSDHead implements its loss directly, while the shared dense-head predict
    path probes ``loss_cls.custom_cls_channels``. Adding this minimal marker is
    enough for validation/inference and does not change trainable parameters.
    """

    for module in model.modules():
        if module.__class__.__name__ == "SSDHead" and not hasattr(module, "loss_cls"):
            module.loss_cls = SimpleNamespace(custom_cls_channels=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--load-from", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu", help="Documentary flag; device is controlled by the config/runtime.")
    parser.add_argument("--summary-out", default="reports/summaries/widerface_smoke_train_summary.json")
    parser.add_argument("--loss-plot-out", default="reports/assets/training/smoke_loss_curve.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = build_runner(args)
    if args.mode == "train":
        runner.train()
        write_summary(Path(runner.work_dir), Path(args.summary_out), Path(args.loss_plot_out))
    else:
        runner.test()


if __name__ == "__main__":
    main()
