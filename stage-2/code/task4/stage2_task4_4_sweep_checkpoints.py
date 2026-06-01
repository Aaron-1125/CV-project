#!/usr/bin/env python3
"""Evaluate Task4 checkpoints on 300W valid/common/challenge splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.runner import Runner

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_task4_run_mmpose import (  # noqa: E402
    EVAL_SPLITS,
    count_missing_images_for_cfg,
    find_metric,
    jsonable,
    register_mmpose,
)


def collect_checkpoints(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.checkpoints:
        paths.extend(Path(item.strip()) for item in args.checkpoints.split(",") if item.strip())
    if args.checkpoint_dir:
        directory = Path(args.checkpoint_dir)
        paths.extend(sorted(directory.glob("best*.pth")))
        paths.extend(sorted(directory.glob("epoch_*.pth")))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path)] = path
    return list(unique.values())


def evaluate_checkpoint(config: str, checkpoint: Path, work_dir: Path, splits: list[str]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for split in splits:
        ann_file = EVAL_SPLITS[split]
        cfg = Config.fromfile(config)
        cfg.work_dir = str(work_dir / checkpoint.stem / split)
        cfg.load_from = str(checkpoint)
        cfg.launcher = "none"
        cfg.test_dataloader.dataset.ann_file = ann_file
        missing_images = count_missing_images_for_cfg(cfg, ann_file)
        if missing_images:
            results[split] = {
                "skipped": True,
                "ann_file": ann_file,
                "missing_images": missing_images,
            }
            continue
        runner = Runner.from_cfg(cfg)
        metrics = runner.test()
        results[split] = jsonable(metrics)
    return results


def plot_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [Path(row["checkpoint"]).stem for row in rows]
    x = range(len(rows))
    plt.figure(figsize=(max(8, len(rows) * 1.0), 4.6))
    for split, color in (("common", "#16a34a"), ("challenge", "#f97316"), ("valid", "#2563eb")):
        values = [row.get(f"{split}_nme") for row in rows]
        if any(value is not None for value in values):
            plt.plot(x, values, marker="o", linewidth=1.8, label=split, color=color)
    plt.xticks(list(x), labels, rotation=35, ha="right")
    plt.ylabel("NME")
    plt.title("Task4 v2 checkpoint sweep")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    register_mmpose()
    checkpoints = collect_checkpoints(args)
    if not checkpoints:
        raise SystemExit("No checkpoints found. Pass --checkpoint-dir or --checkpoints.")
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    rows = []
    raw_results = {}
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            rows.append({"checkpoint": str(checkpoint), "missing": True})
            continue
        metrics = evaluate_checkpoint(args.config, checkpoint, Path(args.work_dir), splits)
        raw_results[str(checkpoint)] = metrics
        row: dict[str, Any] = {"checkpoint": str(checkpoint), "missing": False}
        for split, values in metrics.items():
            row[f"{split}_nme"] = find_metric(values) if isinstance(values, dict) else None
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    valid_rows = [row for row in rows if not row.get("missing")]
    best_by = {}
    for split in splits:
        split_rows = [row for row in valid_rows if row.get(f"{split}_nme") is not None]
        if split_rows:
            best_by[split] = min(split_rows, key=lambda row: row[f"{split}_nme"])

    plot_path = Path(args.plot_out)
    plot_rows(valid_rows, plot_path)
    summary = {
        "config": args.config,
        "splits": splits,
        "rows": rows,
        "best_by": best_by,
        "raw_results": raw_results,
        "plot": str(plot_path) if plot_path.exists() else "",
    }
    output_path = Path(args.summary_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoints", default="", help="Comma-separated checkpoint paths.")
    parser.add_argument("--splits", default="valid,common,challenge")
    parser.add_argument("--work-dir", default="work_dirs/task4_v2/checkpoint_sweep")
    parser.add_argument("--summary-out", default="reports/task4_v2/summaries/checkpoint_sweep_summary.json")
    parser.add_argument("--plot-out", default="reports/task4_v2/assets/evaluation/checkpoint_sweep.png")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
