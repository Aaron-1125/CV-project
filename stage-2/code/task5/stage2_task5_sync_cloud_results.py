#!/usr/bin/env python3
"""Sync Task5 report assets to the AutoDL 800k/60-epoch cloud baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_SUMMARY = "reports/task5/summaries/ms1mv3_dense_train_summary.json"
LFW_SUMMARY = "reports/task5/summaries/lfw_eval_summary.json"
LFW_ROC = "reports/task5/assets/evaluation/lfw_roc_curve.png"
LFW_HIST = "reports/task5/assets/evaluation/lfw_similarity_histogram.png"
CLOUD_TRAIN_CURVE = "reports/task5/assets/training/ms1mv3_dense_loss_acc_curve.png"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def backup_existing(report_dir: Path, archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    relative_paths = [
        Path("summaries/ms1mv3_dense_train_summary.json"),
        Path("summaries/lfw_eval_summary.json"),
        Path("assets/evaluation/lfw_roc_curve.png"),
        Path("assets/evaluation/lfw_similarity_histogram.png"),
        Path("assets/training/ms1mv3_dense_loss_acc_curve.png"),
    ]
    for relative_path in relative_paths:
        source = report_dir / relative_path
        if not source.exists():
            continue
        target = archive_dir / relative_path
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"Archived {source} -> {target}")


def extract_member(tar: tarfile.TarFile, member_name: str, output_path: Path) -> None:
    member = tar.getmember(member_name)
    handle = tar.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as dst:
        shutil.copyfileobj(handle, dst)
    print(f"Extracted {member_name} -> {output_path}")


def plot_cloud_training(summary: dict[str, Any], output_path: Path) -> None:
    history = summary.get("history", [])
    if not history:
        raise ValueError("Cloud train summary has no history rows.")
    epochs = [int(row["epoch"]) for row in history]
    losses = [float(row["train_loss"]) for row in history]
    train_acc = [float(row["train_accuracy"]) for row in history]
    lfw_acc = [float(row["lfw_accuracy"]) for row in history if row.get("lfw_accuracy") is not None]
    lfw_epochs = [int(row["epoch"]) for row in history if row.get("lfw_accuracy") is not None]

    best_idx = max(range(len(lfw_acc)), key=lambda idx: lfw_acc[idx])
    best_epoch = lfw_epochs[best_idx]
    best_acc = lfw_acc[best_idx]
    target = float(summary.get("target_lfw_accuracy", 0.985))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 10.2), sharex=True)
    fig.suptitle("MS1MV3 Dense 800k ArcFace Cloud Training", fontsize=15, fontweight="bold")

    axes[0].plot(epochs, losses, color="#dc2626", linewidth=2.0)
    axes[0].set_ylabel("cross entropy loss")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.24)

    axes[1].plot(epochs, train_acc, color="#2563eb", linewidth=2.0)
    axes[1].set_ylabel("top-1 accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Closed-set training top-1")
    axes[1].grid(alpha=0.24)

    axes[2].plot(lfw_epochs, lfw_acc, color="#16a34a", linewidth=2.0, label="LFW 10-fold")
    axes[2].axhline(target, color="#f59e0b", linestyle="--", linewidth=1.6, label=f"target {target:.3f}")
    axes[2].scatter([best_epoch], [best_acc], color="#111827", s=35, zorder=3)
    axes[2].annotate(
        f"best {best_acc:.4f} @ epoch {best_epoch}",
        xy=(best_epoch, best_acc),
        xytext=(max(1, best_epoch - 18), min(0.97, best_acc + 0.055)),
        arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.0},
        fontsize=9,
        color="#111827",
    )
    axes[2].set_ylabel("LFW accuracy")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylim(0.70, 1.0)
    axes[2].set_title("Open-set LFW verification accuracy")
    axes[2].grid(alpha=0.24)
    axes[2].legend(loc="lower right", frameon=True)

    fig.text(
        0.5,
        0.012,
        "LFW plateaus while train loss keeps falling, indicating weak open-set generalization for this custom subset/pipeline.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {output_path}")


def write_baseline_note(report_dir: Path, train_summary: dict[str, Any], lfw_summary: dict[str, Any]) -> None:
    metrics = lfw_summary.get("metrics", {})
    path = report_dir / "cloud_8167_baseline_note.md"
    path.write_text(
        f"""# Task5 Cloud 8167 Baseline Note

This file records the synchronized AutoDL cloud result used by Task5 and Task6.

- train images: `{train_summary.get('num_images')}`
- identities: `{train_summary.get('num_identities')}`
- epochs completed: `{train_summary.get('epochs_completed')}`
- actual batch size: `{train_summary.get('actual_batch_size')}`
- best LFW accuracy: `{train_summary.get('best_lfw_accuracy')}`
- LFW ROC AUC: `{metrics.get('roc_auc')}`
- target met: `{train_summary.get('target_met')}`

The LFW curve starts around 0.75-0.80 because one epoch already means a full pass
over the 800k-image subset, and LFW is an aligned 1:1 verification benchmark with
threshold selection. Later epochs lower ArcFace classification loss, but LFW
does not keep improving, which points to open-set generalization limits in this
custom subset/pipeline rather than simply too few epochs.
""",
        encoding="utf-8",
    )
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-archive", type=Path, default=Path("reports/task5/task5_cloud_results_8167.tar.gz"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/task5"))
    parser.add_argument("--archive-dir", type=Path, default=Path("reports/task5/archive/local_200k_13epoch"))
    parser.add_argument("--task6-source-dir", type=Path, default=Path("reports/task6/source_task5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.cloud_archive.exists():
        raise FileNotFoundError(f"Missing cloud archive: {args.cloud_archive}")

    backup_existing(args.report_dir, args.archive_dir)

    with tarfile.open(args.cloud_archive, "r:gz") as tar:
        extract_member(tar, TRAIN_SUMMARY, args.report_dir / "summaries/ms1mv3_dense_train_summary.json")
        extract_member(tar, LFW_SUMMARY, args.report_dir / "summaries/lfw_eval_summary.json")
        extract_member(tar, LFW_ROC, args.report_dir / "assets/evaluation/lfw_roc_curve.png")
        extract_member(tar, LFW_HIST, args.report_dir / "assets/evaluation/lfw_similarity_histogram.png")
        if args.task6_source_dir:
            extract_member(tar, LFW_ROC, args.task6_source_dir / "assets/lfw_roc_curve.png")
            extract_member(tar, LFW_HIST, args.task6_source_dir / "assets/lfw_similarity_histogram.png")

    train_summary = read_json(args.report_dir / "summaries/ms1mv3_dense_train_summary.json")
    lfw_summary = read_json(args.report_dir / "summaries/lfw_eval_summary.json")
    train_curve = args.report_dir / "assets/training/ms1mv3_dense_loss_acc_curve.png"
    plot_cloud_training(train_summary, train_curve)
    if args.task6_source_dir:
        plot_cloud_training(train_summary, args.task6_source_dir / "assets/ms1mv3_dense_loss_acc_curve.png")
        shutil.copy2(args.report_dir / "summaries/ms1mv3_dense_train_summary.json", args.task6_source_dir / "ms1mv3_dense_train_summary_8167.json")
        shutil.copy2(args.report_dir / "summaries/lfw_eval_summary.json", args.task6_source_dir / "lfw_eval_summary_8167.json")
    write_baseline_note(args.report_dir, train_summary, lfw_summary)

    assert train_summary.get("num_images") == 800000
    assert train_summary.get("num_identities") == 20000
    assert train_summary.get("epochs_completed") == 60
    print("Task5 cloud 8167 report assets synchronized.")


if __name__ == "__main__":
    main()
