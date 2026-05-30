#!/usr/bin/env python3
"""Run a score/max-per-image sweep for WIDER FACE detector diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_task3_3_evaluate_widerface import (  # noqa: E402
    evaluate_records,
    parse_prediction,
    patch_ssd_head_predict_compat,
    read_image_ids,
    read_xml_record,
)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def current_nms_iou(config_path: str) -> float | None:
    try:
        from mmengine.config import Config
    except ImportError:
        return None
    cfg = Config.fromfile(config_path)
    try:
        return float(cfg.model.test_cfg.nms.iou_threshold)
    except Exception:
        return None


def plot_sweep(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_values = sorted({int(row["max_per_img"]) for row in rows})
    metrics = [
        ("ap50", "AP50", "#f97316"),
        ("precision", "Precision", "#2563eb"),
        ("recall", "Recall", "#16a34a"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharex=True)
    for ax, (key, title, color) in zip(axes, metrics):
        for max_per_img in max_values:
            subset = [row for row in rows if int(row["max_per_img"]) == max_per_img]
            subset = sorted(subset, key=lambda item: float(item["score_thr"]))
            ax.plot(
                [float(row["score_thr"]) for row in subset],
                [float(row[key]) for row in subset],
                marker="o",
                linewidth=1.8,
                label=f"max {max_per_img}",
                color=color if len(max_values) == 1 else None,
            )
        ax.set_title(title)
        ax.set_xlabel("score threshold")
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0.0)
    axes[0].set_ylabel("metric")
    axes[-1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from mmdet.apis import DetInferencer
    except ImportError as exc:
        raise SystemExit("Missing MMDetection. Run this script inside the stage-2 GPU Docker environment.") from exc

    score_thrs = parse_float_list(args.score_thrs)
    max_per_img_values = parse_int_list(args.max_per_img_values)
    min_score_thr = min(score_thrs)
    data_root = Path(args.data_root)
    image_ids = read_image_ids(data_root / args.ann_file, args.limit)
    records = [read_xml_record(data_root, args.split, image_id) for image_id in image_ids]
    inferencer = DetInferencer(model=args.config, weights=args.checkpoint, device=args.device)
    patch_ssd_head_predict_compat(inferencer.model)

    raw_predictions: dict[str, list[dict[str, Any]]] = {}
    for idx, record in enumerate(records, start=1):
        result = inferencer(
            inputs=str(record.image_path),
            pred_score_thr=min_score_thr,
            no_save_vis=True,
            no_save_pred=True,
            return_datasamples=False,
        )
        raw_predictions[record.image_id] = parse_prediction(result["predictions"][0], min_score_thr)
        if idx % args.log_interval == 0:
            print(f"inferred {idx}/{len(records)} images")

    rows = []
    fixed_nms_iou = current_nms_iou(args.config)
    for max_per_img in max_per_img_values:
        for score_thr in score_thrs:
            predictions = {
                image_id: [det for det in detections if float(det["score"]) >= score_thr][:max_per_img]
                for image_id, detections in raw_predictions.items()
            }
            metrics = evaluate_records(records, predictions, args.iou_thr)
            rows.append(
                {
                    "score_thr": score_thr,
                    "max_per_img": max_per_img,
                    "nms_iou": fixed_nms_iou,
                    "images": metrics["images"],
                    "gt_faces": metrics["gt_faces"],
                    "detections": metrics["tp"] + metrics["fp"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "ap50": metrics["ap50"],
                    "fp_per_image": metrics["fp"] / max(metrics["images"], 1),
                }
            )

    plot_path = Path(args.plot_out)
    plot_sweep(rows, plot_path)
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "ann_file": args.ann_file,
        "split": args.split,
        "iou_thr": args.iou_thr,
        "note": "NMS IoU is read from the model config. This sweep reuses one inference pass and varies score_thr and max_per_img.",
        "rows": rows,
        "plot": str(plot_path),
    }
    output_path = Path(args.summary_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/WIDERFace")
    parser.add_argument("--ann-file", default="val.txt")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thrs", default="0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--max-per-img-values", default="50,100,200")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--summary-out", default="reports/task3_v2/summaries/ssd300_threshold_sweep_summary.json")
    parser.add_argument("--plot-out", default="reports/task3_v2/assets/diagnostics/ssd300_threshold_sweep.png")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
