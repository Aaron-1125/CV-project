#!/usr/bin/env python3
"""Evaluate and visualize a WIDER FACE detector checkpoint.

The script runs MMDetection inference on an annotation list, computes IoU=0.5
precision/recall/AP, and writes visualization images for a few public WIDER
FACE validation samples.
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ImageRecord:
    image_id: str
    image_path: Path
    boxes: list[list[float]]


def read_image_ids(path: Path, limit: int = 0) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids[:limit] if limit > 0 else ids


def read_xml_record(data_root: Path, split: str, image_id: str) -> ImageRecord:
    xml_path = data_root / f"WIDER_{split}" / "Annotations" / f"{image_id}.xml"
    root = ET.parse(xml_path).getroot()
    folder = root.findtext("folder", "")
    image_path = data_root / f"WIDER_{split}" / "images" / folder / f"{image_id}.jpg"
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        boxes.append(
            [
                float(box.findtext("xmin", "0")),
                float(box.findtext("ymin", "0")),
                float(box.findtext("xmax", "0")),
                float(box.findtext("ymax", "0")),
            ]
        )
    return ImageRecord(image_id=image_id, image_path=image_path, boxes=boxes)


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def parse_prediction(prediction: dict[str, Any], score_thr: float) -> list[dict[str, Any]]:
    bboxes = prediction.get("bboxes", [])
    scores = prediction.get("scores", [])
    detections = []
    for idx, bbox in enumerate(bboxes):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        if score < score_thr:
            continue
        detections.append(
            {
                "bbox": [float(v) for v in bbox],
                "score": score,
                "label": "face",
            }
        )
    return sorted(detections, key=lambda item: item["score"], reverse=True)


def draw_detections(record: ImageRecord, detections: list[dict[str, Any]], output_path: Path, top_k: int) -> None:
    image = cv2.imread(str(record.image_path))
    if image is None:
        raise FileNotFoundError(record.image_path)
    for x1, y1, x2, y2 in record.boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (60, 160, 255), 2)
    for idx, det in enumerate(detections[:top_k], start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        score = float(det["score"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
        if idx <= 10:
            cv2.putText(
                image,
                f"{idx}:face {score:.2f}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 0),
                2,
            )
    cv2.rectangle(image, (8, 8), (250, 68), (20, 20, 20), -1)
    cv2.putText(image, "orange: GT face", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 160, 255), 2)
    cv2.putText(image, "green: prediction", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 180, 0), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def clean_visualization_outputs(input_dir: Path, detection_dir: Path) -> None:
    for directory, patterns in (
        (input_dir, ("input_*_*.jpg", "wider_val_public_*.jpg")),
        (detection_dir, ("detection_*_*.jpg", "ssd300_*_faces.jpg")),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            for path in directory.glob(pattern):
                path.unlink()


def compute_ap(matches: list[tuple[float, int]], num_gt: int) -> float:
    if num_gt == 0 or not matches:
        return 0.0
    ordered = sorted(matches, key=lambda item: item[0], reverse=True)
    tp = np.array([item[1] for item in ordered], dtype=np.float64)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(num_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # VOC-style integral AP.
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def patch_ssd_head_predict_compat(model: Any) -> None:
    """Patch MMDetection 3.3 SSDHead prediction compatibility."""

    for module in model.modules():
        if module.__class__.__name__ == "SSDHead" and not hasattr(module, "loss_cls"):
            module.loss_cls = SimpleNamespace(custom_cls_channels=False)


def evaluate_records(records: list[ImageRecord], predictions: dict[str, list[dict[str, Any]]], iou_thr: float) -> dict[str, Any]:
    total_gt = sum(len(record.boxes) for record in records)
    tp = fp = fn = 0
    ap_matches: list[tuple[float, int]] = []
    per_image = []
    for record in records:
        matched_gt: set[int] = set()
        image_tp = image_fp = 0
        for det in predictions.get(record.image_id, []):
            best_iou = 0.0
            best_idx = -1
            for gt_idx, gt_box in enumerate(record.boxes):
                if gt_idx in matched_gt:
                    continue
                iou = bbox_iou(det["bbox"], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx
            is_tp = int(best_iou >= iou_thr and best_idx >= 0)
            if is_tp:
                matched_gt.add(best_idx)
                tp += 1
                image_tp += 1
            else:
                fp += 1
                image_fp += 1
            ap_matches.append((float(det["score"]), is_tp))
        image_fn = len(record.boxes) - len(matched_gt)
        fn += image_fn
        per_image.append(
            {
                "image_id": record.image_id,
                "gt_faces": len(record.boxes),
                "detections": len(predictions.get(record.image_id, [])),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
            }
        )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    return {
        "iou_threshold": iou_thr,
        "images": len(records),
        "gt_faces": total_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "ap50": compute_ap(ap_matches, total_gt),
        "per_image": per_image,
    }


def plot_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Precision", "Recall", "AP50"]
    values = [float(metrics["precision"]), float(metrics["recall"]), float(metrics["ap50"])]
    plt.figure(figsize=(6.8, 4.2))
    bars = plt.bar(labels, values, color=["#2563eb", "#16a34a", "#f97316"])
    plt.ylim(0, max(0.05, max(values) * 1.25))
    plt.ylabel("score")
    plt.title("WIDER FACE Smoke Evaluation (IoU=0.5)")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{value:.5f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from mmdet.apis import DetInferencer
    except ImportError as exc:
        raise SystemExit("Missing MMDetection. Run inside the stage-2 Docker image or install mmdet/mmcv/mmengine.") from exc

    data_root = Path(args.data_root)
    ids = read_image_ids(data_root / args.ann_file, args.limit)
    records = [read_xml_record(data_root, args.split, image_id) for image_id in ids]
    inferencer = DetInferencer(model=args.config, weights=args.checkpoint, device=args.device)
    patch_ssd_head_predict_compat(inferencer.model)

    predictions: dict[str, list[dict[str, Any]]] = {}
    input_dir = Path(args.input_dir)
    detection_dir = Path(args.out_dir)
    clean_visualization_outputs(input_dir, detection_dir)
    visualizations = []
    visualization_pairs = []
    for idx, record in enumerate(records):
        result = inferencer(
            inputs=str(record.image_path),
            pred_score_thr=args.score_thr,
            no_save_vis=True,
            no_save_pred=True,
            return_datasamples=False,
        )
        detections = parse_prediction(result["predictions"][0], args.score_thr)
        predictions[record.image_id] = detections
        if idx < args.visualize_count:
            input_path = input_dir / f"input_{idx:02d}_{record.image_id}.jpg"
            vis_path = detection_dir / f"detection_{idx:02d}_{record.image_id}.jpg"
            shutil.copy2(record.image_path, input_path)
            draw_detections(record, detections, vis_path, args.vis_top_k)
            visualizations.append(str(vis_path))
            visualization_pairs.append(
                {
                    "index": idx,
                    "image_id": record.image_id,
                    "input": str(input_path),
                    "detection": str(vis_path),
                    "gt_color": "orange",
                    "prediction_color": "green",
                    "prediction_top_k": args.vis_top_k,
                }
            )

    metrics = evaluate_records(records, predictions, args.iou_thr)
    metrics_plot = Path(args.metrics_plot_out)
    plot_metrics(metrics, metrics_plot)
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "ann_file": args.ann_file,
        "split": args.split,
        "score_threshold": args.score_thr,
        "visualization_top_k": args.vis_top_k,
        "metrics": metrics,
        "metrics_plot": str(metrics_plot),
        "visualizations": visualizations,
        "visualization_pairs": visualization_pairs,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/WIDERFace")
    parser.add_argument("--ann-file", default="smoke_val.txt")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--input-dir", default="reports/assets/inputs/wider_val")
    parser.add_argument("--out-dir", default="reports/assets/detection")
    parser.add_argument("--summary-out", default="reports/summaries/widerface_smoke_eval_summary.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--visualize-count", type=int, default=4)
    parser.add_argument("--vis-top-k", type=int, default=20)
    parser.add_argument("--metrics-plot-out", default="reports/assets/evaluation/widerface_smoke_eval_metrics.png")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
