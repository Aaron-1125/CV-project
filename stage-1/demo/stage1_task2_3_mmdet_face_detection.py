#!/usr/bin/env python3
"""Stage 1 task 2.3: open-vocabulary face detection with MMDetection.

This uses the MMDetection GroundingDINO implementation with the text prompt
"face . human face ." to produce face boxes on provided images.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import cv2


CONFIG_FILES = {
    "configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py": "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py",
    "configs/_base_/datasets/coco_detection.py": "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/datasets/coco_detection.py",
    "configs/_base_/schedules/schedule_1x.py": "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/schedules/schedule_1x.py",
    "configs/_base_/default_runtime.py": "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/default_runtime.py",
}
DEFAULT_WEIGHT_URL = "https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
DEFAULT_WEIGHT_MIN_BYTES = 600_000_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as file:
        shutil.copyfileobj(response, file)
    tmp_path.replace(output_path)


def ensure_file(url: str, output_path: Path, skip_download: bool, min_bytes: int = 1) -> None:
    if output_path.exists() and output_path.stat().st_size >= min_bytes:
        return
    if output_path.exists():
        output_path.unlink()
    if skip_download:
        raise FileNotFoundError(f"Missing file and --skip-download was set: {output_path}")
    download_file(url, output_path)
    if output_path.stat().st_size < min_bytes:
        raise RuntimeError(
            f"Downloaded file is too small: {output_path} "
            f"({output_path.stat().st_size} bytes, expected at least {min_bytes})"
        )


def ensure_mmdet_config(checkpoint_dir: Path, skip_download: bool) -> Path:
    config_root = checkpoint_dir / "mmdetection_configs"
    for relative_path, url in CONFIG_FILES.items():
        ensure_file(url, config_root / relative_path, skip_download)
    return config_root / "configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py"


def collect_images(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]
    images = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No images found under {input_dir}")
    return images


def draw_boxes(image_path: Path, detections: list[dict[str, Any]], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    for idx, item in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        score = float(item["score"])
        label = str(item.get("label", "face"))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(
            image,
            f"{idx}:{label} {score:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 0),
            2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def parse_prediction(prediction: dict[str, Any], score_thr: float) -> list[dict[str, Any]]:
    bboxes = prediction.get("bboxes", [])
    scores = prediction.get("scores", [])
    labels = prediction.get("labels", [])
    label_names = prediction.get("label_names", [])
    detections = []
    for idx, bbox in enumerate(bboxes):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        if score < score_thr:
            continue
        label = labels[idx] if idx < len(labels) else "face"
        if idx < len(label_names):
            label = label_names[idx]
        elif str(label).isdigit():
            label = "face"
        detections.append(
            {
                "bbox": [round(float(v), 2) for v in bbox],
                "score": round(score, 4),
                "label": str(label),
            }
        )
    return nms_detections(detections)


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


def nms_detections(detections: list[dict[str, Any]], iou_thr: float = 0.7) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(detections, key=lambda det: float(det["score"]), reverse=True):
        if all(bbox_iou(item["bbox"], kept_item["bbox"]) < iou_thr for kept_item in kept):
            kept.append(item)
    return kept


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from mmdet.apis import DetInferencer
    except ImportError as exc:
        raise SystemExit(
            "Missing MMDetection. Use the Docker environment or install mmdet/mmcv/mmengine."
        ) from exc

    checkpoint_dir = Path(args.checkpoint_dir)
    config_path = Path(args.config) if args.config else ensure_mmdet_config(checkpoint_dir, args.skip_download)
    weights_path = Path(args.weights) if args.weights else checkpoint_dir / "groundingdino_swint_ogc_mmdet-822d7e9d.pth"
    if not args.weights:
        ensure_file(DEFAULT_WEIGHT_URL, weights_path, args.skip_download, DEFAULT_WEIGHT_MIN_BYTES)

    images = collect_images(Path(args.input_dir))
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inferencer = DetInferencer(
        model=str(config_path),
        weights=str(weights_path),
        device=args.device,
    )

    records = []
    for image_path in images:
        print(f"Running MMDetection on {image_path}")
        result = inferencer(
            inputs=str(image_path),
            texts=args.texts,
            custom_entities=True,
            pred_score_thr=args.score_thr,
            no_save_vis=True,
            no_save_pred=True,
            return_datasamples=False,
        )
        prediction = result["predictions"][0]
        detections = parse_prediction(prediction, args.score_thr)
        vis_path = output_dir / f"mmdet_{image_path.stem}_faces.jpg"
        draw_boxes(image_path, detections, vis_path)
        records.append(
            {
                "image": str(image_path),
                "visualization": str(vis_path),
                "detections": detections,
                "num_detections": len(detections),
            }
        )

    summary = {
        "model": "MMDetection GroundingDINO",
        "config": str(config_path),
        "weights": str(weights_path),
        "texts": args.texts,
        "score_threshold": args.score_thr,
        "images": records,
    }
    summary_path = output_dir / "mmdet_face_detection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Image file or directory of test images.")
    parser.add_argument("--out-dir", default="reports/assets/detection")
    parser.add_argument("--checkpoint-dir", default="checkpoints/mmdet")
    parser.add_argument("--config", default="", help="Optional local MMDetection config path.")
    parser.add_argument("--weights", default="", help="Optional local checkpoint path.")
    parser.add_argument("--texts", default="face . human face .")
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
