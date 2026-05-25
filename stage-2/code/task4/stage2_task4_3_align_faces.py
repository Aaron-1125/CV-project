#!/usr/bin/env python3
"""Visualize 300W landmarks and align faces with a trained MMPose checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def register_mmpose() -> None:
    try:
        from mmpose.utils import register_all_modules
    except ImportError as exc:
        raise SystemExit("Missing mmpose. Rebuild the stage2-gpu Docker image with mmpose==1.3.2.") from exc
    register_all_modules(init_default_scope=True)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def bbox_from_center_scale(center: list[float], scale: float) -> np.ndarray:
    size = float(scale) * 200.0
    cx, cy = float(center[0]), float(center[1])
    return np.array([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2], dtype=np.float32)


def load_samples(data_dir: Path, ann_file: str, count: int) -> list[dict[str, Any]]:
    ann_path = data_dir / ann_file
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in data.get("images", [])}
    samples = []
    for ann in data.get("annotations", []):
        image = images.get(ann["image_id"])
        if not image:
            continue
        image_path = data_dir / "images" / image["file_name"]
        if not image_path.exists():
            continue
        samples.append(
            {
                "image_id": image["id"],
                "file_name": image["file_name"],
                "image_path": image_path,
                "bbox": bbox_from_center_scale(ann["center"], ann["scale"]),
            }
        )
        if len(samples) >= count:
            break
    return samples


def predict_keypoints(model: Any, image_path: Path, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from mmpose.apis import inference_topdown

    results = inference_topdown(model, str(image_path), bboxes=bbox[None, :], bbox_format="xyxy")
    if not results:
        raise RuntimeError(f"No pose result for {image_path}")
    pred = results[0].pred_instances
    keypoints = tensor_to_numpy(pred.keypoints)
    scores = tensor_to_numpy(pred.keypoint_scores)
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if scores.ndim == 2:
        scores = scores[0]
    return keypoints.astype(np.float32), scores.astype(np.float32)


def five_points_from_68(keypoints: np.ndarray) -> np.ndarray:
    left_eye = keypoints[36:42].mean(axis=0)
    right_eye = keypoints[42:48].mean(axis=0)
    nose = keypoints[30]
    left_mouth = keypoints[48]
    right_mouth = keypoints[54]
    return np.stack([left_eye, right_eye, nose, left_mouth, right_mouth]).astype(np.float32)


def estimate_affine(keypoints: np.ndarray) -> np.ndarray:
    src = five_points_from_68(keypoints)
    matrix, _ = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if matrix is None:
        matrix = cv2.getAffineTransform(src[:3], ARCFACE_TEMPLATE[:3])
    return matrix.astype(np.float32)


def clip_bbox(bbox: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(width, int(np.ceil(x2))),
        min(height, int(np.ceil(y2))),
    )


def draw_overlay(image: np.ndarray, bbox: np.ndarray, keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 0), 2)
    for idx, (x, y) in enumerate(keypoints):
        color = (0, 80, 255) if scores[idx] >= 0.5 else (80, 80, 180)
        cv2.circle(out, (int(round(x)), int(round(y))), 2, color, -1)
    return out


def make_grid(image: np.ndarray, bbox: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, width, height)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    aligned_big = cv2.resize(aligned, (224, 224), interpolation=cv2.INTER_CUBIC)
    grid = np.concatenate([crop, aligned_big], axis=1)
    cv2.putText(grid, "before", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(grid, "aligned", (236, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return grid


def safe_stem(file_name: str, idx: int) -> str:
    stem = Path(file_name).with_suffix("").as_posix().replace("/", "_")
    return f"{idx:02d}_{stem}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/task4_300w/mmpose/300w")
    parser.add_argument("--ann-file", default="annotations/face_landmarks_300w_valid.json")
    parser.add_argument("--out-dir", default="reports/task4/assets/alignment")
    parser.add_argument("--summary-out", default="reports/task4/summaries/300w_alignment_summary.json")
    parser.add_argument("--visualize-count", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    register_mmpose()
    from mmpose.apis import init_model

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = init_model(args.config, args.checkpoint, device=args.device)
    samples = load_samples(data_dir, args.ann_file, args.visualize_count)
    if not samples:
        raise SystemExit(f"No samples with existing images found in {data_dir / args.ann_file}")

    records = []
    for idx, sample in enumerate(samples):
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            continue
        keypoints, scores = predict_keypoints(model, sample["image_path"], sample["bbox"])
        matrix = estimate_affine(keypoints)
        aligned = cv2.warpAffine(image, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        overlay = draw_overlay(image, sample["bbox"], keypoints, scores)
        grid = make_grid(image, sample["bbox"], aligned)

        stem = safe_stem(sample["file_name"], idx)
        overlay_path = out_dir / f"{stem}_landmarks.jpg"
        aligned_path = out_dir / f"{stem}_aligned.jpg"
        grid_path = out_dir / f"{stem}_before_after.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(aligned_path), aligned)
        cv2.imwrite(str(grid_path), grid)
        records.append(
            {
                "image": str(sample["image_path"]),
                "landmark_overlay": str(overlay_path),
                "aligned_face": str(aligned_path),
                "before_after": str(grid_path),
                "num_keypoints": int(keypoints.shape[0]),
                "mean_keypoint_score": float(np.mean(scores)),
                "affine_matrix": matrix.tolist(),
            }
        )

    write_json(
        Path(args.summary_out),
        {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "data_dir": str(data_dir),
            "ann_file": args.ann_file,
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
