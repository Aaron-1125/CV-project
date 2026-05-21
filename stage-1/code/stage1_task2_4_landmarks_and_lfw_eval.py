#!/usr/bin/env python3
"""Stage 1 task 2.4: face landmarks and LFW verification with InsightFace."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_insightface_app(det_size: int):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise SystemExit(
            "Missing insightface. Install insightface and onnxruntime, or use the stage-1 environment."
        ) from exc
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def load_recognition_model():
    try:
        from insightface.model_zoo import get_model
    except ImportError as exc:
        raise SystemExit("Missing insightface. Install insightface and onnxruntime.") from exc
    model_path = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"
    model = get_model(str(model_path), providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=-1)
    return model


def rgb_to_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.dtype != np.uint8:
        if float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def upscale_for_detection(rgb: np.ndarray, min_side: int = 224) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = max(1.0, min_side / max(1, min(height, width)))
    if scale == 1.0:
        return rgb
    return cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)


def image_hash(rgb: np.ndarray) -> str:
    digest = hashlib.sha1()
    digest.update(str(rgb.shape).encode("utf-8"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def face_area(face: Any) -> float:
    x1, y1, x2, y2 = face.bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def crop_embedding_for_image(recognizer: Any, rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_AREA)
    embedding = np.asarray(recognizer.get_feat(bgr), dtype=np.float32).reshape(-1)
    return embedding / np.linalg.norm(embedding)


def detect_embedding_for_image(app: Any, rgb: np.ndarray) -> np.ndarray | None:
    image = upscale_for_detection(rgb)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    faces = sorted(app.get(bgr), key=face_area, reverse=True)
    if not faces:
        return None
    return np.asarray(faces[0].normed_embedding, dtype=np.float32)


def embedding_for_image(
    app: Any,
    recognizer: Any,
    rgb: np.ndarray,
    cache: dict[str, Any],
    mode: str,
) -> np.ndarray | None:
    key = image_hash(rgb)
    if key in cache:
        cached = cache[key]
        return None if cached is None else np.asarray(cached, dtype=np.float32)

    if mode == "crop":
        embedding = crop_embedding_for_image(recognizer, rgb)
    else:
        embedding = detect_embedding_for_image(app, rgb)
    cache[key] = embedding
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def same_label_from_target_names(target_names: Any) -> int:
    names = [str(name).lower() for name in target_names]
    for idx, name in enumerate(names):
        if "same" in name:
            return idx
    return 1


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(float(scores.min()) - 1e-6, float(scores.max()) + 1e-6, 1000)
    accuracies = [float(np.mean((scores >= threshold) == labels)) for threshold in thresholds]
    best_idx = int(np.argmax(accuracies))
    return float(thresholds[best_idx]), float(accuracies[best_idx])


def cross_validate(scores: np.ndarray, labels: np.ndarray, folds: int = 10) -> dict[str, Any]:
    indices = np.arange(len(scores))
    fold_indices = np.array_split(indices, folds)
    rows = []
    for fold_id, valid_idx in enumerate(fold_indices, start=1):
        train_idx = np.setdiff1d(indices, valid_idx)
        threshold, train_acc = choose_threshold(scores[train_idx], labels[train_idx])
        valid_acc = float(np.mean((scores[valid_idx] >= threshold) == labels[valid_idx]))
        rows.append(
            {
                "fold": fold_id,
                "threshold": round(threshold, 6),
                "train_accuracy": round(train_acc, 6),
                "valid_accuracy": round(valid_acc, 6),
                "valid_pairs": int(len(valid_idx)),
            }
        )
    valid_accs = np.asarray([row["valid_accuracy"] for row in rows], dtype=np.float64)
    return {
        "folds": rows,
        "mean_accuracy": round(float(valid_accs.mean()), 6),
        "std_accuracy": round(float(valid_accs.std(ddof=1)), 6) if len(valid_accs) > 1 else 0.0,
    }


def save_score_plots(scores: np.ndarray, labels: np.ndarray, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "lfw_similarity_histogram.png"
    roc_path = out_dir / "lfw_roc_curve.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[labels == 1], bins=40, alpha=0.65, label="same", color="#16a34a")
    ax.hist(scores[labels == 0], bins=40, alpha=0.65, label="different", color="#dc2626")
    ax.set_title("LFW InsightFace Cosine Similarity")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("pair count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(hist_path, dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}", color="#2563eb")
    ax.plot([0, 1], [0, 1], "--", color="#64748b")
    ax.set_title("LFW Verification ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=180)
    plt.close(fig)

    return {
        "histogram": str(hist_path),
        "roc_curve": str(roc_path),
    }


def draw_landmarks(app: Any, image_path: Path, output_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    faces = sorted(app.get(image), key=face_area, reverse=True)
    annotated = image.copy()
    for face_idx, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(
            annotated,
            f"face {face_idx}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 0),
            2,
        )
        if getattr(face, "landmark_2d_106", None) is not None:
            for x, y in np.asarray(face.landmark_2d_106).astype(int):
                cv2.circle(annotated, (x, y), 1, (255, 160, 0), -1)
        if getattr(face, "kps", None) is not None:
            for x, y in np.asarray(face.kps).astype(int):
                cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    return {
        "image": str(image_path),
        "visualization": str(output_path),
        "num_faces": len(faces),
    }


def save_lfw_landmark_preview(app: Any, pairs: np.ndarray, out_dir: Path, count: int) -> list[dict[str, Any]]:
    previews = []
    for idx in range(min(count, len(pairs))):
        rgb = upscale_for_detection(rgb_to_uint8(pairs[idx, 0]), min_side=224)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tmp_path = out_dir / f"lfw_preview_{idx:02d}.jpg"
        cv2.imwrite(str(tmp_path), bgr)
        previews.append(draw_landmarks(app, tmp_path, out_dir / f"lfw_preview_{idx:02d}_landmarks.jpg"))
    return previews


def collect_landmark_images(input_dir: str) -> list[Path]:
    if not input_dir:
        return []
    root = Path(input_dir)
    if root.is_file():
        return [root]
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def run_lfw_eval(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lfw_home = data_dir / "lfw"
    cache_path = lfw_home / f"insightface_lfw_{args.embedding_mode}_embedding_cache.pkl"
    lfw_home.mkdir(parents=True, exist_ok=True)

    app = load_insightface_app(args.det_size) if args.embedding_mode == "detect" else None
    recognizer = load_recognition_model() if args.embedding_mode == "crop" else None
    pairs_data = fetch_lfw_pairs(
        data_home=str(lfw_home),
        subset="10_folds",
        color=True,
        resize=args.resize,
        download_if_missing=args.download,
    )

    pairs = pairs_data.pairs
    targets = np.asarray(pairs_data.target)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
        targets = targets[: args.max_pairs]

    cache: dict[str, Any] = {}
    if cache_path.exists():
        with cache_path.open("rb") as file:
            cache = pickle.load(file)

    same_label = same_label_from_target_names(pairs_data.target_names)
    scores = []
    labels = []
    failed_pairs = 0
    failed_images = 0
    for idx in tqdm(range(len(pairs)), desc="LFW pairs"):
        rgb_a = rgb_to_uint8(pairs[idx, 0])
        rgb_b = rgb_to_uint8(pairs[idx, 1])
        emb_a = embedding_for_image(app, recognizer, rgb_a, cache, args.embedding_mode)
        emb_b = embedding_for_image(app, recognizer, rgb_b, cache, args.embedding_mode)
        if emb_a is None or emb_b is None:
            failed_pairs += 1
            failed_images += int(emb_a is None) + int(emb_b is None)
            continue
        scores.append(cosine_similarity(emb_a, emb_b))
        labels.append(int(targets[idx]) == same_label)
        if idx % 100 == 0:
            with cache_path.open("wb") as file:
                pickle.dump(cache, file)

    with cache_path.open("wb") as file:
        pickle.dump(cache, file)

    if not scores:
        raise RuntimeError("No valid LFW pairs had two detectable faces.")

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int32)
    cv_result = cross_validate(score_array, label_array)
    auc_score = float(roc_auc_score(label_array, score_array)) if len(set(labels)) == 2 else None
    plot_paths = save_score_plots(score_array, label_array, out_dir)

    landmark_app = app or load_insightface_app(args.det_size)
    landmark_records = []
    for image_path in collect_landmark_images(args.landmark_input_dir):
        landmark_records.append(
            draw_landmarks(landmark_app, image_path, out_dir / f"{image_path.stem}_landmarks.jpg")
        )
    if not landmark_records:
        landmark_records = save_lfw_landmark_preview(landmark_app, pairs, out_dir, args.landmark_preview_count)

    summary = {
        "model": "InsightFace buffalo_l",
        "embedding_mode": args.embedding_mode,
        "det_size": args.det_size,
        "resize": args.resize,
        "total_pairs_requested": int(len(pairs)),
        "valid_pairs": int(len(score_array)),
        "failed_pairs": int(failed_pairs),
        "failed_images": int(failed_images),
        "target_names": [str(name) for name in pairs_data.target_names],
        "same_label": int(same_label),
        "mean_accuracy": cv_result["mean_accuracy"],
        "std_accuracy": cv_result["std_accuracy"],
        "auc": round(auc_score, 6) if auc_score is not None else None,
        "folds": cv_result["folds"],
        "plots": plot_paths,
        "landmark_visualizations": landmark_records,
    }
    summary_path = out_dir / "lfw_insightface_verification_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="reports/assets")
    parser.add_argument("--download", action="store_true", help="Download missing LFW data.")
    parser.add_argument("--resize", type=float, default=0.5)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument(
        "--embedding-mode",
        choices=["crop", "detect"],
        default="crop",
        help="Use direct buffalo_l recognition on LFW crops, or detect faces before embedding.",
    )
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--landmark-input-dir", default="")
    parser.add_argument("--landmark-preview-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    run_lfw_eval(parse_args())


if __name__ == "__main__":
    main()
