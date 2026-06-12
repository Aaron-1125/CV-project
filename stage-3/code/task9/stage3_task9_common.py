#!/usr/bin/env python3
"""Shared helpers for Stage3 Task9 MediaPipe face effects."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import re
import runpy
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}

INSTALL_HINT = "pip install mediapipe opencv-python pillow tqdm"
NO_VIDEO_HINT = "Please place an mp4 video at reports/task9/assets/input/user_video.mp4 or pass --video <path>."

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
OUTER_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
    0, 37, 39, 40, 185,
]
INNER_LIPS = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312,
    13, 82, 81, 80, 191,
]
RIGHT_EYE_REGION = [33, 133, 159, 145, 153, 154, 155, 173, 157, 158, 160, 161, 246]
LEFT_EYE_REGION = [263, 362, 386, 374, 380, 381, 382, 398, 384, 385, 387, 388, 466]
BROW_REGION = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336]


def stage3_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cv_project_root() -> Path:
    return stage3_root().parent


def resolve_stage3_path(path: Any) -> Path:
    value = Path(str(path)).expanduser()
    if value.is_absolute():
        return value
    return stage3_root() / value


def load_config(config_path: Any) -> Dict[str, Any]:
    path = resolve_stage3_path(config_path)
    namespace = runpy.run_path(str(path))
    return {key: value for key, value in namespace.items() if not key.startswith("__")}


def cfg_get(cfg: Dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = cfg.get(section, {})
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def cfg_path(cfg: Dict[str, Any], section: str, key: str, default: str) -> Path:
    return resolve_stage3_path(cfg_get(cfg, section, key, default))


def report_dir(cfg: Dict[str, Any]) -> Path:
    return cfg_path(cfg, "reports", "report_dir", "reports/task9")


def asset_dir(cfg: Dict[str, Any]) -> Path:
    return cfg_path(cfg, "reports", "asset_dir", "reports/task9/assets")


def summary_dir(cfg: Dict[str, Any]) -> Path:
    return cfg_path(cfg, "reports", "summary_dir", "reports/task9/summaries")


def input_dir(cfg: Dict[str, Any]) -> Path:
    return asset_dir(cfg) / "input"


def static_images_dir(cfg: Dict[str, Any]) -> Path:
    return resolve_stage3_path(cfg_get(cfg, "input", "image_dir", "reports/task9/assets/input/static_images"))


def input_videos_dir(cfg: Dict[str, Any]) -> Path:
    return input_dir(cfg) / "videos"


def stickers_dir(cfg: Dict[str, Any]) -> Path:
    return asset_dir(cfg) / "stickers"


def outputs_dir(cfg: Dict[str, Any]) -> Path:
    return asset_dir(cfg) / "outputs"


def keyframes_dir(cfg: Dict[str, Any]) -> Path:
    return outputs_dir(cfg) / "keyframes"


def debug_geometry_dir(cfg: Dict[str, Any]) -> Path:
    return outputs_dir(cfg) / "debug_geometry"


def videos_dir(cfg: Dict[str, Any]) -> Path:
    return asset_dir(cfg) / "videos"


def default_video_path(cfg: Dict[str, Any]) -> Path:
    return resolve_stage3_path(cfg_get(cfg, "input", "video_path", "reports/task9/assets/input/user_video.mp4"))


def alternate_video_path(cfg: Dict[str, Any]) -> Path:
    return resolve_stage3_path(cfg_get(cfg, "input", "alternate_video_path", "reports/task9/assets/input/videos/user_video.mp4"))


def glasses_path(cfg: Dict[str, Any]) -> Path:
    return cfg_path(cfg, "stickers", "glasses_path", "reports/task9/assets/stickers/glasses.png")


def hat_path(cfg: Dict[str, Any]) -> Path:
    return cfg_path(cfg, "stickers", "hat_path", "reports/task9/assets/stickers/hat.png")


def demo_video_path(cfg: Dict[str, Any]) -> Path:
    return videos_dir(cfg) / "task9_dynamic_effects_demo.mp4"


def static_contact_sheet_path(cfg: Dict[str, Any]) -> Path:
    return outputs_dir(cfg) / "task9_static_effects_contact_sheet.jpg"


def prepare_summary_path(cfg: Dict[str, Any]) -> Path:
    return summary_dir(cfg) / "task9_prepare_summary.json"


def env_summary_path(cfg: Dict[str, Any]) -> Path:
    return summary_dir(cfg) / "task9_env_summary.json"


def effects_summary_path(cfg: Dict[str, Any]) -> Path:
    return summary_dir(cfg) / "task9_effects_summary.json"


def demo_summary_path(cfg: Dict[str, Any]) -> Path:
    return summary_dir(cfg) / "task9_demo_summary.json"


def performance_summary_path(cfg: Dict[str, Any]) -> Path:
    return summary_dir(cfg) / "task9_performance_summary.json"


def report_summary_path(cfg: Dict[str, Any]) -> Path:
    return summary_dir(cfg) / "task9_report_summary.json"


def report_path(cfg: Dict[str, Any]) -> Path:
    return report_dir(cfg) / "stage3_task9_dynamic_face_effects_report.md"


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_task9_dirs(cfg: Dict[str, Any]) -> None:
    for path in [
        report_dir(cfg),
        summary_dir(cfg),
        input_dir(cfg),
        static_images_dir(cfg),
        input_videos_dir(cfg),
        stickers_dir(cfg),
        outputs_dir(cfg),
        keyframes_dir(cfg),
        debug_geometry_dir(cfg),
        videos_dir(cfg),
    ]:
        path.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass
    return value


def write_json(path: Any, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote {}".format(target))


def read_json(path: Any) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maybe_read_json(path: Any) -> Dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        value = read_json(target)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_status(names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    status: Dict[str, Dict[str, Any]] = {}
    for name in names:
        entry: Dict[str, Any] = {"available": module_available(name)}
        if entry["available"]:
            try:
                module = __import__(name)
                entry["version"] = getattr(module, "__version__", "unknown")
            except Exception as exc:
                entry["import_error"] = "{}: {}".format(type(exc).__name__, str(exc))
        status[name] = entry
    return status


def check_runtime_dependencies() -> None:
    missing = [name for name in ["cv2", "mediapipe", "numpy", "PIL"] if not module_available(name)]
    if missing:
        raise ImportError("Missing Task9 dependencies: {}. Install with: {}".format(", ".join(missing), INSTALL_HINT))


def find_celeba_image_dir(celeba_root: Any) -> Path:
    root = Path(celeba_root).expanduser().resolve()
    candidates = [
        root,
        root / "images",
        root / "img_align_celeba",
        root / "img_align_celeba_png",
        root / "celeba" / "images",
        root / "celeba" / "img_align_celeba",
        root / "CelebA" / "Img" / "img_align_celeba",
        root / "CelebA" / "Img" / "img_align_celeba_png",
        root / "Img" / "img_align_celeba",
        root / "Img" / "img_align_celeba_png",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(child.suffix in IMAGE_EXTS for child in candidate.iterdir() if child.is_file()):
            return candidate
    for candidate in candidates:
        if candidate.is_dir():
            try:
                next(path for path in candidate.rglob("*") if path.is_file() and path.suffix in IMAGE_EXTS)
                return candidate
            except StopIteration:
                pass
    raise FileNotFoundError("Could not find CelebA image files under {}".format(root))


def list_images(directory: Any, recursive: bool = False) -> List[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Image directory does not exist: {}".format(root))
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(path for path in iterator if path.is_file() and path.suffix in IMAGE_EXTS)
    if not paths and not recursive:
        paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in IMAGE_EXTS)
    return paths


def list_videos(directory: Any, recursive: bool = False) -> List[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return []
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix in VIDEO_EXTS)


def select_samples(paths: Sequence[Path], count: int, seed: int) -> List[Path]:
    if count <= 0:
        raise ValueError("sample_count must be positive")
    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise ValueError("No images are available")
    rng = random.Random(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def safe_stem(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    clean = clean.strip("._-")
    return clean or "image"


def copy_sample(source: Path, target: Path, force: bool = False) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not force:
            return "existing"
        target.unlink()
    shutil.copy2(str(source), str(target))
    return "copy"


def locate_user_video(cfg: Dict[str, Any], explicit_video: Optional[Any] = None) -> Optional[Path]:
    candidates = []
    if explicit_video:
        candidates.append(resolve_stage3_path(explicit_video))
    candidates.extend([default_video_path(cfg), alternate_video_path(cfg)])
    for path in candidates:
        if path.is_file() and path.suffix in VIDEO_EXTS:
            return path
    return None


def relpath_for_markdown(path_value: Any, markdown_file: Path) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if path.is_absolute() and not path.exists():
        parts = path.parts
        if "stage-3" in parts:
            idx = parts.index("stage-3")
            path = stage3_root().joinpath(*parts[idx + 1 :])
    if not path.is_absolute():
        path = stage3_root() / path
    try:
        return Path(os.path.relpath(str(path), str(markdown_file.parent))).as_posix()
    except Exception:
        return str(path_value)


def polygon_center(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    if not points:
        return 0.0, 0.0
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def euclidean(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))


def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))


def quantize_number(value: float, step: float) -> float:
    step = float(step or 1.0)
    if step <= 0:
        return float(value)
    return round(float(value) / step) * step


def expanded_bbox(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    margin: float = 0.15,
    min_size: int = 8,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * float(margin)
    pad_y = height * float(margin)
    x1_i = clamp_int(math.floor(x1 - pad_x), 0, max(0, image_width - 1))
    y1_i = clamp_int(math.floor(y1 - pad_y), 0, max(0, image_height - 1))
    x2_i = clamp_int(math.ceil(x2 + pad_x), 1, image_width)
    y2_i = clamp_int(math.ceil(y2 + pad_y), 1, image_height)
    if x2_i - x1_i < min_size:
        center = (x1_i + x2_i) // 2
        x1_i = clamp_int(center - min_size // 2, 0, max(0, image_width - 1))
        x2_i = clamp_int(x1_i + min_size, 1, image_width)
    if y2_i - y1_i < min_size:
        center = (y1_i + y2_i) // 2
        y1_i = clamp_int(center - min_size // 2, 0, max(0, image_height - 1))
        y2_i = clamp_int(y1_i + min_size, 1, image_height)
    return x1_i, y1_i, x2_i, y2_i


def compute_process_size(
    source_width: int,
    source_height: int,
    target_width: int = 0,
    target_height: int = 0,
    keep_aspect_ratio: bool = True,
) -> Tuple[int, int]:
    """Return the frame size used for detection/effects/output."""
    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))
    target_width = int(target_width or 0)
    target_height = int(target_height or 0)
    if target_width <= 0 and target_height <= 0:
        return source_width, source_height
    if target_width <= 0:
        scale = target_height / float(source_height)
        return max(1, int(round(source_width * scale))), max(1, target_height)
    if target_height <= 0:
        scale = target_width / float(source_width)
        return max(1, target_width), max(1, int(round(source_height * scale)))
    if not keep_aspect_ratio:
        return max(1, target_width), max(1, target_height)
    scale = min(target_width / float(source_width), target_height / float(source_height))
    return max(1, int(round(source_width * scale))), max(1, int(round(source_height * scale)))


def mean_landmark_point(points: Any, indices: Sequence[int]) -> Tuple[float, float]:
    valid = [idx for idx in indices if idx < len(points)]
    if not valid:
        return 0.0, 0.0
    x = sum(float(points[idx][0]) for idx in valid) / len(valid)
    y = sum(float(points[idx][1]) for idx in valid) / len(valid)
    return x, y


def landmark_bbox(points: Any, indices: Sequence[int]) -> Tuple[float, float, float, float]:
    valid = [idx for idx in indices if idx < len(points)]
    if not valid:
        return 0.0, 0.0, 0.0, 0.0
    xs = [float(points[idx][0]) for idx in valid]
    ys = [float(points[idx][1]) for idx in valid]
    return min(xs), min(ys), max(xs), max(ys)


def estimate_face_transform_from_landmarks(points: Any, image_width: int, image_height: int) -> Optional[Dict[str, Any]]:
    """Estimate stable face geometry from MediaPipe Face Mesh pixel landmarks."""
    if points is None or len(points) <= max(LEFT_EYE_REGION + RIGHT_EYE_REGION + FACE_OVAL):
        return None

    eye_a = mean_landmark_point(points, RIGHT_EYE_REGION)
    eye_b = mean_landmark_point(points, LEFT_EYE_REGION)
    if eye_a[0] <= eye_b[0]:
        left_eye_center, right_eye_center = eye_a, eye_b
    else:
        left_eye_center, right_eye_center = eye_b, eye_a

    eye_distance = euclidean(left_eye_center, right_eye_center)
    if eye_distance < max(12.0, min(image_width, image_height) * 0.025):
        return None

    eye_mid = (
        (left_eye_center[0] + right_eye_center[0]) / 2.0,
        (left_eye_center[1] + right_eye_center[1]) / 2.0,
    )
    face_x_axis = (
        (right_eye_center[0] - left_eye_center[0]) / eye_distance,
        (right_eye_center[1] - left_eye_center[1]) / eye_distance,
    )
    face_up_axis = (face_x_axis[1], -face_x_axis[0])
    angle_deg = math.degrees(
        math.atan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0])
    )
    face_min_x, face_min_y, face_max_x, face_max_y = landmark_bbox(points, FACE_OVAL)
    bbox_width = max(0.0, face_max_x - face_min_x)
    cheek_width = 0.0
    if len(points) > 454:
        cheek_width = euclidean((float(points[234][0]), float(points[234][1])), (float(points[454][0]), float(points[454][1])))
    fallback_face_width = eye_distance * 2.35
    if cheek_width >= eye_distance * 1.6:
        face_width = cheek_width
    elif bbox_width >= eye_distance * 1.6:
        face_width = min(bbox_width, eye_distance * 3.2)
    else:
        face_width = fallback_face_width
    face_width = max(face_width, fallback_face_width)

    face_center = ((face_min_x + face_max_x) / 2.0, (face_min_y + face_max_y) / 2.0)
    brow_center = mean_landmark_point(points, BROW_REGION)
    if brow_center == (0.0, 0.0):
        brow_center = (
            eye_mid[0] + face_up_axis[0] * eye_distance * 0.35,
            eye_mid[1] + face_up_axis[1] * eye_distance * 0.35,
        )
    forehead_anchor = (
        brow_center[0] + face_up_axis[0] * eye_distance * 0.45,
        brow_center[1] + face_up_axis[1] * eye_distance * 0.45,
    )
    hat_anchor = (
        forehead_anchor[0] + face_up_axis[0] * eye_distance * 0.20,
        forehead_anchor[1] + face_up_axis[1] * eye_distance * 0.20,
    )
    return {
        "left_eye_center": left_eye_center,
        "right_eye_center": right_eye_center,
        "eye_center": eye_mid,
        "eye_distance": eye_distance,
        "face_x_axis": face_x_axis,
        "face_up_axis": face_up_axis,
        "angle_deg": angle_deg,
        "sticker_angle_deg": -angle_deg,
        "face_width": face_width,
        "face_bbox": (face_min_x, face_min_y, face_max_x, face_max_y),
        "face_center": face_center,
        "brow_center": brow_center,
        "forehead_anchor": forehead_anchor,
        "hat_anchor": hat_anchor,
        "hat_width": max(face_width * 1.35, eye_distance * 3.0),
        "hat_angle_deg": -angle_deg,
    }


def compute_glasses_transform(
    geometry: Dict[str, Any],
    scale_factor: float = 2.2,
    y_offset_factor: float = 0.03,
) -> Dict[str, float]:
    eye_center = geometry["eye_center"]
    eye_distance = float(geometry["eye_distance"])
    return {
        "center_x": float(eye_center[0]),
        "center_y": float(eye_center[1]) + eye_distance * float(y_offset_factor),
        "width": eye_distance * float(scale_factor),
        "angle_deg": float(geometry["sticker_angle_deg"]),
        "head_angle_deg": float(geometry["angle_deg"]),
    }


def compute_hat_transform(
    geometry: Dict[str, Any],
    sticker_aspect: float,
    scale_factor: float = 1.35,
    y_offset_factor: float = 0.55,
) -> Dict[str, float]:
    face_width = float(geometry["face_width"])
    eye_distance = float(geometry["eye_distance"])
    hat_width = max(face_width * float(scale_factor), eye_distance * 3.0)
    hat_height = hat_width * float(sticker_aspect)
    hat_anchor = geometry.get("hat_anchor")
    if not hat_anchor:
        brow_center = geometry["brow_center"]
        face_up_axis = geometry.get("face_up_axis", (0.0, -1.0))
        hat_anchor = (
            float(brow_center[0]) + float(face_up_axis[0]) * face_width * float(y_offset_factor),
            float(brow_center[1]) + float(face_up_axis[1]) * face_width * float(y_offset_factor),
        )
    return {
        "anchor_x": float(hat_anchor[0]),
        "anchor_y": float(hat_anchor[1]),
        "width": hat_width,
        "height": hat_height,
        "angle_deg": float(geometry["sticker_angle_deg"]),
        "head_angle_deg": float(geometry["angle_deg"]),
        "sticker_anchor_x": 0.5,
        "sticker_anchor_y": 0.82,
    }


def make_default_glasses(path: Path) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 520, 190
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    frame = (18, 18, 18, 245)
    shine = (255, 255, 255, 54)
    left = (42, 42, 222, 154)
    right = (298, 42, 478, 154)
    draw.rounded_rectangle(left, radius=40, outline=frame, width=16, fill=(10, 18, 25, 36))
    draw.rounded_rectangle(right, radius=40, outline=frame, width=16, fill=(10, 18, 25, 36))
    draw.line((222, 95, 298, 95), fill=frame, width=16)
    draw.line((42, 92, 4, 70), fill=frame, width=12)
    draw.line((478, 92, 516, 70), fill=frame, width=12)
    draw.arc((70, 52, 190, 140), 205, 255, fill=shine, width=6)
    draw.arc((326, 52, 446, 140), 205, 255, fill=shine, width=6)
    image.save(path)


def make_default_hat(path: Path) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 560, 310
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    crown = (34, 72, 170, 225)
    brim = (18, 16, 40, 245)
    shadow = (4, 4, 4, 78)
    draw.ellipse((60, 190, 500, 286), fill=shadow)
    draw.rounded_rectangle((108, 64, 452, 238), radius=48, fill=crown)
    draw.rectangle((116, 168, 444, 238), fill=(25, 25, 36, 230))
    draw.ellipse((45, 178, 515, 286), fill=brim)
    draw.ellipse((96, 190, 464, 252), fill=(54, 84, 175, 190))
    draw.arc((150, 80, 410, 230), 205, 335, fill=(255, 255, 255, 55), width=8)
    image.save(path)


def ensure_default_stickers(cfg: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    g_path = glasses_path(cfg)
    h_path = hat_path(cfg)
    actions = {}
    if force or not g_path.exists():
        make_default_glasses(g_path)
        actions["glasses"] = "generated"
    else:
        actions["glasses"] = "existing"
    if force or not h_path.exists():
        make_default_hat(h_path)
        actions["hat"] = "generated"
    else:
        actions["hat"] = "existing"
    return {
        "glasses_path": str(g_path),
        "hat_path": str(h_path),
        "actions": actions,
    }


def save_image_grid(image_paths: Sequence[Path], labels: Sequence[str], output_path: Path, thumb_size: int = 240) -> Optional[Path]:
    if not image_paths:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    cols = min(3, max(1, len(image_paths)))
    rows = (len(image_paths) + cols - 1) // cols
    label_h = 28
    canvas = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, path in enumerate(image_paths):
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                image.thumbnail((thumb_size, thumb_size))
        except Exception:
            continue
        x = (idx % cols) * thumb_size
        y = (idx // cols) * (thumb_size + label_h)
        draw.text((x + 6, y + 7), str(labels[idx])[:32], fill=(20, 20, 20), font=font)
        offset_x = x + (thumb_size - image.width) // 2
        offset_y = y + label_h + (thumb_size - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)
    return output_path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("Wrote {}".format(path))


def python_version_summary() -> Dict[str, Any]:
    return {
        "version": sys.version,
        "executable": sys.executable,
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "micro": sys.version_info.micro,
    }
