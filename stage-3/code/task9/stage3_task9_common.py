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
