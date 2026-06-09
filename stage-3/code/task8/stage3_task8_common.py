#!/usr/bin/env python3
"""Shared helpers for Stage3 Task8 official 3DDFA_V2 wrappers."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}
DEFAULT_OFFICIAL_OUTPUTS = ["2d_sparse", "3d", "pose", "obj"]


def stage3_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cv_project_root() -> Path:
    return stage3_root().parent


def task_root() -> Path:
    return cv_project_root().parent


def resolve_stage3_path(path: Any) -> Path:
    path = Path(str(path)).expanduser()
    if path.is_absolute():
        return path
    return stage3_root() / path


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
    return cfg_path(cfg, "reports", "report_dir", "reports/task8")


def asset_dir(cfg: Dict[str, Any]) -> Path:
    return cfg_path(cfg, "reports", "asset_dir", "reports/task8/assets")


def summary_dir(cfg: Dict[str, Any]) -> Path:
    return cfg_path(cfg, "reports", "summary_dir", "reports/task8/summaries")


def input_samples_dir(cfg: Dict[str, Any]) -> Path:
    return asset_dir(cfg) / "input_samples"


def reconstruction_dir(cfg: Dict[str, Any]) -> Path:
    return asset_dir(cfg) / "reconstruction"


def rendered_views_dir(cfg: Dict[str, Any]) -> Path:
    return asset_dir(cfg) / "rendered_views"


def prepare_summary_path(cfg: Dict[str, Any]) -> Path:
    return summary_dir(cfg) / "task8_prepare_summary.json"


def reconstruction_summary_path(cfg: Dict[str, Any]) -> Path:
    return summary_dir(cfg) / "task8_reconstruction_summary.json"


def render_summary_path(cfg: Dict[str, Any]) -> Path:
    return summary_dir(cfg) / "task8_render_summary.json"


def env_summary_path(cfg: Dict[str, Any]) -> Path:
    return summary_dir(cfg) / "task8_env_summary.json"


def report_path(cfg: Dict[str, Any]) -> Path:
    return report_dir(cfg) / "stage3_task8_3d_face_reconstruction_report.md"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote {}".format(path))


def read_json(path: Any) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def three_ddfa_repo(cfg: Dict[str, Any]) -> Path:
    override = os.environ.get("3DDFA_REPO")
    configured = cfg_get(cfg, "third_party", "repo_path", str(task_root() / "3DDFA_V2"))
    return Path(override or configured).expanduser().resolve()


def official_config_path(cfg: Dict[str, Any], repo: Optional[Path] = None) -> Path:
    repo = repo or three_ddfa_repo(cfg)
    configured = cfg_get(cfg, "third_party", "official_config", "configs/mb1_120x120.yml")
    path = Path(str(configured))
    return path if path.is_absolute() else repo / path


def repo_commit(repo: Path) -> Optional[str]:
    if not (repo / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def inspect_3ddfa_repo(cfg: Dict[str, Any]) -> Dict[str, Any]:
    repo = three_ddfa_repo(cfg)
    required_paths = list(
        cfg_get(
            cfg,
            "third_party",
            "required_paths",
            [
                "demo.py",
                "TDDFA.py",
                "TDDFA_ONNX.py",
                "configs/mb1_120x120.yml",
                "FaceBoxes",
                "utils/serialization.py",
                "utils/functions.py",
            ],
        )
    )
    missing = [rel for rel in required_paths if not (repo / rel).exists()]

    checkpoint_candidates = list(
        cfg_get(
            cfg,
            "third_party",
            "checkpoint_paths",
            ["weights/mb1_120x120.pth", "weights/mb1_120x120.onnx"],
        )
    )
    checkpoints = []
    for rel in checkpoint_candidates:
        path = repo / rel
        checkpoints.append({"path": str(path), "relative_path": rel, "exists": path.exists()})

    faceboxes_candidates = list(
        cfg_get(
            cfg,
            "third_party",
            "faceboxes_checkpoint_paths",
            [
                "FaceBoxes/weights/FaceBoxesProd.pth",
                "FaceBoxes/weights/FaceBoxesProd.onnx",
            ],
        )
    )
    faceboxes = []
    for rel in faceboxes_candidates:
        path = repo / rel
        faceboxes.append({"path": str(path), "relative_path": rel, "exists": path.exists()})
    has_faceboxes_weight = any(item["exists"] for item in faceboxes)

    pth_path = repo / str(cfg_get(cfg, "third_party", "preferred_checkpoint", "weights/mb1_120x120.pth"))
    onnx_path = repo / str(cfg_get(cfg, "third_party", "preferred_onnx", "weights/mb1_120x120.onnx"))
    has_pth = pth_path.exists()
    has_onnx = onnx_path.exists()
    ready = repo.exists() and not missing and (has_pth or has_onnx) and has_faceboxes_weight
    return {
        "repo_path": str(repo),
        "repo_exists": repo.exists(),
        "commit": repo_commit(repo) if repo.exists() else None,
        "required_paths": required_paths,
        "missing_required_paths": missing,
        "official_config": str(official_config_path(cfg, repo)),
        "preferred_checkpoint": str(pth_path),
        "preferred_checkpoint_exists": has_pth,
        "preferred_onnx": str(onnx_path),
        "preferred_onnx_exists": has_onnx,
        "checkpoint_candidates": checkpoints,
        "faceboxes_checkpoint_candidates": faceboxes,
        "faceboxes_checkpoint_exists": has_faceboxes_weight,
        "ready": ready,
        "hint": build_repo_hint(repo, missing, has_pth, has_onnx, has_faceboxes_weight),
    }


def build_repo_hint(repo: Path, missing: List[str], has_pth: bool, has_onnx: bool, has_faceboxes_weight: bool) -> str:
    if not repo.exists():
        return (
            "Missing 3DDFA_V2 repo. Clone it outside stage-3, for example: "
            "git clone https://github.com/cleardusk/3DDFA_V2.git /root/autodl-tmp/task/3DDFA_V2"
        )
    if missing:
        return "3DDFA_V2 repo exists but is missing required paths: {}".format(", ".join(missing))
    if not (has_pth or has_onnx):
        return (
            "Missing 3DDFA_V2 model weights. Prepare weights/mb1_120x120.pth or "
            "weights/mb1_120x120.onnx inside the external 3DDFA_V2 repo."
        )
    if not has_faceboxes_weight:
        return (
            "Missing FaceBoxes detector weights. Prepare FaceBoxes/weights/FaceBoxesProd.pth "
            "or FaceBoxes/weights/FaceBoxesProd.onnx inside the external 3DDFA_V2 repo."
        )
    return "3DDFA_V2 repo looks ready for the Task8 wrapper."


def ensure_3ddfa_repo(cfg: Dict[str, Any]) -> Dict[str, Any]:
    info = inspect_3ddfa_repo(cfg)
    if not info["ready"]:
        raise FileNotFoundError(info["hint"])
    return info


def choose_backend(cfg: Dict[str, Any], requested: str, repo_info: Dict[str, Any]) -> Tuple[str, str]:
    requested = requested.lower()
    has_onnx = bool(repo_info.get("preferred_onnx_exists"))
    has_pth = bool(repo_info.get("preferred_checkpoint_exists"))
    has_onnxruntime = module_available("onnxruntime")
    if requested == "onnx":
        return "onnx", "requested explicitly"
    if requested == "pth":
        return "pth", "requested explicitly"
    if has_onnx and has_onnxruntime:
        return "onnx", "auto selected ONNX because weights/mb1_120x120.onnx and onnxruntime are available"
    if has_pth:
        reason = "auto selected PyTorch pth"
        if not has_onnx:
            reason += " because weights/mb1_120x120.onnx is unavailable"
        elif not has_onnxruntime:
            reason += " because onnxruntime is unavailable"
        return "pth", reason
    return "onnx", "auto selected ONNX as a last attempt, but required files may be missing"


def choose_mode(cfg: Dict[str, Any], requested: str, backend: str) -> Tuple[str, str]:
    requested = requested.lower()
    if requested in ("gpu", "cpu"):
        return requested, "requested explicitly"
    configured = str(cfg_get(cfg, "reconstruction", "mode", "gpu")).lower()
    if configured in ("gpu", "cpu"):
        return configured, "selected from config"
    if backend == "onnx":
        return "cpu", "ONNX demo path does not use the PyTorch gpu flag"
    return "gpu", "default for AutoDL A800"


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
                next(p for p in candidate.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTS)
                return candidate
            except StopIteration:
                pass
    raise FileNotFoundError("Could not find CelebA image files under {}".format(root))


def list_images(input_dir: Any, recursive: bool = False) -> List[Path]:
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Image directory does not exist: {}".format(root))
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(p for p in iterator if p.is_file() and p.suffix in IMAGE_EXTS)
    if not paths and not recursive:
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTS)
    return paths


def read_image_list(path: Any) -> List[Path]:
    list_path = Path(path).expanduser().resolve()
    if not list_path.exists():
        raise FileNotFoundError("Image list does not exist: {}".format(list_path))
    base = list_path.parent
    images = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.exists() and candidate.suffix in IMAGE_EXTS:
            images.append(candidate.resolve())
    if not images:
        raise ValueError("No valid images found in image list: {}".format(list_path))
    return images


def select_samples(paths: List[Path], count: int, seed: int, strategy: str = "random") -> List[Path]:
    if count <= 0:
        raise ValueError("sample_count must be positive")
    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise ValueError("No input images are available for Task8")
    strategy = strategy.lower()
    if strategy in ("first", "sequential"):
        return unique[: min(count, len(unique))]
    if strategy != "random":
        raise ValueError("Unsupported sample_strategy: {}".format(strategy))
    rng = random.Random(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def safe_stem(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    clean = clean.strip("._-")
    return clean or "image"


def copy_or_symlink(source: Path, target: Path, mode: str = "symlink", force: bool = False) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if force:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(str(target))
            else:
                target.unlink()
        else:
            try:
                if target.resolve() == source.resolve():
                    return "existing"
            except Exception:
                pass
            raise FileExistsError(
                "{} already exists and does not point to {}. Pass --clear-existing or --force.".format(target, source)
            )
    if mode == "copy":
        shutil.copy2(str(source), str(target))
        return "copy"
    os.symlink(str(source), str(target), target_is_directory=source.is_dir())
    return "symlink"


def official_results_dir(repo: Path) -> Path:
    return repo / "examples" / "results"


def file_snapshot(directory: Path, basename: str) -> Dict[str, Dict[str, Any]]:
    if not directory.exists():
        return {}
    files = []
    patterns = ["{}*".format(basename), "*{}*".format(basename)]
    seen = set()
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file() and path not in seen:
                files.append(path)
                seen.add(path)
    snapshot: Dict[str, Dict[str, Any]] = {}
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[str(path)] = {
            "mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000)),
            "size": stat.st_size,
            "path": str(path),
        }
    return snapshot


def changed_files(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> List[Path]:
    changed = []
    for key, meta in after.items():
        old = before.get(key)
        if old is None or old.get("mtime_ns") != meta.get("mtime_ns") or old.get("size") != meta.get("size"):
            changed.append(Path(key))
    return sorted(changed, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def scan_official_outputs(results_dir: Path, basename: str, opt: str, before: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Path]:
    before = before or {}
    after = file_snapshot(results_dir, basename)
    candidates = changed_files(before, after)
    if not candidates:
        candidates = [Path(meta["path"]) for meta in after.values()]
    ext = { "obj": ".obj", "ply": ".ply" }.get(opt, ".jpg")
    filtered = [
        p for p in candidates
        if p.suffix.lower() == ext and (("_" + opt) in p.stem or opt in p.stem or opt == "obj")
    ]
    if not filtered and opt == "3d":
        filtered = [p for p in candidates if p.suffix.lower() == ".jpg" and "_3d" in p.stem]
    return sorted(filtered, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def truncate_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def relpath_for_markdown(path_value: Any, markdown_file: Path) -> str:
    if not path_value:
        return ""
    path = resolve_existing_stage3_path(path_value)
    if not path.is_absolute():
        path = stage3_root() / path
    try:
        return Path(os.path.relpath(str(path), str(markdown_file.parent))).as_posix()
    except Exception:
        return str(path_value)


def resolve_existing_stage3_path(path_value: Any) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() and not path.exists():
        parts = path.parts
        if "stage-3" in parts:
            idx = parts.index("stage-3")
            return stage3_root().joinpath(*parts[idx + 1 :])
    return path


def save_image_grid(image_paths: List[Path], labels: List[str], output_path: Path, thumb_size: int = 180) -> Optional[Path]:
    if not image_paths:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    cols = min(4, max(1, len(image_paths)))
    rows = (len(image_paths) + cols - 1) // cols
    label_h = 22
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
        draw.text((x + 4, y + 4), labels[idx][:28], fill=(20, 20, 20), font=font)
        offset_x = x + (thumb_size - image.width) // 2
        offset_y = y + label_h + (thumb_size - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return output_path


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def python_executable() -> str:
    return sys.executable or "python"
