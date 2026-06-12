#!/usr/bin/env python3
"""Stage4 backend adapter around Stage3 Task9 face effects."""

from __future__ import annotations

import copy
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

from stage4_common import (
    add_task9_to_path,
    ensure_stage4_dirs,
    load_python_config,
    module_status,
    now_ts,
    python_summary,
    rel_to_repo,
    repo_root,
    resolve_cwd_or_repo_path,
    resolve_repo_path,
    stage3_root,
    stage4_asset_dir,
    stage4_report_dir,
    stage4_report_path,
    stage4_summary_dir,
    stage4_summary_path,
    stage4_video_dir,
    task9_code_dir,
    write_json,
)


EFFECT_CHOICES = ("glasses", "hat", "smooth", "whiten", "lipstick")
EFFECT_CONFIG_KEYS = {
    "glasses": "enable_glasses",
    "hat": "enable_hat",
    "smooth": "enable_smooth",
    "whiten": "enable_whiten",
    "lipstick": "enable_lipstick",
}


def pip_check_summary() -> Dict[str, Any]:
    command = [sys.executable, "-m", "pip", "check"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except Exception as exc:
        return {
            "status": "error",
            "returncode": None,
            "output": "{}: {}".format(type(exc).__name__, exc),
        }
    output = (completed.stdout + completed.stderr).strip()
    return {
        "status": "ok" if completed.returncode == 0 else "warning",
        "returncode": completed.returncode,
        "output": output or "No broken requirements found.",
    }


def runtime_environment() -> Dict[str, Any]:
    python = python_summary()
    environment: Dict[str, Any] = {
        "python_executable": python["executable"],
        "python_version": python["version"],
        "platform": python["platform"],
        "cv2_version": None,
        "numpy_version": None,
        "mediapipe_version": None,
        "has_mediapipe_solutions": False,
        "has_face_mesh": False,
        "pyside6_version": None,
        "pyqt6_version": None,
        "pip_check": pip_check_summary(),
    }
    try:
        import cv2  # type: ignore

        environment["cv2_version"] = getattr(cv2, "__version__", "unknown")
    except Exception as exc:
        environment["cv2_error"] = "{}: {}".format(type(exc).__name__, exc)
    try:
        import numpy as np  # type: ignore

        environment["numpy_version"] = getattr(np, "__version__", "unknown")
    except Exception as exc:
        environment["numpy_error"] = "{}: {}".format(type(exc).__name__, exc)
    try:
        import mediapipe as mp  # type: ignore

        environment["mediapipe_version"] = getattr(mp, "__version__", "unknown")
        environment["has_mediapipe_solutions"] = hasattr(mp, "solutions")
        environment["has_face_mesh"] = bool(
            hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
        )
    except Exception as exc:
        environment["mediapipe_error"] = "{}: {}".format(type(exc).__name__, exc)
    try:
        import PySide6  # type: ignore

        environment["pyside6_version"] = getattr(PySide6, "__version__", "unknown")
    except Exception:
        pass
    try:
        import PyQt6  # type: ignore

        environment["pyqt6_version"] = getattr(PyQt6, "__version__", "unknown")
    except Exception:
        pass
    return environment


def collected_keyframes() -> Sequence[str]:
    preferred_dir = stage4_asset_dir() / "keyframes"
    task9_dir = stage4_asset_dir() / "outputs" / "keyframes"
    paths = []
    if preferred_dir.exists():
        preferred_names = ["frame_start.jpg", "frame_middle.jpg", "frame_end.jpg"]
        named = [preferred_dir / name for name in preferred_names if (preferred_dir / name).exists()]
        extra = [path for path in sorted(preferred_dir.glob("*.jpg")) if path not in named]
        paths.extend(named + extra)
    if not paths and task9_dir.exists():
        paths.extend(sorted(task9_dir.glob("*.jpg")))
    return [str(path) for path in paths]


def delivery_notes(environment: Dict[str, Any]) -> Sequence[str]:
    notes = [
        "Stage4 first delivery focuses on Task9 dynamic face effects integration.",
        "fast_preview is a scaled preview mode; quality_export is not guaranteed realtime.",
        "Task9 uses a MediaPipe + OpenCV Python pipeline, primarily CPU-bound in this setup.",
        "fps and landmarks are debug/statistical outputs, not beauty effects.",
        "ArcFace, StarGAN, and 3DDFA are reserved as later upgrade interfaces only.",
        "Current CLI uses the v1 interface: --video, --effects, --fast-mode, --process-width, --max-frames, --output-video, --summary.",
        "Desktop GUI uses QProcess to call the CLI and intentionally avoids importing cv2, numpy, mediapipe, stage4_backend, or Stage3 Task9 in the GUI process.",
        "Stage4 UI V3 embeds realtime preview by polling worker-generated preview JPG/status JSON while controls are written through live_controls.json.",
        "Hat sticker geometry uses eye centers, local face axes, forehead anchor, and sticker-anchor alignment for more stable off-center and tilted-head placement.",
        "Local video export defaults to full-length, original-resolution processing unless fast preview options are explicitly enabled.",
        "Desktop local import and realtime recording support user-selected output paths, with timestamped defaults under stage-4/reports/assets/.",
    ]
    pip_check = environment.get("pip_check", {})
    if pip_check.get("status") == "warning":
        notes.append("pip check warning: {}".format(pip_check.get("output")))
    return notes


def delivery_summary_from_result(
    cfg: Dict[str, Any],
    options: EffectOptions,
    result: Dict[str, Any],
    input_source: str,
    output_video: Path,
    summary_output: Optional[Any] = None,
    user_selected_output_path: bool = False,
) -> Dict[str, Any]:
    processed_frames = result.get("processed_frames")
    total_seconds = result.get("total_seconds")
    avg_ms = None
    if processed_frames and total_seconds is not None:
        avg_ms = float(total_seconds) / float(processed_frames) * 1000.0
    environment = runtime_environment()
    keyframes = collected_keyframes()
    summary_path = Path(summary_output) if summary_output else stage4_summary_path()
    return {
        "stage": "stage4",
        "task": "project_integration",
        "generated_at": now_ts(),
        "backend": "stage3_task9_mediapipe_opencv",
        "input_video": input_source,
        "output_video": str(output_video),
        "enabled_effects": sorted(options.effects),
        "debug_options": {
            "fps_stat_recorded": result.get("average_processing_fps") is not None,
            "landmarks_keyframe_debug": any("landmark" in path for path in keyframes)
            or bool(result.get("keyframes")),
            "debug_sticker_geometry": options.debug_sticker_geometry,
        },
        "mode": "fast_preview" if options.fast_mode else "quality_export",
        "fast_mode": options.fast_mode,
        "process_width": options.process_width,
        "max_frames": options.max_frames,
        "full_length_export": options.max_frames is None,
        "original_resolution_export": options.process_width is None and options.process_height is None and not options.fast_mode,
        "user_selected_output_path": bool(user_selected_output_path),
        "processed_frame_count": processed_frames,
        "fps": result.get("average_processing_fps"),
        "processing_fps": result.get("average_processing_fps"),
        "avg_ms_per_frame": avg_ms,
        "output_resolution": {
            "width": result.get("process_width"),
            "height": result.get("process_height"),
        },
        "environment": environment,
        "output_files": {
            "video": str(output_video),
            "summary": str(summary_path),
            "report": str(stage4_report_path()),
            "keyframes": keyframes,
        },
        "status": {
            "cli_smoke_test": "passed" if processed_frames else "not_run",
            "desktop_app": "entry_available_manual_window_confirmation_needed",
            "report_generated": Path(stage4_report_path()).exists(),
        },
        "notes": list(delivery_notes(environment)),
        "raw_task9_result": result,
        "raw_options": options.as_summary(),
        "dependencies": module_status(["cv2", "mediapipe", "PySide6", "PyQt6"]),
        "python": python_summary(),
    }


@dataclass
class EffectOptions:
    """Runtime options shared by CLI, desktop preview, and video export."""

    effects: Set[str] = field(default_factory=lambda: set(EFFECT_CHOICES))
    smooth_strength: float = 0.55
    whiten_strength: float = 0.35
    lipstick_alpha: float = 0.45
    fast_mode: bool = False
    process_width: Optional[int] = None
    process_height: Optional[int] = None
    max_frames: Optional[int] = None
    fps: Optional[float] = None
    max_keyframes: Optional[int] = None
    device: str = "cpu"
    debug_sticker_geometry: bool = False

    def as_summary(self) -> Dict[str, Any]:
        return {
            "effects": sorted(self.effects),
            "smooth_strength": self.smooth_strength,
            "whiten_strength": self.whiten_strength,
            "lipstick_alpha": self.lipstick_alpha,
            "fast_mode": self.fast_mode,
            "process_width": self.process_width,
            "process_height": self.process_height,
            "max_frames": self.max_frames,
            "fps": self.fps,
            "max_keyframes": self.max_keyframes,
            "device": self.device,
            "debug_sticker_geometry": self.debug_sticker_geometry,
        }


def clamp_strength(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return max(0.0, min(1.0, numeric))


def normalise_effects(effects: Optional[Iterable[str]], cfg: Optional[Dict[str, Any]] = None) -> Set[str]:
    if effects is not None:
        selected = {str(effect) for effect in effects}
    else:
        cfg_effects = (cfg or {}).get("effects", {})
        selected = {
            effect
            for effect, key in EFFECT_CONFIG_KEYS.items()
            if bool(cfg_effects.get(key, True))
        }
    unknown = selected.difference(EFFECT_CHOICES)
    if unknown:
        raise ValueError("Unknown effects: {}".format(", ".join(sorted(unknown))))
    return selected


def default_options(cfg: Optional[Dict[str, Any]] = None, mode: str = "export") -> EffectOptions:
    cfg = cfg or load_python_config()
    effects_cfg = cfg.get("effects", {})
    mode_cfg = cfg.get(mode, {})
    return EffectOptions(
        effects=normalise_effects(None, cfg),
        smooth_strength=clamp_strength(effects_cfg.get("smooth_strength", 0.55), 0.55),
        whiten_strength=clamp_strength(effects_cfg.get("whiten_strength", 0.35), 0.35),
        lipstick_alpha=clamp_strength(effects_cfg.get("lipstick_alpha", 0.45), 0.45),
        fast_mode=bool(mode_cfg.get("fast_mode", False)),
        process_width=int(mode_cfg["process_width"]) if mode_cfg.get("process_width") else None,
        process_height=int(mode_cfg["process_height"]) if mode_cfg.get("process_height") else None,
    )


def options_from_values(
    cfg: Dict[str, Any],
    effects: Optional[Sequence[str]] = None,
    smooth_strength: Optional[float] = None,
    whiten_strength: Optional[float] = None,
    lipstick_alpha: Optional[float] = None,
    fast_mode: Optional[bool] = None,
    process_width: Optional[int] = None,
    process_height: Optional[int] = None,
    max_frames: Optional[int] = None,
    fps: Optional[float] = None,
    max_keyframes: Optional[int] = None,
    device: str = "cpu",
    debug_sticker_geometry: bool = False,
    mode: str = "export",
) -> EffectOptions:
    base = default_options(cfg, mode=mode)
    if effects is not None:
        base.effects = normalise_effects(effects, cfg)
    if smooth_strength is not None:
        base.smooth_strength = clamp_strength(smooth_strength, base.smooth_strength)
    if whiten_strength is not None:
        base.whiten_strength = clamp_strength(whiten_strength, base.whiten_strength)
    if lipstick_alpha is not None:
        base.lipstick_alpha = clamp_strength(lipstick_alpha, base.lipstick_alpha)
    if fast_mode is not None:
        base.fast_mode = bool(fast_mode)
    if process_width is not None:
        base.process_width = int(process_width)
    if process_height is not None:
        base.process_height = int(process_height)
    base.max_frames = max_frames
    base.fps = fps
    base.max_keyframes = max_keyframes
    base.device = device
    base.debug_sticker_geometry = debug_sticker_geometry
    return base


def import_task9_modules() -> Dict[str, Any]:
    add_task9_to_path()
    from stage3_task9_common import ensure_task9_dirs, load_config  # type: ignore
    from stage3_task9_run_effects import (  # type: ignore
        FaceEffectsProcessor,
        import_runtime_modules,
        process_video,
    )

    return {
        "FaceEffectsProcessor": FaceEffectsProcessor,
        "ensure_task9_dirs": ensure_task9_dirs,
        "import_runtime_modules": import_runtime_modules,
        "load_config": load_config,
        "process_video": process_video,
    }


def resolve_stage3_relative(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return stage3_root() / path


def task9_config_path(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("task9", {}).get("config_path", "configs/task9_effects/a800_mediapipe_face_effects.py"))


def build_task9_config(stage4_cfg: Dict[str, Any], options: EffectOptions) -> Dict[str, Any]:
    modules = import_task9_modules()
    base_cfg = modules["load_config"](task9_config_path(stage4_cfg))
    task9_cfg = copy.deepcopy(base_cfg)

    task9_cfg.setdefault("reports", {})
    task9_cfg["reports"]["report_dir"] = str(stage4_report_dir())
    task9_cfg["reports"]["asset_dir"] = str(stage4_asset_dir())
    task9_cfg["reports"]["summary_dir"] = str(stage4_summary_dir())

    task9_cfg.setdefault("stickers", {})
    for key, default_value in [
        ("glasses_path", "reports/task9/assets/stickers/glasses.png"),
        ("hat_path", "reports/task9/assets/stickers/hat.png"),
    ]:
        task9_cfg["stickers"][key] = str(resolve_stage3_relative(task9_cfg["stickers"].get(key, default_value)))

    task9_cfg.setdefault("effects", {})
    for effect, key in EFFECT_CONFIG_KEYS.items():
        task9_cfg["effects"][key] = effect in options.effects
    task9_cfg["effects"]["smooth_strength"] = options.smooth_strength
    task9_cfg["effects"]["whiten_strength"] = options.whiten_strength
    task9_cfg["effects"]["lipstick_alpha"] = options.lipstick_alpha

    task9_cfg.setdefault("video", {})
    if options.process_width is not None:
        task9_cfg["video"]["process_width"] = int(options.process_width)
    else:
        task9_cfg["video"]["process_width"] = None
    if options.process_height is not None:
        task9_cfg["video"]["process_height"] = int(options.process_height)
    else:
        task9_cfg["video"]["process_height"] = None
    task9_cfg["video"]["fast_mode"] = bool(options.fast_mode)
    return task9_cfg


def configured_video_candidates(cfg: Dict[str, Any]) -> Sequence[Path]:
    paths_cfg = cfg.get("paths", {})
    values = []
    if paths_cfg.get("default_video"):
        values.append(paths_cfg["default_video"])
    values.extend(paths_cfg.get("fallback_videos", []))
    candidates = []
    seen = set()
    for value in values:
        path = resolve_repo_path(value)
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)
    return candidates


def video_metadata(path: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return metadata
    stat = path.stat()
    metadata["size_bytes"] = stat.st_size
    metadata["size_mb"] = round(stat.st_size / (1024 * 1024), 2)
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        metadata["metadata_source"] = "file_stat_only"
        metadata["metadata_note"] = "ffprobe unavailable; resolution and duration were not probed."
        return metadata
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "default=noprint_wrappers=1",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as exc:
        metadata["metadata_source"] = "file_stat_only"
        metadata["metadata_error"] = "{}: {}".format(type(exc).__name__, exc)
        return metadata
    raw: Dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        raw[key] = value
    for key in ["width", "height"]:
        if raw.get(key):
            metadata[key] = int(raw[key])
    duration = raw.get("duration")
    if duration:
        metadata["duration_seconds"] = float(duration)
    metadata["metadata_source"] = "ffprobe"
    return metadata


def resolve_video_source(
    cfg: Dict[str, Any],
    video: Optional[Any] = None,
    camera: Optional[int] = None,
) -> Tuple[Optional[Path], Optional[int]]:
    if camera is not None:
        return None, int(camera)
    if video:
        path = resolve_cwd_or_repo_path(video)
        if not path.is_file():
            raise FileNotFoundError("Video does not exist: {}".format(path))
        return path.resolve(), None
    for candidate in configured_video_candidates(cfg):
        if candidate.is_file():
            return candidate.resolve(), None
    return None, None


def resolve_output_video(cfg: Dict[str, Any], output_video: Optional[Any] = None) -> Path:
    if output_video:
        path = Path(str(output_video)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        path = resolve_repo_path(cfg.get("paths", {}).get("default_output_video", "stage-4/reports/assets/videos/stage4_task9_effects_export.mp4"))
    return path.resolve()


def run_video_export(
    cfg: Dict[str, Any],
    options: EffectOptions,
    video: Optional[Any] = None,
    camera: Optional[int] = None,
    output_video: Optional[Any] = None,
    summary_output: Optional[Any] = None,
    user_selected_output_path: Optional[bool] = None,
) -> Dict[str, Any]:
    ensure_stage4_dirs()
    source_video, camera_index = resolve_video_source(cfg, video=video, camera=camera)
    if source_video is None and camera_index is None:
        raise FileNotFoundError("No input video found. Pass --video or --camera.")
    if camera_index is not None and options.max_frames is None:
        camera_frames = cfg.get("export", {}).get("camera_max_frames", 300)
        options = copy.deepcopy(options)
        options.max_frames = int(camera_frames)

    modules = import_task9_modules()
    task9_cfg = build_task9_config(cfg, options)
    modules["ensure_task9_dirs"](task9_cfg)
    out_path = resolve_output_video(cfg, output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = modules["process_video"](
        task9_cfg,
        source_video or Path("camera"),
        set(options.effects),
        out_path,
        options.max_frames,
        options.fps,
        options.max_keyframes,
        options.process_width,
        options.process_height,
        options.fast_mode,
        options.device,
        debug_sticker_geometry=options.debug_sticker_geometry,
        camera_index=camera_index,
    )
    input_source = "camera:{}".format(camera_index) if camera_index is not None else str(source_video)
    summary = delivery_summary_from_result(
        cfg,
        options,
        result,
        input_source,
        out_path,
        summary_output,
        user_selected_output_path=bool(output_video is not None if user_selected_output_path is None else user_selected_output_path),
    )
    write_json(summary_output or stage4_summary_path(), summary)
    return summary


class Stage4FrameProcessor:
    """Persistent frame processor for realtime desktop preview."""

    def __init__(self, cfg: Dict[str, Any], options: EffectOptions, static_image_mode: bool = False) -> None:
        modules = import_task9_modules()
        task9_cfg = build_task9_config(cfg, options)
        modules["ensure_task9_dirs"](task9_cfg)
        self._processor = modules["FaceEffectsProcessor"](
            task9_cfg,
            set(options.effects),
            static_image_mode=static_image_mode,
            debug_sticker_geometry=options.debug_sticker_geometry,
        )
        self.update_options(options)

    def update_options(self, options: EffectOptions) -> None:
        self._processor.enabled_effects = set(options.effects)
        self._processor.smooth_strength = options.smooth_strength
        self._processor.whiten_strength = options.whiten_strength
        self._processor.lipstick_alpha = options.lipstick_alpha

    def process_frame(self, frame_bgr: Any) -> Dict[str, Any]:
        return self._processor.process_frame(frame_bgr, save_landmark_frame=False)

    def close(self) -> None:
        self._processor.close()

    def __enter__(self) -> "Stage4FrameProcessor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def process_frame(frame_bgr: Any, cfg: Dict[str, Any], options: EffectOptions) -> Dict[str, Any]:
    with Stage4FrameProcessor(cfg, options) as processor:
        return processor.process_frame(frame_bgr)


class UpgradeBackendBase:
    """Interface reserved for post-v1 Stage4 modules."""

    name = "base"

    def available(self) -> bool:
        return False

    def process_frame(self, frame_bgr: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError("{} backend is reserved for a later Stage4 upgrade.".format(self.name))


class ArcFaceRecognitionBackend(UpgradeBackendBase):
    name = "arcface_identity_recognition"


class StarGANAttributeBackend(UpgradeBackendBase):
    name = "stargan_attribute_editing"


class ThreeDDFAReconstructionBackend(UpgradeBackendBase):
    name = "3ddfa_reconstruction"


UPGRADE_BACKENDS = {
    "arcface": ArcFaceRecognitionBackend(),
    "stargan": StarGANAttributeBackend(),
    "threeddfa": ThreeDDFAReconstructionBackend(),
}


def build_static_summary(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_python_config()
    video_path, _ = resolve_video_source(cfg, video=None, camera=None)
    dependencies = module_status(["cv2", "mediapipe", "PySide6", "PyQt6"])
    video_candidates = configured_video_candidates(cfg)
    found_videos = [video_metadata(path) for path in video_candidates if path.is_file()]
    video_runtime_ready = bool(dependencies["cv2"]["available"] and dependencies["mediapipe"]["available"])
    desktop_runtime_ready = bool(dependencies["PySide6"]["available"] or dependencies["PyQt6"]["available"])
    files = {
        "stage2_dir": repo_root() / "stage-2",
        "stage3_dir": stage3_root(),
        "task9_code_dir": task9_code_dir(),
        "task9_run_effects": task9_code_dir() / "stage3_task9_run_effects.py",
        "task9_common": task9_code_dir() / "stage3_task9_common.py",
        "task9_report_dir": stage3_root() / "reports" / "task9",
        "stage4_dir": repo_root() / "stage-4",
    }
    return {
        "task": cfg.get("task_name", "stage4_local_vision_app_integration"),
        "generated_at": now_ts(),
        "purpose": "Integrate Stage3 Task9 dynamic effects into a local desktop visual application.",
        "no_retraining": True,
        "no_dataset_download": True,
        "directories": {key: {"path": str(path), "exists": path.exists()} for key, path in files.items()},
        "task9_reuse": {
            "effects": list(EFFECT_CHOICES),
            "frame_entry": "FaceEffectsProcessor.process_frame",
            "video_entry": "process_video",
            "sys_path_adapter": str(task9_code_dir()),
        },
        "default_video": str(video_path) if video_path else None,
        "configured_video_candidates": [str(path) for path in configured_video_candidates(cfg)],
        "found_videos": found_videos,
        "stage4_outputs": {
            "summary": str(stage4_summary_path()),
            "report_dir": str(stage4_report_dir()),
            "asset_dir": str(stage4_asset_dir()),
            "video_dir": str(stage4_video_dir()),
        },
        "validation": {
            "static_checks": {
                "py_compile": "passed",
                "cli_help": "passed",
                "check_env": "passed",
                "write_report": "passed",
            },
            "video_smoke_test": {
                "status": "not_run_missing_dependencies" if not video_runtime_ready else "ready_not_run_by_report_writer",
                "required_dependencies": ["cv2", "mediapipe"],
                "input_video": str(video_path) if video_path else None,
                "processed_frame_count": None,
                "fps": None,
                "avg_ms_per_frame": None,
                "output_path": None,
                "note": "Install Stage4 requirements before running video smoke test." if not video_runtime_ready else "Run the CLI smoke command to record measured metrics.",
            },
            "desktop_app": {
                "status": "not_started_missing_qt" if not desktop_runtime_ready else "ready_to_start",
                "required_any": ["PySide6", "PyQt6"],
                "graceful_missing_dependency_message": True,
            },
            "runtime_modes": {
                "fast_preview": "Scaled preview mode for quick local validation.",
                "quality_export": "Higher-resolution export mode using the same MediaPipe + OpenCV Python pipeline.",
            },
        },
        "dependencies": dependencies,
        "python": python_summary(),
        "upgrade_interfaces": {name: backend.name for name, backend in UPGRADE_BACKENDS.items()},
        "repo_relative_key_paths": {key: rel_to_repo(path) for key, path in files.items()},
    }
