#!/usr/bin/env python3
"""Benchmark Stage3 Task9 MediaPipe + OpenCV face effects."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from stage3_task9_common import (
    NO_VIDEO_HINT,
    cfg_get,
    ensure_task9_dirs,
    list_images,
    load_config,
    locate_user_video,
    performance_summary_path,
    static_images_dir,
    summary_dir,
    write_json,
)
from stage3_task9_run_effects import (
    FaceEffectsProcessor,
    enabled_effects_from_args,
    import_runtime_modules,
    resize_frame_to_config,
)


PROFILE_CHOICES = ["landmark_only", "stickers_only", "beauty_only", "lipstick_only", "full_effects"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task9_effects/a800_mediapipe_face_effects.py")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--effects", nargs="+", choices=["glasses", "hat", "smooth", "whiten", "lipstick"], default=None)
    parser.add_argument("--profiles", nargs="+", choices=PROFILE_CHOICES, default=["full_effects"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--process-width", type=int, default=None)
    parser.add_argument("--process-height", type=int, default=None)
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def torch_cuda_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "pipeline_note": "Task9 uses MediaPipe Face Mesh + OpenCV. This benchmark treats the pipeline as CPU-bound unless a custom GPU MediaPipe/OpenCV build is used.",
        "torch_available": False,
        "cuda_available": False,
    }
    try:
        import torch  # type: ignore

        info["torch_available"] = True
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
    except Exception as exc:
        info["torch_error"] = "{}: {}".format(type(exc).__name__, str(exc))
    return info


def effects_for_profile(profile: str, cfg: Dict[str, Any], requested_effects: Set[str]) -> Set[str]:
    if profile == "landmark_only":
        return set()
    if profile == "stickers_only":
        return {"glasses", "hat"}
    if profile == "beauty_only":
        return {"smooth", "whiten"}
    if profile == "lipstick_only":
        return {"lipstick"}
    if requested_effects:
        return set(requested_effects)
    return {"glasses", "hat", "smooth", "whiten", "lipstick"}


def empty_profile_totals() -> Dict[str, float]:
    return {
        "detection_seconds": 0.0,
        "sticker_seconds": 0.0,
        "beauty_seconds": 0.0,
        "lipstick_seconds": 0.0,
        "render_seconds": 0.0,
        "write_seconds": 0.0,
    }


def profile_metrics(
    profile: str,
    effects: Set[str],
    frame_count: int,
    faces_detected: int,
    elapsed: float,
    totals: Dict[str, float],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def avg_ms(key: str) -> Optional[float]:
        return totals[key] / frame_count * 1000.0 if frame_count else None

    payload: Dict[str, Any] = {
        "profile": profile,
        "effects": sorted(effects),
        "total_frames": frame_count,
        "faces_detected_frames": faces_detected,
        "total_seconds": elapsed,
        "fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "detection_ms_per_frame": avg_ms("detection_seconds"),
        "sticker_ms_per_frame": avg_ms("sticker_seconds"),
        "beauty_ms_per_frame": avg_ms("beauty_seconds"),
        "lipstick_ms_per_frame": avg_ms("lipstick_seconds"),
        "render_ms_per_frame": avg_ms("render_seconds"),
        "write_ms_per_frame": avg_ms("write_seconds"),
        "total_ms_per_frame": elapsed / frame_count * 1000.0 if frame_count else None,
        "detection_seconds": totals["detection_seconds"],
        "sticker_seconds": totals["sticker_seconds"],
        "beauty_seconds": totals["beauty_seconds"],
        "lipstick_seconds": totals["lipstick_seconds"],
        "render_seconds": totals["render_seconds"],
        "write_seconds": totals["write_seconds"],
        "cpu_or_gpu_note": "CPU MediaPipe/OpenCV pipeline. CUDA is recorded only as environment info unless a custom GPU path is implemented.",
    }
    if extra:
        payload.update(extra)
    return payload


def benchmark_video_profile(
    cfg: Dict[str, Any],
    video_path: Path,
    profile: str,
    effects: Set[str],
    max_frames: Optional[int],
    process_width: Optional[int],
    process_height: Optional[int],
    fast_mode: Optional[bool],
    device: str,
) -> Dict[str, Any]:
    cv2, _, _ = import_runtime_modules()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError("Could not open video: {}".format(video_path))
    processor = FaceEffectsProcessor(cfg, effects, static_image_mode=False)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    output_fps = source_fps or float(cfg_get(cfg, "video", "fps", 20) or 20)
    tmp_handle = tempfile.NamedTemporaryFile(prefix="task9_benchmark_", suffix=".mp4", delete=False)
    tmp_output = Path(tmp_handle.name)
    tmp_handle.close()
    writer = None
    frame_count = 0
    faces_detected = 0
    totals = empty_profile_totals()
    process_size = None
    started = time.perf_counter()
    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_frame_to_config(cv2, cfg, frame, process_width, process_height, fast_mode)
            if process_size is None:
                process_size = (int(frame.shape[1]), int(frame.shape[0]))
            result = processor.process_frame(frame, save_landmark_frame=False)
            if writer is None:
                h, w = result["frame"].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(tmp_output), fourcc, output_fps, (w, h))
                if not writer.isOpened():
                    raise IOError("Could not open temporary benchmark writer: {}".format(tmp_output))
            totals["detection_seconds"] += result["detection_seconds"]
            totals["sticker_seconds"] += result.get("sticker_seconds", 0.0)
            totals["beauty_seconds"] += result.get("beauty_seconds", 0.0)
            totals["lipstick_seconds"] += result.get("lipstick_seconds", 0.0)
            totals["render_seconds"] += result["render_seconds"]
            write_started = time.perf_counter()
            writer.write(result["frame"])
            totals["write_seconds"] += time.perf_counter() - write_started
            faces_detected += 1 if result["face_detected"] else 0
            frame_count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        processor.close()
        try:
            tmp_output.unlink()
        except OSError:
            pass
    elapsed = time.perf_counter() - started
    return profile_metrics(
        profile,
        effects,
        frame_count,
        faces_detected,
        elapsed,
        totals,
        {
            "benchmark_type": "video",
            "input_video": str(video_path),
            "source_total_frames": total_frames,
            "source_fps": source_fps,
            "output_fps_for_writer": output_fps,
            "process_width": process_size[0] if process_size else None,
            "process_height": process_size[1] if process_size else None,
            "temporary_output_video_deleted": str(tmp_output),
            "sticker_cache_size": len(processor.sticker_cache),
            "sticker_cache_enabled": processor.sticker_cache_enabled,
            "device_requested": device,
            "device_used": "cpu",
            "device_note": "CUDA requested; benchmark fell back to CPU for standard MediaPipe/OpenCV steps." if device == "cuda" else "CPU benchmark.",
        },
    )


def benchmark_video(
    cfg: Dict[str, Any],
    video_path: Path,
    profiles: Sequence[str],
    requested_effects: Set[str],
    max_frames: Optional[int],
    process_width: Optional[int],
    process_height: Optional[int],
    fast_mode: Optional[bool],
    device: str,
) -> Dict[str, Any]:
    profile_records = []
    for profile in profiles:
        effects = effects_for_profile(profile, cfg, requested_effects)
        profile_records.append(
            benchmark_video_profile(
                cfg,
                video_path,
                profile,
                effects,
                max_frames,
                process_width,
                process_height,
                fast_mode,
                device,
            )
        )
    primary = profile_records[-1] if profile_records else {}
    return {
        "benchmark_type": "video",
        "input_video": str(video_path),
        "profiles": profile_records,
        "profile_names": list(profiles),
        "processed_frames": primary.get("total_frames", 0),
        "faces_detected_frames": primary.get("faces_detected_frames", 0),
        "average_fps": primary.get("fps", 0.0),
        "average_detection_ms": primary.get("detection_ms_per_frame"),
        "average_sticker_ms": primary.get("sticker_ms_per_frame"),
        "average_beauty_ms": primary.get("beauty_ms_per_frame"),
        "average_lipstick_ms": primary.get("lipstick_ms_per_frame"),
        "average_render_ms": primary.get("render_ms_per_frame"),
        "average_write_ms": primary.get("write_ms_per_frame"),
        "total_seconds": primary.get("total_seconds", 0.0),
        "detection_seconds": primary.get("detection_seconds", 0.0),
        "sticker_seconds": primary.get("sticker_seconds", 0.0),
        "beauty_seconds": primary.get("beauty_seconds", 0.0),
        "lipstick_seconds": primary.get("lipstick_seconds", 0.0),
        "render_seconds": primary.get("render_seconds", 0.0),
        "write_seconds": primary.get("write_seconds", 0.0),
        "note": "Each profile writes processed frames to a temporary mp4 and deletes it after timing.",
    }


def benchmark_image_profile(cfg: Dict[str, Any], input_dir: Path, profile: str, effects: Set[str]) -> Dict[str, Any]:
    image_paths = list_images(input_dir, recursive=False)
    processor = FaceEffectsProcessor(cfg, effects, static_image_mode=True)
    cv2 = processor.cv2
    processed = 0
    faces_detected = 0
    totals = empty_profile_totals()
    records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                records.append({"input_path": str(image_path), "success": False, "error": "cv2.imread returned None"})
                continue
            result = processor.process_frame(frame, save_landmark_frame=False)
            totals["detection_seconds"] += result["detection_seconds"]
            totals["sticker_seconds"] += result.get("sticker_seconds", 0.0)
            totals["beauty_seconds"] += result.get("beauty_seconds", 0.0)
            totals["lipstick_seconds"] += result.get("lipstick_seconds", 0.0)
            totals["render_seconds"] += result["render_seconds"]
            faces_detected += 1 if result["face_detected"] else 0
            processed += 1
            records.append({"input_path": str(image_path), "success": True, "face_detected": result["face_detected"]})
    finally:
        processor.close()
    elapsed = time.perf_counter() - started
    profile_data = profile_metrics(profile, effects, processed, faces_detected, elapsed, totals)
    profile_data["records"] = records
    return profile_data


def benchmark_image_batch(cfg: Dict[str, Any], input_dir: Path, profiles: Sequence[str], requested_effects: Set[str]) -> Dict[str, Any]:
    profile_records = []
    for profile in profiles:
        effects = effects_for_profile(profile, cfg, requested_effects)
        profile_records.append(benchmark_image_profile(cfg, input_dir, profile, effects))
    primary = profile_records[-1] if profile_records else {}
    return {
        "benchmark_type": "image_batch",
        "input_dir": str(input_dir),
        "profiles": profile_records,
        "profile_names": list(profiles),
        "processed_images": primary.get("total_frames", 0),
        "faces_detected_images": primary.get("faces_detected_frames", 0),
        "total_seconds": primary.get("total_seconds", 0.0),
        "images_per_second": primary.get("fps", 0.0),
        "average_detection_ms": primary.get("detection_ms_per_frame"),
        "average_sticker_ms": primary.get("sticker_ms_per_frame"),
        "average_beauty_ms": primary.get("beauty_ms_per_frame"),
        "average_lipstick_ms": primary.get("lipstick_ms_per_frame"),
        "average_render_ms": primary.get("render_ms_per_frame"),
        "average_write_ms": 0.0,
        "detection_seconds": primary.get("detection_seconds", 0.0),
        "sticker_seconds": primary.get("sticker_seconds", 0.0),
        "beauty_seconds": primary.get("beauty_seconds", 0.0),
        "lipstick_seconds": primary.get("lipstick_seconds", 0.0),
        "render_seconds": primary.get("render_seconds", 0.0),
        "write_seconds": 0.0,
        "note": "No user mp4 was provided, so this is an image batch throughput benchmark, not video FPS.",
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_task9_dirs(cfg)
    effects = enabled_effects_from_args(args, cfg)
    video_path = locate_user_video(cfg, args.video)
    if video_path:
        result = benchmark_video(
            cfg,
            video_path,
            args.profiles,
            effects,
            args.max_frames,
            args.process_width,
            args.process_height,
            True if args.fast_mode else None,
            args.device,
        )
    else:
        print(NO_VIDEO_HINT)
        input_dir = args.input_dir or static_images_dir(cfg)
        result = benchmark_image_batch(cfg, input_dir, args.profiles, effects)
    payload = {
        "task": cfg.get("task_name"),
        "effects": sorted(effects),
        "profiles_requested": args.profiles,
        "fast_mode_requested": bool(args.fast_mode),
        "process_width_requested": args.process_width,
        "process_height_requested": args.process_height,
        "device_requested": args.device,
        "device_used": "cpu",
        "cpu_gpu_note": "This experiment mainly uses MediaPipe + OpenCV and is primarily CPU-executed in the standard Python pipeline. A800 is recorded if visible but is not required for Task9.",
        "gpu": torch_cuda_info(),
    }
    payload.update(result)
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    write_json(performance_summary_path(cfg), payload)


if __name__ == "__main__":
    main()
