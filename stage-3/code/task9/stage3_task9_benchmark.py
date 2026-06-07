#!/usr/bin/env python3
"""Benchmark Stage3 Task9 MediaPipe + OpenCV face effects."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task9_effects/a800_mediapipe_face_effects.py")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--effects", nargs="+", choices=["glasses", "hat", "smooth", "whiten", "lipstick"], default=None)
    parser.add_argument("--max-frames", type=int, default=None)
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


def benchmark_video(cfg: Dict[str, Any], video_path: Path, effects: Set[str], max_frames: Optional[int]) -> Dict[str, Any]:
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
    detection_seconds = 0.0
    render_seconds = 0.0
    write_seconds = 0.0
    started = time.perf_counter()
    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_frame_to_config(cv2, cfg, frame)
            result = processor.process_frame(frame, save_landmark_frame=False)
            if writer is None:
                h, w = result["frame"].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(tmp_output), fourcc, output_fps, (w, h))
                if not writer.isOpened():
                    raise IOError("Could not open temporary benchmark writer: {}".format(tmp_output))
            detection_seconds += result["detection_seconds"]
            render_seconds += result["render_seconds"]
            write_started = time.perf_counter()
            writer.write(result["frame"])
            write_seconds += time.perf_counter() - write_started
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
    return {
        "benchmark_type": "video",
        "input_video": str(video_path),
        "temporary_output_video_deleted": str(tmp_output),
        "source_total_frames": total_frames,
        "source_fps": source_fps,
        "output_fps_for_writer": output_fps,
        "processed_frames": frame_count,
        "faces_detected_frames": faces_detected,
        "total_seconds": elapsed,
        "average_fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "average_detection_ms": (detection_seconds / frame_count * 1000.0) if frame_count else None,
        "average_render_ms": (render_seconds / frame_count * 1000.0) if frame_count else None,
        "average_write_ms": (write_seconds / frame_count * 1000.0) if frame_count else None,
        "detection_seconds": detection_seconds,
        "render_seconds": render_seconds,
        "write_seconds": write_seconds,
        "note": "Benchmark writes processed frames to a temporary mp4 and deletes it after timing.",
    }


def benchmark_image_batch(cfg: Dict[str, Any], input_dir: Path, effects: Set[str]) -> Dict[str, Any]:
    image_paths = list_images(input_dir, recursive=False)
    processor = FaceEffectsProcessor(cfg, effects, static_image_mode=True)
    cv2 = processor.cv2
    processed = 0
    faces_detected = 0
    detection_seconds = 0.0
    render_seconds = 0.0
    records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                records.append({"input_path": str(image_path), "success": False, "error": "cv2.imread returned None"})
                continue
            result = processor.process_frame(frame, save_landmark_frame=False)
            detection_seconds += result["detection_seconds"]
            render_seconds += result["render_seconds"]
            faces_detected += 1 if result["face_detected"] else 0
            processed += 1
            records.append({"input_path": str(image_path), "success": True, "face_detected": result["face_detected"]})
    finally:
        processor.close()
    elapsed = time.perf_counter() - started
    return {
        "benchmark_type": "image_batch",
        "input_dir": str(input_dir),
        "processed_images": processed,
        "faces_detected_images": faces_detected,
        "total_seconds": elapsed,
        "images_per_second": processed / elapsed if elapsed > 0 else 0.0,
        "average_detection_ms": (detection_seconds / processed * 1000.0) if processed else None,
        "average_render_ms": (render_seconds / processed * 1000.0) if processed else None,
        "detection_seconds": detection_seconds,
        "render_seconds": render_seconds,
        "write_seconds": 0.0,
        "records": records,
        "note": "No user mp4 was provided, so this is an image batch throughput benchmark, not video FPS.",
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_task9_dirs(cfg)
    effects = enabled_effects_from_args(args, cfg)
    video_path = locate_user_video(cfg, args.video)
    if video_path:
        result = benchmark_video(cfg, video_path, effects, args.max_frames)
    else:
        print(NO_VIDEO_HINT)
        input_dir = args.input_dir or static_images_dir(cfg)
        result = benchmark_image_batch(cfg, input_dir, effects)
    payload = {
        "task": cfg.get("task_name"),
        "effects": sorted(effects),
        "cpu_gpu_note": "This experiment mainly uses MediaPipe + OpenCV and is primarily CPU-executed in the standard Python pipeline. A800 is recorded if visible but is not required for Task9.",
        "gpu": torch_cuda_info(),
    }
    payload.update(result)
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    write_json(performance_summary_path(cfg), payload)


if __name__ == "__main__":
    main()
