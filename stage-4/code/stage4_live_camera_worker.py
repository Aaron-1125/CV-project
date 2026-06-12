#!/usr/bin/env python3
"""Stage4 realtime camera worker for embedded desktop preview.

This process owns all cv2/mediapipe/Task9 imports. It does not open an
OpenCV window; the desktop app polls the preview image and status JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage4_backend import EFFECT_CHOICES, Stage4FrameProcessor, options_from_values
from stage4_common import load_python_config


DEFAULT_CONTROLS: Dict[str, Any] = {
    "effects": ["glasses", "whiten"],
    "smooth_strength": 0.55,
    "whiten_strength": 0.35,
    "lipstick_alpha": 0.45,
    "show_fps": True,
    "process_width": 720,
    "camera_index": 0,
    "recording": False,
    "recording_output_path": "",
    "user_selected_recording_path": False,
    "recording_fps": 30.0,
    "recording_fourcc": "mp4v",
    "stop_signal": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--process-width", type=int, default=720)
    parser.add_argument("--controls", type=Path, default=Path("stage-4/reports/runtime/live_controls.json"))
    parser.add_argument("--preview", type=Path, default=Path("stage-4/reports/runtime/live_preview.jpg"))
    parser.add_argument("--status", type=Path, default=Path("stage-4/reports/runtime/live_status.json"))
    parser.add_argument("--screenshot-dir", type=Path, default=Path("stage-4/reports/assets/screenshots"))
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_controls(path: Path) -> Dict[str, Any]:
    controls = dict(DEFAULT_CONTROLS)
    if not path.exists():
        return controls
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            controls.update(value)
    except Exception:
        pass
    effects = controls.get("effects") or []
    controls["effects"] = [effect for effect in effects if effect in EFFECT_CHOICES]
    if not controls["effects"]:
        controls["effects"] = []
    return controls


def clamp_strength(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return max(0.0, min(1.0, numeric))


def positive_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return numeric if numeric > 0 else default


def resize_to_width(cv2: Any, frame: Any, width: int):
    if width <= 0:
        return frame
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame
    scale = float(width) / float(current_width)
    return cv2.resize(frame, (width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def write_preview(cv2: Any, path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.jpg")
    if not cv2.imwrite(str(tmp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]):
        raise IOError("Could not write preview: {}".format(path))
    os.replace(str(tmp), str(path))


def status_payload(
    *,
    camera_opened: bool,
    running: bool,
    frame_count: int,
    fps: float | None,
    enabled_effects: list[str],
    last_error: str | None = None,
    recording: bool = False,
    recording_output_path: str | None = None,
    user_selected_recording_path: bool = False,
    recording_frame_count: int = 0,
    last_recording_path: str | None = None,
    last_recording_frame_count: int = 0,
    last_recording_saved: bool = False,
    recording_error: str | None = None,
) -> Dict[str, Any]:
    return {
        "camera_opened": camera_opened,
        "running": running,
        "frame_count": frame_count,
        "fps": fps,
        "enabled_effects": enabled_effects,
        "last_error": last_error,
        "recording": recording,
        "recording_output_path": recording_output_path,
        "user_selected_recording_path": user_selected_recording_path,
        "recording_frame_count": recording_frame_count,
        "last_recording_path": last_recording_path,
        "last_recording_frame_count": last_recording_frame_count,
        "last_recording_saved": last_recording_saved,
        "recording_error": recording_error,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def open_recording_writer(cv2: Any, output_frame: Any, controls: Dict[str, Any]):
    path_value = str(controls.get("recording_output_path") or "").strip()
    if not path_value:
        raise ValueError("recording_output_path is empty")
    output_path = Path(path_value).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = positive_float(controls.get("recording_fps"), 30.0)
    fourcc_text = str(controls.get("recording_fourcc") or "mp4v")
    if len(fourcc_text) != 4:
        fourcc_text = "mp4v"
    height, width = output_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc_text),
        fps,
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter: {}".format(output_path))
    return writer, output_path, (int(width), int(height))


def main() -> int:
    args = parse_args()
    controls_path = resolve_path(args.controls)
    preview_path = resolve_path(args.preview)
    status_path = resolve_path(args.status)
    screenshot_dir = resolve_path(args.screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_python_config(args.config)
    controls = read_controls(controls_path)
    controls["process_width"] = int(controls.get("process_width") or args.process_width)
    controls["camera_index"] = int(controls.get("camera_index") or args.camera)
    atomic_write_json(
        status_path,
        status_payload(
            camera_opened=False,
            running=True,
            frame_count=0,
            fps=None,
            enabled_effects=list(controls.get("effects") or []),
            last_error=None,
            user_selected_recording_path=bool(controls.get("user_selected_recording_path")),
        ),
    )

    options = options_from_values(
        cfg,
        effects=controls.get("effects"),
        smooth_strength=clamp_strength(controls.get("smooth_strength"), 0.55),
        whiten_strength=clamp_strength(controls.get("whiten_strength"), 0.35),
        lipstick_alpha=clamp_strength(controls.get("lipstick_alpha"), 0.45),
        fast_mode=True,
        process_width=int(controls.get("process_width") or args.process_width),
        mode="preview",
    )

    frame_count = 0
    latest_fps: float | None = None
    fps_started = time.perf_counter()
    fps_frames = 0
    last_signature = None
    video_writer = None
    recording_active = False
    recording_path: Path | None = None
    recording_size: tuple[int, int] | None = None
    recording_frame_count = 0
    last_recording_path: str | None = None
    last_recording_frame_count = 0
    last_recording_saved = False
    recording_error: str | None = None
    failed_recording_path: str | None = None

    def release_recording() -> None:
        nonlocal video_writer, recording_active, recording_path, recording_size
        nonlocal recording_frame_count, last_recording_path, last_recording_frame_count, last_recording_saved
        if video_writer is not None:
            video_writer.release()
        if recording_active and recording_path is not None:
            last_recording_path = str(recording_path)
            last_recording_frame_count = recording_frame_count
            last_recording_saved = True
        video_writer = None
        recording_active = False
        recording_path = None
        recording_size = None
        recording_frame_count = 0

    try:
        with Stage4FrameProcessor(cfg, options, static_image_mode=False) as processor:
            cv2 = processor._processor.cv2
            cap = cv2.VideoCapture(int(args.camera))
            if not cap.isOpened():
                message = "Cannot open camera. Please check macOS camera permission."
                atomic_write_json(
                    status_path,
                    status_payload(
                        camera_opened=False,
                        running=False,
                        frame_count=0,
                        fps=None,
                        enabled_effects=list(options.effects),
                        last_error=message,
                        user_selected_recording_path=bool(controls.get("user_selected_recording_path")),
                    ),
                )
                print(message, file=sys.stderr, flush=True)
                print(
                    "Open System Settings > Privacy & Security > Camera and allow Terminal/Python.",
                    file=sys.stderr,
                    flush=True,
                )
                return 2

            print("Stage4 embedded realtime worker started.", flush=True)
            atomic_write_json(
                status_path,
                status_payload(
                    camera_opened=True,
                    running=True,
                    frame_count=0,
                    fps=None,
                    enabled_effects=list(options.effects),
                    last_error=None,
                    user_selected_recording_path=bool(controls.get("user_selected_recording_path")),
                ),
            )

            while True:
                controls = read_controls(controls_path)
                if controls.get("stop_signal"):
                    break

                selected_effects = list(controls.get("effects") or [])
                user_selected_recording_path = bool(controls.get("user_selected_recording_path"))
                smooth_strength = clamp_strength(controls.get("smooth_strength"), 0.55)
                whiten_strength = clamp_strength(controls.get("whiten_strength"), 0.35)
                lipstick_alpha = clamp_strength(controls.get("lipstick_alpha"), 0.45)
                process_width = int(controls.get("process_width") or args.process_width)
                signature = (tuple(sorted(selected_effects)), smooth_strength, whiten_strength, lipstick_alpha)
                if signature != last_signature:
                    options = options_from_values(
                        cfg,
                        effects=selected_effects,
                        smooth_strength=smooth_strength,
                        whiten_strength=whiten_strength,
                        lipstick_alpha=lipstick_alpha,
                        fast_mode=True,
                        process_width=process_width,
                        mode="preview",
                    )
                    processor.update_options(options)
                    last_signature = signature

                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera frame read failed.")

                frame = resize_to_width(cv2, frame, process_width)
                result = processor.process_frame(frame)
                output = result["frame"]
                fps_frames += 1
                elapsed = time.perf_counter() - fps_started
                if elapsed >= 1.0:
                    latest_fps = fps_frames / elapsed
                    fps_started = time.perf_counter()
                    fps_frames = 0

                if bool(controls.get("show_fps", True)) and latest_fps is not None:
                    cv2.putText(
                        output,
                        "FPS {:.1f}".format(latest_fps),
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                requested_recording_path = str(controls.get("recording_output_path") or "")
                wants_recording = bool(controls.get("recording")) and bool(requested_recording_path)
                if not wants_recording and recording_active:
                    release_recording()
                    recording_error = None
                    failed_recording_path = None
                    atomic_write_json(
                        status_path,
                        status_payload(
                            camera_opened=True,
                            running=True,
                            frame_count=frame_count,
                            fps=latest_fps,
                            enabled_effects=sorted(options.effects),
                            last_error=None,
                            recording=recording_active,
                            recording_output_path=None,
                            user_selected_recording_path=user_selected_recording_path,
                            recording_frame_count=recording_frame_count,
                            last_recording_path=last_recording_path,
                            last_recording_frame_count=last_recording_frame_count,
                            last_recording_saved=last_recording_saved,
                            recording_error=recording_error,
                        ),
                    )
                elif wants_recording and not recording_active and requested_recording_path != failed_recording_path:
                    try:
                        video_writer, recording_path, recording_size = open_recording_writer(cv2, output, controls)
                        recording_frame_count = 0
                        recording_active = True
                        recording_error = None
                        last_recording_saved = False
                        print("recording_started={}".format(recording_path), flush=True)
                        atomic_write_json(
                            status_path,
                            status_payload(
                                camera_opened=True,
                                running=True,
                                frame_count=frame_count,
                                fps=latest_fps,
                                enabled_effects=sorted(options.effects),
                                last_error=None,
                                recording=recording_active,
                                recording_output_path=str(recording_path) if recording_path else None,
                                user_selected_recording_path=user_selected_recording_path,
                                recording_frame_count=recording_frame_count,
                                last_recording_path=last_recording_path,
                                last_recording_frame_count=last_recording_frame_count,
                                last_recording_saved=last_recording_saved,
                                recording_error=recording_error,
                            ),
                        )
                    except Exception as exc:
                        recording_error = "{}: {}".format(type(exc).__name__, exc)
                        failed_recording_path = requested_recording_path
                        print(recording_error, file=sys.stderr, flush=True)
                        atomic_write_json(
                            status_path,
                            status_payload(
                                camera_opened=True,
                                running=True,
                                frame_count=frame_count,
                                fps=latest_fps,
                                enabled_effects=sorted(options.effects),
                                last_error=None,
                                recording=False,
                                recording_output_path=None,
                                user_selected_recording_path=user_selected_recording_path,
                                recording_frame_count=0,
                                last_recording_path=last_recording_path,
                                last_recording_frame_count=last_recording_frame_count,
                                last_recording_saved=last_recording_saved,
                                recording_error=recording_error,
                            ),
                        )
                if recording_active and video_writer is not None:
                    write_frame = output
                    if recording_size and (output.shape[1], output.shape[0]) != recording_size:
                        write_frame = cv2.resize(output, recording_size, interpolation=cv2.INTER_AREA)
                    video_writer.write(write_frame)
                    recording_frame_count += 1

                write_preview(cv2, preview_path, output)
                frame_count += 1
                if frame_count % 5 == 0:
                    atomic_write_json(
                        status_path,
                        status_payload(
                            camera_opened=True,
                            running=True,
                            frame_count=frame_count,
                            fps=latest_fps,
                            enabled_effects=sorted(options.effects),
                            last_error=None,
                            recording=recording_active,
                            recording_output_path=str(recording_path) if recording_path else None,
                            user_selected_recording_path=user_selected_recording_path,
                            recording_frame_count=recording_frame_count,
                            last_recording_path=last_recording_path,
                            last_recording_frame_count=last_recording_frame_count,
                            last_recording_saved=last_recording_saved,
                            recording_error=recording_error,
                        ),
                    )

            release_recording()
            cap.release()
            atomic_write_json(
                status_path,
                status_payload(
                    camera_opened=True,
                    running=False,
                    frame_count=frame_count,
                    fps=latest_fps,
                    enabled_effects=sorted(options.effects),
                    last_error=None,
                    recording=recording_active,
                    recording_output_path=str(recording_path) if recording_path else None,
                    user_selected_recording_path=bool(controls.get("user_selected_recording_path")),
                    recording_frame_count=recording_frame_count,
                    last_recording_path=last_recording_path,
                    last_recording_frame_count=last_recording_frame_count,
                    last_recording_saved=last_recording_saved,
                    recording_error=recording_error,
                ),
            )
            print("Stage4 embedded realtime worker stopped.", flush=True)
            return 0
    except Exception as exc:
        release_recording()
        message = "{}: {}".format(type(exc).__name__, exc)
        atomic_write_json(
            status_path,
            status_payload(
                camera_opened=False,
                running=False,
                frame_count=frame_count,
                fps=latest_fps,
                enabled_effects=list(controls.get("effects") or []),
                last_error=message,
                recording=recording_active,
                recording_output_path=str(recording_path) if recording_path else None,
                user_selected_recording_path=bool(controls.get("user_selected_recording_path")),
                recording_frame_count=recording_frame_count,
                last_recording_path=last_recording_path,
                last_recording_frame_count=last_recording_frame_count,
                last_recording_saved=last_recording_saved,
                recording_error=recording_error,
            ),
        )
        print(message, file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
