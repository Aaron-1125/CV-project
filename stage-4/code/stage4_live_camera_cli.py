#!/usr/bin/env python3
"""Stage4 realtime camera CLI with an independent OpenCV window."""

from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage4_backend import EFFECT_CHOICES, Stage4FrameProcessor, options_from_values
from stage4_common import load_python_config, now_ts


WINDOW_TITLE = "Stage4 Realtime Face Effects"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--effects", nargs="+", choices=EFFECT_CHOICES, default=None)
    parser.add_argument("--process-width", type=int, default=720)
    parser.add_argument("--show-fps", action="store_true")
    parser.add_argument("--screenshot-dir", type=Path, default=Path("stage-4/reports/assets/screenshots"))
    return parser.parse_args()


def resize_to_width(cv2, frame, width: int):
    if width <= 0:
        return frame
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame
    scale = float(width) / float(current_width)
    return cv2.resize(frame, (width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def stdin_command() -> Optional[str]:
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
    except Exception:
        return None
    if not readable:
        return None
    value = sys.stdin.readline().strip().lower()
    return value or None


def save_screenshot(cv2, frame, screenshot_dir: Path, frame_index: int) -> Path:
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    path = screenshot_dir / "stage4_camera_{:06d}_{}.jpg".format(frame_index, now_ts().replace(":", "-"))
    if not cv2.imwrite(str(path), frame):
        raise IOError("Could not write screenshot: {}".format(path))
    return path


def main() -> None:
    args = parse_args()
    cfg = load_python_config(args.config)
    options = options_from_values(
        cfg,
        effects=args.effects,
        fast_mode=True,
        process_width=args.process_width,
        mode="preview",
    )
    with Stage4FrameProcessor(cfg, options, static_image_mode=False) as processor:
        cv2 = processor._processor.cv2
        cap = cv2.VideoCapture(int(args.camera))
        if not cap.isOpened():
            print("Cannot open camera. Please check macOS camera permission.", file=sys.stderr)
            print("Open System Settings > Privacy & Security > Camera and allow Terminal/Python.", file=sys.stderr)
            return

        print("Stage4 realtime camera started. Press q to quit, s to save screenshot.", flush=True)
        frame_index = 0
        fps_started = time.perf_counter()
        fps_frames = 0
        latest_fps = 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame read failed.", file=sys.stderr, flush=True)
                    break
                frame = resize_to_width(cv2, frame, args.process_width)
                result = processor.process_frame(frame)
                output = result["frame"]
                fps_frames += 1
                elapsed = time.perf_counter() - fps_started
                if elapsed >= 1.0:
                    latest_fps = fps_frames / elapsed
                    if args.show_fps:
                        print("fps={:.2f} frame={}".format(latest_fps, frame_index), flush=True)
                    fps_started = time.perf_counter()
                    fps_frames = 0
                if args.show_fps:
                    cv2.putText(
                        output,
                        "FPS {:.1f}".format(latest_fps),
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(WINDOW_TITLE, output)
                key = cv2.waitKey(1) & 0xFF
                command = stdin_command()
                if key == ord("s") or command == "s":
                    path = save_screenshot(cv2, output, args.screenshot_dir, frame_index)
                    print("screenshot={}".format(path), flush=True)
                if key == ord("q") or command in {"q", "quit", "exit"}:
                    break
                frame_index += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Stage4 realtime camera stopped.", flush=True)


if __name__ == "__main__":
    main()
