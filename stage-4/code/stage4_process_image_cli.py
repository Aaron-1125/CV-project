#!/usr/bin/env python3
"""Stage4 image-processing CLI for local import mode."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage4_backend import EFFECT_CHOICES, Stage4FrameProcessor, options_from_values, runtime_environment
from stage4_common import load_python_config, now_ts, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--effects", nargs="+", choices=EFFECT_CHOICES, default=None)
    parser.add_argument("--smooth-strength", type=float, default=None)
    parser.add_argument("--whiten-strength", type=float, default=None)
    parser.add_argument("--lipstick-alpha", type=float, default=None)
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--process-width", type=int, default=None)
    parser.add_argument("--output-image", type=Path, default=Path("stage-4/reports/assets/images/stage4_image_effects_export.jpg"))
    parser.add_argument("--summary", type=Path, default=Path("stage-4/reports/summaries/stage4_image_effects_summary.json"))
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def resize_to_width(cv2: Any, frame: Any, width: int):
    if width <= 0:
        return frame
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame
    scale = float(width) / float(current_width)
    return cv2.resize(frame, (width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    cfg = load_python_config(args.config)
    image_path = resolve_path(args.image)
    output_path = resolve_path(args.output_image)
    summary_path = resolve_path(args.summary)
    if not image_path.is_file():
        raise FileNotFoundError("Image does not exist: {}".format(image_path))

    options = options_from_values(
        cfg,
        effects=args.effects,
        smooth_strength=args.smooth_strength,
        whiten_strength=args.whiten_strength,
        lipstick_alpha=args.lipstick_alpha,
        fast_mode=True if args.fast_mode else None,
        process_width=args.process_width,
        mode="preview" if args.fast_mode else "export",
    )
    started = time.perf_counter()
    with Stage4FrameProcessor(cfg, options, static_image_mode=True) as processor:
        cv2 = processor._processor.cv2
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError("Could not read image: {}".format(image_path))
        input_shape = frame.shape[:2]
        frame = resize_to_width(cv2, frame, args.process_width or 0)
        result = processor.process_frame(frame)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), result["frame"]):
            raise IOError("Could not write output image: {}".format(output_path))
    elapsed = time.perf_counter() - started

    payload: Dict[str, Any] = {
        "stage": "stage4",
        "task": "local_image_import",
        "generated_at": now_ts(),
        "input_image": str(image_path),
        "output_image": str(output_path),
        "enabled_effects": sorted(options.effects),
        "strengths": {
            "smooth_strength": options.smooth_strength,
            "whiten_strength": options.whiten_strength,
            "lipstick_alpha": options.lipstick_alpha,
        },
        "mode": "fast_preview" if args.fast_mode else "quality_export",
        "fast_mode": bool(args.fast_mode),
        "process_width": args.process_width,
        "input_resolution": {"height": int(input_shape[0]), "width": int(input_shape[1])},
        "output_resolution": {
            "height": int(result["frame"].shape[0]),
            "width": int(result["frame"].shape[1]),
        },
        "face_detected": bool(result.get("face_detected")),
        "elapsed_seconds": elapsed,
        "avg_ms_per_image": elapsed * 1000.0,
        "environment": runtime_environment(),
        "output_files": {
            "image": str(output_path),
            "summary": str(summary_path),
        },
        "notes": [
            "Image processing runs in this CLI subprocess, not in the desktop GUI process.",
            "The same Task9 FaceEffectsProcessor is reused for single-image effects.",
        ],
    }
    write_json(summary_path, payload)
    print("Stage4 image export complete: {}".format(output_path))
    print("Summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
