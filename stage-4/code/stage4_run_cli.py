#!/usr/bin/env python3
"""Stage4 CLI adapter for Task9 dynamic face effects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage4_backend import EFFECT_CHOICES, build_static_summary, options_from_values, run_video_export
from stage4_common import load_python_config, stage4_summary_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Stage4 config path. Defaults to configs/stage4_app_config.py.")
    parser.add_argument("--video", type=Path, default=None, help="Input video path.")
    parser.add_argument("--camera", type=int, default=None, help="Optional camera index.")
    parser.add_argument("--output-video", type=Path, default=None, help="Output mp4 path.")
    parser.add_argument("--output-path-source", choices=["user", "default"], default=None)
    parser.add_argument("--effects", nargs="+", choices=EFFECT_CHOICES, default=None)
    parser.add_argument("--smooth-strength", type=float, default=None)
    parser.add_argument("--whiten-strength", type=float, default=None)
    parser.add_argument("--lipstick-alpha", type=float, default=None)
    parser.add_argument("--fast-mode", action="store_true", help="Use realtime-friendly 640x360 defaults unless dimensions are overridden.")
    parser.add_argument("--process-width", type=int, default=None)
    parser.add_argument("--process-height", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-keyframes", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--debug-sticker-geometry", action="store_true")
    parser.add_argument("--summary", type=Path, default=None, help="Summary JSON path.")
    parser.add_argument("--check-env", action="store_true", help="Only write and print the Stage4 environment summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_python_config(args.config)
    summary_path = args.summary or stage4_summary_path()
    if args.check_env:
        summary = build_static_summary(cfg)
        env_summary_path = args.summary or stage4_summary_path().with_name("stage4_env_check_summary.json")
        write_json(env_summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    options = options_from_values(
        cfg,
        effects=args.effects,
        smooth_strength=args.smooth_strength,
        whiten_strength=args.whiten_strength,
        lipstick_alpha=args.lipstick_alpha,
        fast_mode=True if args.fast_mode else None,
        process_width=args.process_width,
        process_height=args.process_height,
        max_frames=args.max_frames,
        fps=args.fps,
        max_keyframes=args.max_keyframes,
        device=args.device,
        debug_sticker_geometry=args.debug_sticker_geometry,
        mode="export",
    )
    summary = run_video_export(
        cfg,
        options,
        video=args.video,
        camera=args.camera,
        output_video=args.output_video,
        summary_output=summary_path,
        user_selected_output_path=(
            None if args.output_path_source is None else args.output_path_source == "user"
        ),
    )
    print("Stage4 export complete: {}".format(summary.get("output_video")))
    print("Summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
