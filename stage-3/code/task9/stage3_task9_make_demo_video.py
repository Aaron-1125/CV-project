#!/usr/bin/env python3
"""Finalize the Task9 demo video or static contact sheet."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from stage3_task9_common import (
    NO_VIDEO_HINT,
    demo_summary_path,
    demo_video_path,
    effects_summary_path,
    ensure_task9_dirs,
    list_images,
    load_config,
    maybe_read_json,
    outputs_dir,
    save_image_grid,
    static_contact_sheet_path,
    summary_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task9_effects/a800_mediapipe_face_effects.py")
    parser.add_argument("--processed-video", type=Path, default=None, help="Optional processed video to register as the final demo.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def find_processed_video(cfg: Dict[str, Any], explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit:
        candidate = explicit.expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.exists():
            return candidate.resolve()
    effects = maybe_read_json(effects_summary_path(cfg))
    summary_video = effects.get("output_video")
    if summary_video and Path(str(summary_video)).exists():
        return Path(str(summary_video)).resolve()
    final = demo_video_path(cfg)
    if final.exists():
        return final.resolve()
    return None


def build_static_contact_sheet(cfg: Dict[str, Any]) -> Optional[Path]:
    candidates: List[Path] = []
    for pattern in ["*_before_after.jpg", "*_effects.jpg", "*_landmarks.jpg"]:
        candidates.extend(sorted(outputs_dir(cfg).glob(pattern)))
    if not candidates:
        try:
            candidates = list_images(outputs_dir(cfg), recursive=False)
        except Exception:
            candidates = []
    selected = candidates[:9]
    if not selected:
        return None
    labels = [path.stem[:28] for path in selected]
    return save_image_grid(selected, labels, static_contact_sheet_path(cfg))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_task9_dirs(cfg)
    final_video = demo_video_path(cfg)
    processed_video = find_processed_video(cfg, args.processed_video)
    payload: Dict[str, Any] = {
        "task": cfg.get("task_name"),
        "synthetic_video_from_images": False,
    }
    if processed_video:
        if processed_video.resolve() != final_video.resolve():
            if final_video.exists() and args.force:
                final_video.unlink()
            if not final_video.exists():
                final_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(processed_video), str(final_video))
        payload.update(
            {
                "ready": True,
                "video_demo_status": "available",
                "processed_video": str(processed_video),
                "final_demo_video": str(final_video),
            }
        )
    else:
        contact_sheet = build_static_contact_sheet(cfg)
        print(NO_VIDEO_HINT)
        payload.update(
            {
                "ready": bool(contact_sheet),
                "video_demo_status": "skipped_no_user_video",
                "final_demo_video": None,
                "static_contact_sheet": str(contact_sheet) if contact_sheet else None,
                "video_hint": NO_VIDEO_HINT,
            }
        )
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    write_json(demo_summary_path(cfg), payload)


if __name__ == "__main__":
    main()
