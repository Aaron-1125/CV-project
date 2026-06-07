#!/usr/bin/env python3
"""Prepare static images and stickers for Stage3 Task9 face effects."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from stage3_task9_common import (
    NO_VIDEO_HINT,
    cfg_get,
    copy_sample,
    default_video_path,
    ensure_default_stickers,
    ensure_task9_dirs,
    find_celeba_image_dir,
    list_images,
    load_config,
    locate_user_video,
    now_ts,
    prepare_summary_path,
    safe_stem,
    save_image_grid,
    select_samples,
    static_images_dir,
    summary_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task9_effects/a800_mediapipe_face_effects.py")
    parser.add_argument("--celeba-root", type=Path, default=None, help="CelebA root used only for static image smoke tests.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Use static images from this directory instead of CelebA.")
    parser.add_argument("--video", type=Path, default=None, help="Optional user mp4 path to record in the prepare summary.")
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace existing staged static images and generated stickers.")
    return parser.parse_args()


def collect_static_images(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    recursive = bool(args.recursive or cfg_get(cfg, "data", "recursive", False))
    if args.input_dir:
        images = list_images(args.input_dir, recursive=recursive)
        return {
            "source_dataset": "input_dir",
            "source_root": str(args.input_dir.expanduser().resolve()),
            "source_image_dir": str(args.input_dir.expanduser().resolve()),
            "recursive": recursive,
            "images": images,
        }
    if not bool(cfg_get(cfg, "data", "use_celeba_for_static_images", True)):
        return {
            "source_dataset": "none",
            "source_root": None,
            "source_image_dir": None,
            "recursive": recursive,
            "images": [],
        }
    celeba_root = args.celeba_root or Path(os.environ.get("CELEBA_ROOT", str(cfg_get(cfg, "data", "celeba_root", "data/celeba"))))
    image_dir = find_celeba_image_dir(celeba_root)
    images = list_images(image_dir, recursive=recursive)
    return {
        "source_dataset": "CelebA_static_only",
        "source_root": str(Path(celeba_root).expanduser().resolve()),
        "source_image_dir": str(image_dir),
        "recursive": recursive,
        "images": images,
    }


def stage_static_images(samples: List[Path], output_dir: Path, force: bool) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for idx, source in enumerate(samples):
        sample_id = "static_{:03d}".format(idx)
        suffix = source.suffix.lower() if source.suffix else ".jpg"
        target = output_dir / "{}_{}{}".format(sample_id, safe_stem(source.stem), suffix)
        action = copy_sample(source.resolve(), target, force=force)
        records.append(
            {
                "sample_id": sample_id,
                "source_path": str(source.resolve()),
                "staged_path": str(target.absolute()),
                "action": action,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_task9_dirs(cfg)
    count = args.sample_count if args.sample_count is not None else int(cfg_get(cfg, "data", "sample_count", 8))
    out_dir = args.output_dir or static_images_dir(cfg)
    try:
        source = collect_static_images(args, cfg)
    except FileNotFoundError as exc:
        print("Static image preparation skipped: {}".format(exc))
        source = {
            "source_dataset": "unavailable",
            "source_root": str(args.input_dir or args.celeba_root or cfg_get(cfg, "data", "celeba_root", "")),
            "source_image_dir": None,
            "recursive": bool(args.recursive or cfg_get(cfg, "data", "recursive", False)),
            "images": [],
            "error": str(exc),
        }
    records: List[Dict[str, Any]] = []
    grid = None
    if source["images"]:
        selected = select_samples(source["images"], count, int(cfg.get("seed", 20260605)))
        records = stage_static_images(selected, out_dir, args.force)
        grid = save_image_grid([Path(row["staged_path"]) for row in records], [row["sample_id"] for row in records], out_dir / "static_input_grid.jpg")
    stickers = ensure_default_stickers(cfg, force=args.force)
    user_video = locate_user_video(cfg, args.video)
    video_status = "available" if user_video else "missing"
    if not user_video:
        print(NO_VIDEO_HINT)
    payload = {
        "task": cfg.get("task_name"),
        "ready": True,
        "prepared_at": now_ts(),
        "static_sample_count": len(records),
        "requested_sample_count": count,
        "source_dataset": source["source_dataset"],
        "source_root": source.get("source_root"),
        "source_image_dir": source.get("source_image_dir"),
        "source_error": source.get("error"),
        "output_static_image_dir": str(out_dir),
        "static_input_grid": str(grid) if grid else None,
        "static_images": records,
        "stickers": stickers,
        "user_video_status": video_status,
        "user_video_path": str(user_video) if user_video else None,
        "default_video_path": str(default_video_path(cfg)),
        "video_hint": None if user_video else NO_VIDEO_HINT,
        "synthetic_video_from_images": False,
    }
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    write_json(prepare_summary_path(cfg), payload)


if __name__ == "__main__":
    main()
