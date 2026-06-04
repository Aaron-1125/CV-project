#!/usr/bin/env python3
"""Prepare input images for Stage3 Task8 3D face reconstruction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from stage3_task8_common import (
    cfg_get,
    copy_or_symlink,
    find_celeba_image_dir,
    input_samples_dir,
    list_images,
    load_config,
    prepare_summary_path,
    read_image_list,
    safe_stem,
    save_image_grid,
    select_samples,
    summary_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task8_3dface/a800_3ddfa_v2.py")
    parser.add_argument("--celeba-root", type=Path, default=None, help="CelebA root. Defaults to CELEBA_ROOT or config.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Use images from this directory instead of CelebA.")
    parser.add_argument("--image-list", type=Path, default=None, help="Text file with one image path per line.")
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --input-dir or CelebA image root.")
    parser.add_argument("--force", action="store_true", help="Replace existing staged sample files.")
    return parser.parse_args()


def collect_source_images(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    recursive = bool(args.recursive or cfg_get(cfg, "data", "recursive", False))
    if args.image_list:
        images = read_image_list(args.image_list)
        return {
            "source_dataset": "image_list",
            "source_root": str(args.image_list.expanduser().resolve()),
            "images": images,
            "recursive": False,
        }
    if args.input_dir:
        images = list_images(args.input_dir, recursive=recursive)
        return {
            "source_dataset": "input_dir",
            "source_root": str(args.input_dir.expanduser().resolve()),
            "images": images,
            "recursive": recursive,
        }
    celeba_root = args.celeba_root or Path(os.environ.get("CELEBA_ROOT", str(cfg_get(cfg, "data", "celeba_root", "data/celeba"))))
    image_dir = find_celeba_image_dir(celeba_root)
    images = list_images(image_dir, recursive=recursive)
    return {
        "source_dataset": "CelebA",
        "source_root": str(Path(celeba_root).expanduser().resolve()),
        "source_image_dir": str(image_dir),
        "images": images,
        "recursive": recursive,
    }


def stage_samples(samples: List[Path], out_dir: Path, link_mode: str, force: bool) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for idx, source in enumerate(samples):
        sample_id = "sample_{:03d}".format(idx)
        suffix = source.suffix.lower() if source.suffix else ".jpg"
        staged_name = "{}_{}{}".format(sample_id, safe_stem(source.stem), suffix)
        target = out_dir / staged_name
        action = copy_or_symlink(source.resolve(), target, mode=link_mode, force=force)
        records.append(
            {
                "sample_id": sample_id,
                "filename": staged_name,
                "source_path": str(source.resolve()),
                "staged_path": str(target.absolute()),
                "link_mode": link_mode,
                "action": action,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    count = args.sample_count if args.sample_count is not None else int(cfg_get(cfg, "data", "sample_count", 8))
    link_mode = args.link_mode or str(cfg_get(cfg, "data", "link_mode", "symlink"))
    output_dir = args.output_dir or input_samples_dir(cfg)
    source = collect_source_images(args, cfg)
    selected = select_samples(source["images"], count, int(cfg.get("seed", 20260604)))
    records = stage_samples(selected, output_dir, link_mode, args.force)
    grid_path = output_dir / "input_samples_grid.jpg"
    grid = save_image_grid([Path(row["staged_path"]) for row in records], [row["sample_id"] for row in records], grid_path)
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    payload = {
        "task": cfg.get("task_name"),
        "ready": True,
        "sample_count": len(records),
        "requested_sample_count": count,
        "source_dataset": source["source_dataset"],
        "source_root": source.get("source_root"),
        "source_image_dir": source.get("source_image_dir"),
        "recursive": source.get("recursive"),
        "output_directory": str(output_dir),
        "input_grid": str(grid) if grid else None,
        "input_image_paths": [row["staged_path"] for row in records],
        "samples": records,
    }
    write_json(prepare_summary_path(cfg), payload)


if __name__ == "__main__":
    main()
