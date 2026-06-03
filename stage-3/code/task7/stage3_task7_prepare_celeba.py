#!/usr/bin/env python3
"""Prepare CelebA paths, fixed samples, and target labels for Stage3 Task7."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from stage3_task7_common import (
    asset_dir,
    cfg_get,
    copy_or_symlink,
    ensure_stargan_repo,
    find_celeba_paths,
    fixed_manifest_path,
    load_config,
    official_stargan_split,
    parse_attr_file,
    prepared_attr_path,
    prepared_image_dir,
    save_source_grid,
    select_fixed_samples,
    selected_attrs,
    summary_dir,
    write_json,
    write_target_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task7_stargan/a800_full.py")
    parser.add_argument("--celeba-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.report_dir:
        cfg.setdefault("reports", {})
        cfg["reports"]["report_dir"] = str(args.report_dir)
        cfg["reports"]["summary_dir"] = str(args.report_dir / "summaries")
        cfg["reports"]["asset_dir"] = str(args.report_dir / "assets")
    repo = ensure_stargan_repo(cfg)
    attrs = selected_attrs(cfg)
    celeba_root = args.celeba_root or Path(os.environ.get("CELEBA_ROOT", str(cfg_get(cfg, "data", "celeba_root", "data/celeba_source"))))
    source_image_dir, source_attr_path = find_celeba_paths(celeba_root)

    image_dir = prepared_image_dir(cfg)
    attr_path = prepared_attr_path(cfg)
    copy_or_symlink(source_image_dir, image_dir, force=args.force)
    copy_or_symlink(source_attr_path, attr_path, force=args.force)

    all_attrs, rows = parse_attr_file(attr_path)
    train_rows, test_rows = official_stargan_split(rows)
    count = int(cfg_get(cfg, "data", "fixed_sample_count", 16))
    samples = select_fixed_samples(test_rows, attrs, image_dir, count)
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    asset_dir(cfg).mkdir(parents=True, exist_ok=True)
    source_grid = asset_dir(cfg) / "fixed_samples" / "fixed_source_grid.jpg"
    save_source_grid(samples, source_grid, size=int(cfg_get(cfg, "model", "image_size", 128)))
    target_paths = write_target_labels(samples, attrs, summary_dir(cfg) / "fixed_samples_target_labels")
    manifest = {
        "task": cfg.get("task_name"),
        "stargan_repo": repo,
        "celeba_root": str(Path(celeba_root).expanduser().resolve()),
        "source_image_dir": str(source_image_dir),
        "source_attr_path": str(source_attr_path),
        "image_dir": str(image_dir),
        "attr_path": str(attr_path),
        "num_attrs": len(all_attrs),
        "selected_attrs": attrs,
        "total_images": len(rows),
        "official_train_images": len(train_rows),
        "official_test_images": len(test_rows),
        "samples": samples,
        "source_grid": str(source_grid),
        "target_labels": target_paths,
    }
    write_json(fixed_manifest_path(cfg), manifest)
    write_json(
        summary_dir(cfg) / "celeba_prepare_summary.json",
        {
            "ready": True,
            "image_dir": str(image_dir),
            "attr_path": str(attr_path),
            "fixed_manifest": str(fixed_manifest_path(cfg)),
            "fixed_sample_count": len(samples),
        },
    )


if __name__ == "__main__":
    main()
