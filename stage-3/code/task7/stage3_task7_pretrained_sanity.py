#!/usr/bin/env python3
"""Run official StarGAN CelebA-128 pretrained sanity check on fixed samples."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from stage3_task7_common import (
    asset_dir,
    ensure_stargan_repo,
    generate_fixed_samples,
    load_config,
    run_command,
    stargan_repo,
    summary_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task7_stargan/a800_full.py")
    parser.add_argument("--pretrained-zip", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def ensure_pretrained_checkpoint(cfg: dict, pretrained_zip: Path | None) -> Path:
    repo = stargan_repo(cfg)
    checkpoint = repo / "stargan_celeba_128" / "models" / "200000-G.ckpt"
    if checkpoint.exists():
        return checkpoint
    models_dir = checkpoint.parent
    models_dir.mkdir(parents=True, exist_ok=True)
    if pretrained_zip:
        with zipfile.ZipFile(pretrained_zip) as archive:
            archive.extractall(models_dir)
        if checkpoint.exists():
            return checkpoint
        candidates = list(models_dir.rglob("200000-G.ckpt"))
        if candidates:
            shutil.move(str(candidates[0]), checkpoint)
            return checkpoint
        raise FileNotFoundError(f"No 200000-G.ckpt found in {pretrained_zip}")
    run_command(["bash", "download.sh", "pretrained-celeba-128x128"], cwd=repo)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Official pretrained checkpoint was not created: {checkpoint}")
    return checkpoint


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    repo_info = ensure_stargan_repo(cfg)
    checkpoint = ensure_pretrained_checkpoint(cfg, args.pretrained_zip)
    output_dir = asset_dir(cfg) / "pretrained"
    fixed = generate_fixed_samples(cfg, checkpoint, "pretrained_200000", output_dir, device=args.device)
    summary = {
        "ready": True,
        "stargan_repo": repo_info,
        "checkpoint": str(checkpoint),
        "fixed_grid": fixed["grid"],
        "fixed_summary": str(output_dir / "pretrained_200000_fixed_summary.json"),
    }
    write_json(summary_dir(cfg) / "pretrained_sanity_summary.json", summary)


if __name__ == "__main__":
    main()

