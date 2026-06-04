#!/usr/bin/env python3
"""Create same-flow fixed-sample comparison sheets for pretrained and self-trained StarGAN checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from stage3_task7_common import (
    asset_dir,
    cfg_get,
    checkpoint_path_for_iters,
    generate_fixed_samples,
    load_config,
    load_fixed_manifest,
    read_json,
    selected_attrs,
    summary_dir,
    write_json,
)
from stage3_task7_pretrained_sanity import ensure_pretrained_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task7_stargan/a800_full.py")
    parser.add_argument("--test-iters", type=int, default=None)
    parser.add_argument("--pretrained-zip", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--refresh", action="store_true", help="Regenerate fixed grids before creating the comparison sheet.")
    return parser.parse_args()


def summary_path_for(cfg: Dict[str, Any], run_name: str, test_iters: int) -> Path:
    if run_name == "pretrained":
        return asset_dir(cfg) / "pretrained" / "pretrained_200000_fixed_summary.json"
    return asset_dir(cfg) / "self_trained" / f"iter_{test_iters}" / f"self_trained_{test_iters}_fixed_summary.json"


def ensure_fixed_summary(
    cfg: Dict[str, Any],
    run_name: str,
    checkpoint: Path,
    test_iters: int,
    device: Optional[str],
    refresh: bool,
) -> Dict[str, Any]:
    path = summary_path_for(cfg, run_name, test_iters)
    if path.exists() and not refresh:
        return read_json(path)
    if run_name == "pretrained":
        output_dir = asset_dir(cfg) / "pretrained"
        return generate_fixed_samples(cfg, checkpoint, "pretrained_200000", output_dir, device=device)
    output_dir = asset_dir(cfg) / "self_trained" / f"iter_{test_iters}"
    return generate_fixed_samples(cfg, checkpoint, f"self_trained_{test_iters}", output_dir, device=device)


def records_by_key(summary: Dict[str, Any]) -> Dict[Tuple[str, str], Path]:
    records = {}
    for record in summary["generated"]:
        records[(record["sample_id"], record["direction"])] = Path(record["image_path"])
    return records


def open_rgb(path: Path, size: int) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def make_side_by_side(cfg: Dict[str, Any], pretrained_summary: Dict[str, Any], self_summary: Dict[str, Any], test_iters: int) -> Path:
    attrs = selected_attrs(cfg)
    manifest = load_fixed_manifest(cfg)
    samples = manifest["samples"]
    pre_records = records_by_key(pretrained_summary)
    self_records = records_by_key(self_summary)
    cell = int(cfg_get(cfg, "model", "image_size", 128))
    header_h = 32
    row_label_w = 82
    columns = ["source"]
    for attr in attrs:
        columns.extend([f"pre {attr}", f"self {attr}"])
    canvas = Image.new("RGB", (row_label_w + len(columns) * cell, header_h + len(samples) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col_idx, title in enumerate(columns):
        draw.text((row_label_w + col_idx * cell + 4, 9), title[:20], fill=(20, 20, 20), font=font)
    for row_idx, sample in enumerate(samples):
        y = header_h + row_idx * cell
        draw.text((4, y + 6), sample["sample_id"], fill=(20, 20, 20), font=font)
        canvas.paste(open_rgb(Path(sample["source_path"]), cell), (row_label_w, y))
        col_idx = 1
        for attr in attrs:
            pre_path = pre_records[(sample["sample_id"], attr)]
            self_path = self_records[(sample["sample_id"], attr)]
            canvas.paste(open_rgb(pre_path, cell), (row_label_w + col_idx * cell, y))
            canvas.paste(open_rgb(self_path, cell), (row_label_w + (col_idx + 1) * cell, y))
            col_idx += 2
    output_dir = asset_dir(cfg) / "side_by_side" / f"iter_{test_iters}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pretrained_vs_self_trained_fixed_side_by_side.jpg"
    canvas.save(output_path, quality=95)
    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    test_iters = args.test_iters or int(cfg_get(cfg, "train", "final_test_iters", cfg_get(cfg, "train", "num_iters", 200000)))
    pretrained_checkpoint = ensure_pretrained_checkpoint(cfg, args.pretrained_zip)
    self_checkpoint = checkpoint_path_for_iters(cfg, test_iters)
    if not self_checkpoint.exists():
        raise FileNotFoundError(f"Missing self-trained checkpoint: {self_checkpoint}")
    pretrained_summary = ensure_fixed_summary(cfg, "pretrained", pretrained_checkpoint, test_iters, args.device, args.refresh)
    self_summary = ensure_fixed_summary(cfg, "self_trained", self_checkpoint, test_iters, args.device, args.refresh)
    side_by_side = make_side_by_side(cfg, pretrained_summary, self_summary, test_iters)
    summary = {
        "test_iters": test_iters,
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "self_trained_checkpoint": str(self_checkpoint),
        "pretrained_fixed_grid": pretrained_summary["grid"],
        "self_trained_fixed_grid": self_summary["grid"],
        "side_by_side": str(side_by_side),
        "diagnosis_rule": (
            "If pretrained output is clean but self-trained output is abnormal, prioritize training/configuration. "
            "If pretrained output is also abnormal, prioritize preprocessing, postprocessing, and checkpoint-loading checks before tuning losses."
        ),
        "pretrained_normal": "manual_review_required",
        "self_trained_normal": "manual_review_required",
    }
    write_json(summary_dir(cfg) / "fixed_checkpoint_compare_summary.json", summary)
    print("Pretrained normal: manual_review_required")
    print("Self-trained normal: manual_review_required")
    print(summary["diagnosis_rule"])


if __name__ == "__main__":
    main()
