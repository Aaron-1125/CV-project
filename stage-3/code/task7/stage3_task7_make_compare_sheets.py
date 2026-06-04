#!/usr/bin/env python3
"""Build source-vs-generated review sheets from an evaluation generation manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

from stage3_task7_common import asset_dir, cfg_get, load_config, read_json, selected_attrs, summary_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task7_stargan/a800_full.py")
    parser.add_argument("--test-iters", type=int, default=None)
    parser.add_argument("--columns", type=int, default=4, help="Number of source/generated pairs per row.")
    parser.add_argument("--max-per-direction", type=int, default=None, help="Optional cap for each direction sheet.")
    return parser.parse_args()


def open_rgb(path: Path, size: int) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def make_sheet(direction: str, records: List[Dict[str, Any]], output_path: Path, cell: int, columns: int) -> Dict[str, Any]:
    font = ImageFont.load_default()
    pair_gap = 8
    block_w = cell * 2 + pair_gap
    title_h = 30
    label_h = 22
    block_h = label_h + cell
    rows = (len(records) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * block_w, title_h + rows * block_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), direction, fill=(20, 20, 20), font=font)
    for idx, record in enumerate(records):
        row_idx = idx // columns
        col_idx = idx % columns
        x = col_idx * block_w
        y = title_h + row_idx * block_h
        label = Path(record["generated_path"]).name
        draw.text((x + 4, y + 4), label[:36], fill=(20, 20, 20), font=font)
        source = open_rgb(Path(record["source_path"]), cell)
        generated = open_rgb(Path(record["generated_path"]), cell)
        canvas.paste(source, (x, y + label_h))
        canvas.paste(generated, (x + cell + pair_gap, y + label_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    print(f"Wrote {output_path}")
    return {"direction": direction, "records": len(records), "sheet": str(output_path)}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    test_iters = args.test_iters or int(cfg_get(cfg, "train", "final_test_iters", cfg_get(cfg, "train", "num_iters", 200000)))
    eval_dir = asset_dir(cfg) / "evaluation" / f"iter_{test_iters}"
    manifest_path = eval_dir / "eval_generation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing evaluation generation manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in manifest["records"]:
        grouped[record["direction"]].append(record)
    output_dir = eval_dir / "source_vs_generated"
    cell = int(cfg_get(cfg, "model", "image_size", 128))
    summaries = []
    for direction in selected_attrs(cfg):
        records = grouped.get(direction, [])
        if args.max_per_direction is not None:
            records = records[: args.max_per_direction]
        if not records:
            continue
        output_path = output_dir / f"{direction}_source_vs_generated.jpg"
        summaries.append(make_sheet(direction, records, output_path, cell, args.columns))
    write_json(
        summary_dir(cfg) / f"source_vs_generated_sheets_{test_iters}_summary.json",
        {
            "test_iters": test_iters,
            "manifest": str(manifest_path),
            "output_dir": str(output_dir),
            "directions": summaries,
        },
    )


if __name__ == "__main__":
    main()
