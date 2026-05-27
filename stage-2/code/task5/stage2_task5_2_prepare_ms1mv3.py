#!/usr/bin/env python3
"""Prepare an MS1MV3/MS1M-RetinaFace subset for Stage2 task 5.x.

The preferred source is the Hugging Face cleaned MS-Celeb-1M derivative
``gaunernst/ms1mv3-wds-gz``. The script exports aligned 112x112 JPEG images and
a dense identity remap so the ArcFace trainer can use a plain CSV index.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


DATASET_VIEWER = "https://datasets-server.huggingface.co"
DEFAULT_DATASET = "gaunernst/ms1mv3-wds-gz"


def read_json_url(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "CVProjectStage2Task5/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def dataset_viewer_summary(dataset: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(dataset, safe="")
    summary: dict[str, Any] = {"dataset": dataset}
    for name in ("is-valid", "splits", "size", "parquet"):
        try:
            summary[name] = read_json_url(f"{DATASET_VIEWER}/{name}?dataset={encoded}")
        except Exception as exc:  # noqa: BLE001 - metadata is helpful but not required for export.
            summary[name] = {"error": str(exc)}
    return summary


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("._") or "sample"


def infer_max_images(args: argparse.Namespace) -> int | None:
    if args.max_images is not None:
        return args.max_images if args.max_images > 0 else None
    if args.mode == "full":
        return None
    estimate = int(args.target_hours * 3600 * args.estimated_images_per_second / max(args.epochs_for_estimate, 1))
    return max(args.min_subset_images, estimate)


def load_identity_map(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items()}


def save_identity_map(path: Path, mapping: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_image_to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def save_row_image(value: Any, output_path: Path, size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    image = row_image_to_pil(value)
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    image.save(output_path, format="JPEG", quality=95)


def write_index(index_path: Path, rows: list[dict[str, Any]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "source_cls", "key"])
        writer.writeheader()
        writer.writerows(rows)


def read_existing_index(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_class_distribution(counts: Counter[int], output_path: Path) -> None:
    if not counts:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = list(counts.values())
    plt.figure(figsize=(8, 4.5))
    plt.hist(values, bins=min(40, max(5, int(math.sqrt(len(values))))), color="#2563eb", edgecolor="white")
    plt.title("MS1MV3 Subset Images per Identity")
    plt.xlabel("images per identity")
    plt.ylabel("identity count")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def draw_sample_grid(rows: list[dict[str, Any]], output_path: Path, max_images: int = 16) -> None:
    selected = rows[:max_images]
    if not selected:
        return
    thumbs: list[Image.Image] = []
    font = ImageFont.load_default()
    for row in selected:
        image_path = Path(row["path"])
        with Image.open(image_path).convert("RGB") as image:
            image.thumbnail((128, 128))
            canvas = Image.new("RGB", (140, 164), "white")
            canvas.paste(image, ((140 - image.width) // 2, 4))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 140), f"id {row['label']}", fill=(20, 20, 20), font=font)
            thumbs.append(canvas)

    cols = 4
    rows_count = int(math.ceil(len(thumbs) / cols))
    grid = Image.new("RGB", (cols * 140, rows_count * 164), "white")
    for idx, thumb in enumerate(thumbs):
        grid.paste(thumb, ((idx % cols) * 140, (idx // cols) * 164))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def estimate_train_hours(num_images: int, epochs: int, images_per_second: float) -> float:
    if num_images <= 0 or images_per_second <= 0:
        return 0.0
    return num_images * epochs / images_per_second / 3600


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    raw_dir = data_dir / "raw"
    train_dir = data_dir / "train"
    index_dir = data_dir / "index"
    summaries_dir = report_dir / "summaries"
    dataset_assets = report_dir / "assets" / "dataset"
    for path in (raw_dir, train_dir, index_dir, summaries_dir, dataset_assets):
        path.mkdir(parents=True, exist_ok=True)

    dataset_tag = args.output_tag
    if not dataset_tag:
        dataset_tag = "ms1mv3_dense" if "dense" in data_dir.name.lower() else "ms1mv3_subset"
    max_images = infer_max_images(args)
    metadata = dataset_viewer_summary(args.dataset)
    (raw_dir / "hf_dataset_viewer_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    index_path = index_dir / "train_subset.csv"
    identity_map_path = index_dir / "identity_map.json"
    existing_rows = read_existing_index(index_path)
    existing_keys = {row["key"] for row in existing_rows}
    identity_map = load_identity_map(identity_map_path)
    class_counts: Counter[int] = Counter(int(row["label"]) for row in existing_rows)

    rows = list(existing_rows)
    exported = len(rows)
    skipped_cap = 0
    errors = 0
    stream_retries = 0
    skipped_existing_total = 0
    started = time.time()
    progress_total = max_images if max_images is not None else None
    progress = tqdm(total=progress_total, initial=min(exported, progress_total or exported), desc="export MS1MV3")

    try:
        while max_images is None or exported < max_images:
            stream_retries += 1
            pass_start_exported = exported
            skipped_existing_pass = 0
            print(
                f"Opening Hugging Face stream attempt {stream_retries}/{args.max_stream_retries}; "
                f"already exported {exported} images. Resume must scan earlier samples first.",
                file=sys.stderr,
                flush=True,
            )
            try:
                dataset = load_dataset(args.dataset, split=args.split, streaming=True)
                for item in dataset:
                    if max_images is not None and exported >= max_images:
                        break
                    source_cls = str(item["cls"])
                    if source_cls not in identity_map:
                        if args.max_identities and len(identity_map) >= args.max_identities:
                            skipped_cap += 1
                            if skipped_cap % args.resume_log_interval == 0:
                                progress.set_postfix(skipped_cap=skipped_cap, exported=exported)
                                progress.refresh()
                            continue
                        identity_map[source_cls] = len(identity_map)
                    label = identity_map[source_cls]
                    if args.images_per_identity_cap and class_counts[label] >= args.images_per_identity_cap:
                        skipped_cap += 1
                        if skipped_cap % args.resume_log_interval == 0:
                            progress.set_postfix(skipped_cap=skipped_cap, exported=exported)
                            progress.refresh()
                        continue
                    key = sanitize_name(str(item.get("__key__", f"{source_cls}_{class_counts[label]:06d}")))
                    if key in existing_keys:
                        skipped_existing_pass += 1
                        skipped_existing_total += 1
                        if skipped_existing_pass % args.resume_log_interval == 0:
                            progress.set_postfix(rescan_existing=skipped_existing_pass, exported=exported)
                            progress.refresh()
                        continue
                    rel_path = Path(f"{label:06d}") / f"{key}.jpg"
                    image_path = train_dir / rel_path
                    try:
                        save_row_image(item["jpg"], image_path, args.image_size)
                    except Exception as exc:  # noqa: BLE001 - keep a long export from dying on one bad sample.
                        errors += 1
                        if args.fail_on_image_error:
                            raise RuntimeError(f"Failed to export {key}: {exc}") from exc
                        continue
                    row = {
                        "path": str(image_path),
                        "label": label,
                        "source_cls": source_cls,
                        "key": key,
                    }
                    rows.append(row)
                    existing_keys.add(key)
                    class_counts[label] += 1
                    exported += 1
                    progress.update(1)
                    if exported % args.flush_interval == 0:
                        write_index(index_path, rows)
                        save_identity_map(identity_map_path, identity_map)
                break
            except Exception as exc:  # noqa: BLE001 - network/tar stream can truncate on long runs.
                write_index(index_path, rows)
                save_identity_map(identity_map_path, identity_map)
                if stream_retries >= args.max_stream_retries:
                    raise
                print(
                    f"MS1MV3 stream interrupted after exporting {exported} images "
                    f"({type(exc).__name__}: {exc}). Retrying in {args.retry_sleep}s.",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(args.retry_sleep)
                if exported == pass_start_exported:
                    progress.set_postfix(waiting_for_new_samples=True, exported=exported)
                    progress.refresh()
    finally:
        progress.close()

    write_index(index_path, rows)
    save_identity_map(identity_map_path, identity_map)

    filtered_counts = Counter(int(row["label"]) for row in rows)
    sample_grid_path = dataset_assets / f"{dataset_tag}_samples.png"
    identity_distribution_path = dataset_assets / f"{dataset_tag}_identity_distribution.png"
    plot_class_distribution(filtered_counts, identity_distribution_path)
    draw_sample_grid(rows, sample_grid_path)

    elapsed_hours = (time.time() - started) / 3600
    estimated_hours = estimate_train_hours(len(rows), args.epochs_for_estimate, args.estimated_images_per_second)
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "mode": args.mode,
        "data_root": str(data_dir),
        "raw_dir": str(raw_dir),
        "train_dir": str(train_dir),
        "index": str(index_path),
        "identity_map": str(identity_map_path),
        "requested_max_images": max_images,
        "images": len(rows),
        "identities": len(filtered_counts),
        "min_images_per_identity": min(filtered_counts.values()) if filtered_counts else 0,
        "max_images_per_identity": max(filtered_counts.values()) if filtered_counts else 0,
        "mean_images_per_identity": float(sum(filtered_counts.values()) / len(filtered_counts)) if filtered_counts else 0.0,
        "skipped_by_caps": skipped_cap,
        "skipped_existing_during_resume": skipped_existing_total,
        "image_errors": errors,
        "stream_attempts": stream_retries,
        "export_elapsed_hours": round(elapsed_hours, 4),
        "estimated_train_hours": round(estimated_hours, 2),
        "target_hours": args.target_hours,
        "target_hours_note": (
            "This subset is sized from the requested time budget. If LFW accuracy is below 98.5%, "
            "rerun with a higher --max-images or --mode full and resume training."
        ),
        "dataset_viewer_manifest": str(raw_dir / "hf_dataset_viewer_summary.json"),
        "assets": {
            "sample_grid": str(sample_grid_path),
            "identity_distribution": str(identity_distribution_path),
        },
    }
    summary_path = summaries_dir / f"{dataset_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-dir", default="data/task5_ms1mv3")
    parser.add_argument("--report-dir", default="reports/task5")
    parser.add_argument("--target-hours", type=float, default=7.0)
    parser.add_argument("--mode", choices=["subset", "full"], default="subset")
    parser.add_argument("--max-images", type=int, default=None, help="0 means no image cap.")
    parser.add_argument("--max-identities", type=int, default=50000)
    parser.add_argument("--images-per-identity-cap", type=int, default=30)
    parser.add_argument("--min-subset-images", type=int, default=120000)
    parser.add_argument("--epochs-for-estimate", type=int, default=24)
    parser.add_argument("--estimated-images-per-second", type=float, default=80.0)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--flush-interval", type=int, default=5000)
    parser.add_argument("--resume-log-interval", type=int, default=1000)
    parser.add_argument("--max-stream-retries", type=int, default=8)
    parser.add_argument("--retry-sleep", type=int, default=30)
    parser.add_argument("--output-tag", default="", help="Prefix for summary and dataset asset filenames.")
    parser.add_argument("--fail-on-image-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
