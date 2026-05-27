#!/usr/bin/env python3
"""Prepare LFW deep-funneled images and the 6000-pair verification protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


LFW_IMAGE_URLS = [
    "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
    "https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
]
LFW_PAIR_URLS = [
    "http://vis-www.cs.umass.edu/lfw/pairs.txt",
    "https://vis-www.cs.umass.edu/lfw/pairs.txt",
]
HF_LFW_REPO = "DerrickUnleashed/LFW"


def download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "CVProjectStage2Task5/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        with output_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def download_first(urls: list[str], output_path: Path) -> bool:
    errors = []
    for url in urls:
        try:
            download(url, output_path)
            if output_path.exists() and output_path.stat().st_size > 0:
                return True
        except Exception as exc:  # noqa: BLE001 - retry alternate mirrors.
            output_path.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
    if errors:
        print("Direct LFW download failed:")
        for error in errors:
            print(f"  {error}")
    return False


def hf_download(filename: str, output_path: Path) -> None:
    from huggingface_hub import hf_hub_download

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    cached = hf_hub_download(repo_id=HF_LFW_REPO, repo_type="dataset", filename=filename)
    shutil.copy2(cached, output_path)


def ensure_lfw_archive(raw_dir: Path) -> Path:
    archive_path = raw_dir / "lfw-deepfunneled.tgz"
    if download_first(LFW_IMAGE_URLS, archive_path):
        return archive_path
    mirror_path = raw_dir / "lfw-deepfunneled.zip"
    print(f"Falling back to Hugging Face mirror: {HF_LFW_REPO}/lfw-deepfunneled.zip")
    hf_download("lfw-deepfunneled.zip", mirror_path)
    return mirror_path


def convert_hf_pairs_csv(source_csv: Path, output_txt: Path) -> None:
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        rows = []
        for row in reader:
            fields = [field.strip() for field in row if field.strip()]
            if len(fields) not in (3, 4):
                raise ValueError(f"Invalid Hugging Face LFW pair row: {row}")
            rows.append("\t".join(fields))
    output_txt.write_text("10\t300\n" + "\n".join(rows) + "\n", encoding="utf-8")


def ensure_pairs_file(data_dir: Path, raw_dir: Path) -> Path:
    pairs_path = data_dir / "pairs.txt"
    if download_first(LFW_PAIR_URLS, pairs_path):
        return pairs_path
    mirror_pairs = raw_dir / "pairs.csv"
    print(f"Falling back to Hugging Face mirror: {HF_LFW_REPO}/pairs.csv")
    hf_download("pairs.csv", mirror_pairs)
    convert_hf_pairs_csv(mirror_pairs, pairs_path)
    return pairs_path


def extract_lfw(archive_path: Path, data_dir: Path) -> Path:
    image_root = data_dir / "lfw-deepfunneled"
    if image_root.exists() and any(image_root.iterdir()):
        return image_root
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_dir)
    else:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(data_dir)
    return image_root


def lfw_image_path(image_root: Path, person: str, index: int) -> Path:
    return image_root / person / f"{person}_{index:04d}.jpg"


def parse_pairs(pairs_path: Path, image_root: Path) -> list[dict[str, Any]]:
    lines = [line.strip() for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs: list[dict[str, Any]] = []
    for fold_idx, line in enumerate(lines[1:]):
        parts = line.split()
        if len(parts) == 3:
            person, idx1, idx2 = parts
            path1 = lfw_image_path(image_root, person, int(idx1))
            path2 = lfw_image_path(image_root, person, int(idx2))
            same = True
        elif len(parts) == 4:
            person1, idx1, person2, idx2 = parts
            path1 = lfw_image_path(image_root, person1, int(idx1))
            path2 = lfw_image_path(image_root, person2, int(idx2))
            same = False
        else:
            raise ValueError(f"Invalid LFW pair line: {line}")
        pairs.append(
            {
                "fold": fold_idx // 600,
                "path1": str(path1),
                "path2": str(path2),
                "same": same,
                "raw": line,
            }
        )
    return pairs


def write_pairs_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["fold,path1,path2,same,raw"]
    for item in pairs:
        raw = str(item["raw"]).replace('"', '""')
        lines.append(f"{item['fold']},{item['path1']},{item['path2']},{int(item['same'])},\"{raw}\"")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_sample_pairs(pairs: list[dict[str, Any]], output_path: Path, max_pairs: int = 8) -> None:
    selected = pairs[:max_pairs]
    if not selected:
        return
    thumbs = []
    font = ImageFont.load_default()
    for item in selected:
        left = Image.open(item["path1"]).convert("RGB")
        right = Image.open(item["path2"]).convert("RGB")
        left.thumbnail((112, 112))
        right.thumbnail((112, 112))
        canvas = Image.new("RGB", (244, 148), "white")
        canvas.paste(left, (4, 4))
        canvas.paste(right, (128, 4))
        draw = ImageDraw.Draw(canvas)
        label = "same" if item["same"] else "different"
        draw.text((6, 126), f"fold {item['fold']} | {label}", fill=(20, 20, 20), font=font)
        thumbs.append(canvas)

    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    grid = Image.new("RGB", (cols * 244, rows * 148), "white")
    for idx, thumb in enumerate(thumbs):
        grid.paste(thumb, ((idx % cols) * 244, (idx // cols) * 148))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    raw_dir = data_dir / "raw"
    summaries_dir = report_dir / "summaries"
    dataset_assets = report_dir / "assets" / "dataset"
    for path in (raw_dir, summaries_dir, dataset_assets):
        path.mkdir(parents=True, exist_ok=True)

    archive_path = ensure_lfw_archive(raw_dir)
    pairs_path = ensure_pairs_file(data_dir, raw_dir)
    image_root = extract_lfw(archive_path, data_dir)

    pairs = parse_pairs(pairs_path, image_root)
    missing = [item for item in pairs if not Path(item["path1"]).exists() or not Path(item["path2"]).exists()]
    pairs_csv = data_dir / "pairs.csv"
    write_pairs_csv(pairs_csv, pairs)
    draw_sample_pairs(pairs, dataset_assets / "lfw_protocol_samples.png")

    positives = sum(1 for item in pairs if item["same"])
    negatives = len(pairs) - positives
    summary = {
        "data_root": str(data_dir),
        "image_root": str(image_root),
        "pairs_txt": str(pairs_path),
        "pairs_csv": str(pairs_csv),
        "pairs": len(pairs),
        "positive_pairs": positives,
        "negative_pairs": negatives,
        "folds": sorted({item["fold"] for item in pairs}),
        "missing_pairs": len(missing),
        "ready": len(pairs) == 6000 and positives == 3000 and negatives == 3000 and not missing,
        "assets": {
            "sample_pairs": str(dataset_assets / "lfw_protocol_samples.png"),
        },
    }
    summary_path = summaries_dir / "lfw_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    if not summary["ready"]:
        raise SystemExit("LFW preparation finished but the 6000-pair protocol is incomplete; see summary.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/task5_lfw")
    parser.add_argument("--report-dir", default="reports/task5")
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
