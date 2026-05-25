#!/usr/bin/env python3
"""Prepare WIDER FACE for stage-2 task 3.x MMDetection training.

The script uses torchvision.datasets.WIDERFace as the default downloader and
converts train/val annotations into the PASCAL VOC-style layout expected by
MMDetection's WIDERFaceDataset. It also writes deterministic smoke subset
lists without copying the original dataset images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import WIDERFace
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_file_from_google_drive,
    extract_archive,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
WIDER_TRAIN_VAL_FILES = [
    ("15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M", "3fedf70df600953d25982bcd13d91ba2", "WIDER_train.zip"),
    ("1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q", "dfa7d7e790efa35df3788964cf0bbaea", "WIDER_val.zip"),
]
WIDER_ANNOTATIONS_FILE = (
    "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip",
    "0e3767bcf0e326556d407bf5bff5d27c",
    "wider_face_split.zip",
)


def rel_symlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def to_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def valid_boxes(target: dict[str, Any]) -> list[list[int]]:
    bboxes = target["bbox"]
    invalid = target.get("invalid")
    boxes: list[list[int]] = []
    for idx, bbox in enumerate(bboxes):
        x, y, w, h = [to_int(v) for v in bbox]
        is_invalid = bool(to_int(invalid[idx])) if invalid is not None and idx < len(invalid) else False
        if is_invalid or w <= 0 or h <= 0:
            continue
        boxes.append([x, y, x + w, y + h])
    return boxes


def xml_node(parent: ET.Element, name: str, text: str | int) -> ET.Element:
    node = ET.SubElement(parent, name)
    node.text = str(text)
    return node


def write_voc_xml(xml_path: Path, folder: str, filename: str, width: int, height: int, boxes: list[list[int]]) -> None:
    annotation = ET.Element("annotation")
    xml_node(annotation, "folder", folder)
    xml_node(annotation, "filename", filename)
    size = ET.SubElement(annotation, "size")
    xml_node(size, "width", width)
    xml_node(size, "height", height)
    xml_node(size, "depth", 3)

    for x1, y1, x2, y2 in boxes:
        obj = ET.SubElement(annotation, "object")
        xml_node(obj, "name", "face")
        xml_node(obj, "pose", "Unspecified")
        xml_node(obj, "truncated", 0)
        xml_node(obj, "difficult", 0)
        bndbox = ET.SubElement(obj, "bndbox")
        xml_node(bndbox, "xmin", max(0, x1))
        xml_node(bndbox, "ymin", max(0, y1))
        xml_node(bndbox, "xmax", min(width - 1, x2))
        xml_node(bndbox, "ymax", min(height - 1, y2))

    ET.indent(annotation, space="  ")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(annotation).write(xml_path, encoding="utf-8", xml_declaration=True)


def stable_subset(ids: list[str], count: int, seed: str) -> list[str]:
    if count <= 0 or count >= len(ids):
        return ids
    return sorted(
        sorted(ids, key=lambda item: hashlib.sha1(f"{seed}:{item}".encode("utf-8")).hexdigest())[:count]
    )


def ensure_widerface_train_val_download(root: Path) -> None:
    target_root = root / WIDERFace.BASE_FOLDER
    target_root.mkdir(parents=True, exist_ok=True)
    for file_id, md5, filename in WIDER_TRAIN_VAL_FILES:
        extracted_dir = target_root / Path(filename).stem
        if extracted_dir.exists():
            continue
        download_google_drive_zip(file_id, md5, target_root / filename)
        extract_archive(str(target_root / filename))
    if not (target_root / "wider_face_split").exists():
        download_and_extract_archive(
            url=WIDER_ANNOTATIONS_FILE[0],
            download_root=str(target_root),
            md5=WIDER_ANNOTATIONS_FILE[1],
            filename=WIDER_ANNOTATIONS_FILE[2],
        )
    # Torchvision's integrity check expects the test split directory to exist
    # even when only train/val are needed for this task. Keep it empty so we do
    # not download public test images that are outside the stage-2 scope.
    (target_root / "WIDER_test").mkdir(exist_ok=True)


def download_google_drive_zip(file_id: str, md5: str, output_path: Path) -> None:
    if check_integrity(str(output_path), md5):
        return
    output_path.unlink(missing_ok=True)
    try:
        download_file_from_google_drive(file_id, str(output_path.parent), output_path.name, md5)
        return
    except RuntimeError:
        output_path.unlink(missing_ok=True)

    import gdown

    result = gdown.download(id=file_id, output=str(output_path), quiet=False)
    if not result or not check_integrity(str(output_path), md5):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download valid Google Drive archive: {output_path.name}")


def load_split(root: Path, split: str) -> WIDERFace:
    return WIDERFace(root=str(root), split=split, download=False)


def convert_split(dataset: WIDERFace, mmdet_root: Path, split: str) -> dict[str, Any]:
    split_dir_name = f"WIDER_{split}"
    source_images = Path(dataset.root) / split_dir_name / "images"
    target_split_root = mmdet_root / split_dir_name
    target_images = target_split_root / "images"
    target_annotations = target_split_root / "Annotations"
    rel_symlink_or_copy(source_images, target_images)
    for event_dir in sorted(source_images.iterdir()):
        if event_dir.is_dir():
            rel_symlink_or_copy(event_dir, target_split_root / event_dir.name)
    target_annotations.mkdir(parents=True, exist_ok=True)

    image_ids: list[str] = []
    face_counts: list[int] = []
    event_counts: Counter[str] = Counter()
    areas: list[int] = []
    invalid_faces = 0

    for item in dataset.img_info:
        image_path = Path(str(item["img_path"]))
        image_id = image_path.stem
        event_name = image_path.parent.name
        boxes = valid_boxes(item["annotations"])  # type: ignore[arg-type]
        raw_box_count = len(item["annotations"]["bbox"])  # type: ignore[index]
        invalid_faces += max(0, raw_box_count - len(boxes))
        with Image.open(image_path) as img:
            width, height = img.size
        write_voc_xml(
            target_annotations / f"{image_id}.xml",
            folder=event_name,
            filename=f"{image_id}.jpg",
            width=width,
            height=height,
            boxes=boxes,
        )
        if boxes:
            image_ids.append(image_id)
            face_counts.append(len(boxes))
            event_counts[event_name] += 1
            areas.extend((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)

    (mmdet_root / f"{split}.txt").write_text("\n".join(sorted(image_ids)) + "\n", encoding="utf-8")
    return {
        "split": split,
        "images": len(image_ids),
        "faces": int(sum(face_counts)),
        "invalid_or_empty_faces": int(invalid_faces),
        "mean_faces_per_image": float(np.mean(face_counts)) if face_counts else 0.0,
        "median_faces_per_image": float(np.median(face_counts)) if face_counts else 0.0,
        "top_events": event_counts.most_common(12),
        "face_area_p50": float(np.percentile(areas, 50)) if areas else 0.0,
        "face_area_p90": float(np.percentile(areas, 90)) if areas else 0.0,
        "ids": sorted(image_ids),
    }


def read_xml_boxes(xml_path: Path) -> list[list[int]]:
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        boxes.append(
            [
                int(float(box.findtext("xmin", "0"))),
                int(float(box.findtext("ymin", "0"))),
                int(float(box.findtext("xmax", "0"))),
                int(float(box.findtext("ymax", "0"))),
            ]
        )
    return boxes


def draw_sample_grid(mmdet_root: Path, ids: list[str], split: str, output_path: Path, max_images: int = 12) -> None:
    selected = stable_subset(ids, min(max_images, len(ids)), f"sample-{split}")
    thumbs = []
    for image_id in selected:
        xml_path = mmdet_root / f"WIDER_{split}" / "Annotations" / f"{image_id}.xml"
        folder = ET.parse(xml_path).getroot().findtext("folder", "")
        image_path = mmdet_root / f"WIDER_{split}" / "images" / folder / f"{image_id}.jpg"
        with Image.open(image_path).convert("RGB") as img:
            original_width, original_height = img.size
            img.thumbnail((220, 160))
            canvas = Image.new("RGB", (220, 190), "white")
            x = (220 - img.width) // 2
            canvas.paste(img, (x, 0))
            scale_x = img.width / original_width
            scale_y = img.height / original_height
            draw = ImageDraw.Draw(canvas)
            for x1, y1, x2, y2 in read_xml_boxes(xml_path)[:20]:
                draw.rectangle(
                    [x + x1 * scale_x, y1 * scale_y, x + x2 * scale_x, y2 * scale_y],
                    outline=(0, 180, 0),
                    width=2,
                )
            draw.text((6, 170), image_id[:28], fill=(20, 20, 20), font=ImageFont.load_default())
            thumbs.append(canvas)

    cols = 3
    rows = int(np.ceil(len(thumbs) / cols)) if thumbs else 1
    grid = Image.new("RGB", (cols * 220, rows * 190), "white")
    for idx, thumb in enumerate(thumbs):
        grid.paste(thumb, ((idx % cols) * 220, (idx // cols) * 190))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def plot_distribution(values: list[float], title: str, xlabel: str, output_path: Path, bins: int = 30) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.hist(values, bins=bins, color="#2563eb", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def export_public_val_inputs(mmdet_root: Path, val_ids: list[str], output_dir: Path, count: int = 4) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for legacy in output_dir.glob("wider_val_public_*.jpg"):
        legacy.unlink()
    exported: list[str] = []
    for idx, image_id in enumerate(val_ids[: min(count, len(val_ids))]):
        xml_path = mmdet_root / "WIDER_val" / "Annotations" / f"{image_id}.xml"
        folder = ET.parse(xml_path).getroot().findtext("folder", "")
        source = mmdet_root / "WIDER_val" / "images" / folder / f"{image_id}.jpg"
        dest = output_dir / f"input_{idx:02d}_{image_id}.jpg"
        shutil.copy2(source, dest)
        exported.append(str(dest))
    return exported


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    summaries_dir = report_dir / "summaries"
    dataset_assets = report_dir / "assets" / "dataset"
    input_assets = report_dir / "assets" / "inputs" / "wider_val"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        ensure_widerface_train_val_download(data_dir)
    train_set = load_split(data_dir, "train")
    val_set = load_split(data_dir, "val")
    mmdet_root = data_dir / "WIDERFace"
    mmdet_root.mkdir(parents=True, exist_ok=True)

    train_summary = convert_split(train_set, mmdet_root, "train")
    val_summary = convert_split(val_set, mmdet_root, "val")

    smoke_train = stable_subset(train_summary["ids"], args.smoke_train, "stage2-smoke-train")
    smoke_val = stable_subset(val_summary["ids"], args.smoke_val, "stage2-smoke-val")
    (mmdet_root / "smoke_train.txt").write_text("\n".join(smoke_train) + "\n", encoding="utf-8")
    (mmdet_root / "smoke_val.txt").write_text("\n".join(smoke_val) + "\n", encoding="utf-8")

    train_face_counts = [len(read_xml_boxes(mmdet_root / "WIDER_train" / "Annotations" / f"{i}.xml")) for i in smoke_train]
    val_face_counts = [len(read_xml_boxes(mmdet_root / "WIDER_val" / "Annotations" / f"{i}.xml")) for i in smoke_val]
    plot_distribution(
        train_face_counts,
        "WIDER FACE Smoke Train Faces per Image",
        "faces per image",
        dataset_assets / "widerface_smoke_train_faces_per_image.png",
        bins=20,
    )
    plot_distribution(
        val_face_counts,
        "WIDER FACE Smoke Val Faces per Image",
        "faces per image",
        dataset_assets / "widerface_smoke_val_faces_per_image.png",
        bins=20,
    )
    draw_sample_grid(mmdet_root, smoke_val, "val", dataset_assets / "widerface_val_samples_with_boxes.png")
    exported_inputs = export_public_val_inputs(mmdet_root, smoke_val, input_assets, args.export_val_images)

    for summary in (train_summary, val_summary):
        summary.pop("ids", None)
    summary = {
        "data_root": str(mmdet_root),
        "source_root": str(Path(train_set.root)),
        "train": train_summary,
        "val": val_summary,
        "smoke_train_images": len(smoke_train),
        "smoke_train_faces": int(sum(train_face_counts)),
        "smoke_val_images": len(smoke_val),
        "smoke_val_faces": int(sum(val_face_counts)),
        "smoke_train_ann_file": str(mmdet_root / "smoke_train.txt"),
        "smoke_val_ann_file": str(mmdet_root / "smoke_val.txt"),
        "exported_public_val_inputs": exported_inputs,
        "assets": {
            "sample_grid": str(dataset_assets / "widerface_val_samples_with_boxes.png"),
            "smoke_train_faces_per_image": str(dataset_assets / "widerface_smoke_train_faces_per_image.png"),
            "smoke_val_faces_per_image": str(dataset_assets / "widerface_smoke_val_faces_per_image.png"),
        },
    }
    summary_path = summaries_dir / "widerface_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download WIDER FACE through torchvision/gdown.")
    parser.add_argument("--data-dir", default="data", help="Stage-local ignored data directory.")
    parser.add_argument("--report-dir", default="reports", help="Stage-local report directory.")
    parser.add_argument("--smoke-train", type=int, default=128)
    parser.add_argument("--smoke-val", type=int, default=64)
    parser.add_argument("--export-val-images", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
