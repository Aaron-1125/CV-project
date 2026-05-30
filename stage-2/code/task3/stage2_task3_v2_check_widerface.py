#!/usr/bin/env python3
"""Check the full WIDER FACE data used by Task3 v2."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_task3_3_evaluate_widerface import read_image_ids, read_xml_record  # noqa: E402


def safe_read_record(data_root: Path, split: str, image_id: str) -> tuple[dict[str, Any] | None, str]:
    try:
        record = read_xml_record(data_root, split, image_id)
    except (FileNotFoundError, ET.ParseError, OSError) as exc:
        return None, str(exc)
    if not record.image_path.exists():
        return None, f"missing image: {record.image_path}"
    return {
        "image_id": record.image_id,
        "image_path": str(record.image_path),
        "faces": len(record.boxes),
    }, ""


def summarize_split(data_root: Path, split: str, ann_file: str) -> dict[str, Any]:
    ann_path = data_root / ann_file
    image_ids = read_image_ids(ann_path)
    missing: list[dict[str, str]] = []
    total_faces = 0
    zero_face_images = 0
    for image_id in image_ids:
        row, error = safe_read_record(data_root, split, image_id)
        if error:
            missing.append({"image_id": image_id, "error": error})
            continue
        assert row is not None
        faces = int(row["faces"])
        total_faces += faces
        if faces == 0:
            zero_face_images += 1
    return {
        "ann_file": str(ann_path),
        "split": split,
        "images": len(image_ids),
        "faces": total_faces,
        "zero_face_images": zero_face_images,
        "missing_count": len(missing),
        "missing_examples": missing[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/WIDERFace")
    parser.add_argument("--summary-out", default="reports/task3_v2/summaries/widerface_v2_data_check.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    summary = {
        "data_root": str(data_root),
        "train": summarize_split(data_root, "train", "train.txt"),
        "val": summarize_split(data_root, "val", "val.txt"),
        "expected": {"train_images": 12337, "val_images": 3079},
    }
    summary["matches_expected_counts"] = (
        summary["train"]["images"] == summary["expected"]["train_images"]
        and summary["val"]["images"] == summary["expected"]["val_images"]
        and summary["train"]["missing_count"] == 0
        and summary["val"]["missing_count"] == 0
    )
    output_path = Path(args.summary_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
