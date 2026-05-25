#!/usr/bin/env python3
"""Prepare the 300W face landmark dataset for Stage2 task 4.x.

The OpenMMLab annotation archive is directly downloadable. The official 300W
image archive is gated by an iBUG download form in some sessions, so this
script first tries the official URLs and then prints deterministic manual
instructions if the response is an HTML form instead of a zip part.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Any


ANNOTATION_URL = "https://download.openmmlab.com/mmpose/datasets/300w_annotations.tar"
RAW_PARTS = [f"300w.zip.{idx:03d}" for idx in range(1, 5)]
RAW_URLS = {
    part: f"https://ibug.doc.ic.ac.uk/download/annotations/{part}"
    for part in RAW_PARTS
}
EXPECTED_ANNOTATION_COUNTS = {
    "face_landmarks_300w_train.json": 3148,
    "face_landmarks_300w_valid.json": 689,
    "face_landmarks_300w_valid_common.json": 554,
    "face_landmarks_300w_valid_challenge.json": 135,
    "face_landmarks_300w_test.json": 600,
}
EXPECTED_IMAGE_DIRS = ["afw", "helen", "ibug", "lfpw", "Test"]
CORE_IMAGE_DIRS = ["afw", "helen", "ibug", "lfpw"]
CORE_ANNOTATIONS = {
    "face_landmarks_300w_train.json",
    "face_landmarks_300w_valid.json",
    "face_landmarks_300w_valid_common.json",
    "face_landmarks_300w_valid_challenge.json",
}


def download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; CVProjectStage2Task4/1.0)",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with output_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def looks_like_html(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return True
    head = path.read_bytes()[:4096].lower()
    return b"<html" in head or b"download form" in head or b"first name" in head


def manual_download_message(raw_dir: Path) -> str:
    parts = "\n".join(f"  - {part}" for part in RAW_PARTS)
    return (
        "Could not automatically download the official 300W image archive. "
        "The iBUG site returned a download form instead of zip bytes.\n\n"
        "Please open https://ibug.doc.ic.ac.uk/resources/300-W/ in a browser, "
        "download the four official parts, and place them here:\n"
        f"  {raw_dir.resolve()}\n\n"
        f"Required files:\n{parts}\n\n"
        "Then rerun this command. The script will extract the multipart zip "
        "with 7z and keep all task4 outputs under stage-2/data/task4_300w and "
        "stage-2/reports/task4."
    )


def find_unpacked_image_root(raw_dir: Path) -> Path | None:
    roots = [raw_dir] + [path for path in raw_dir.iterdir() if path.is_dir()]
    for root in roots:
        if all((root / name).is_dir() for name in CORE_IMAGE_DIRS):
            return root
    return None


def download_annotations(raw_dir: Path) -> Path:
    archive = raw_dir / "300w_annotations.tar"
    if not archive.exists():
        print(f"Downloading OpenMMLab annotations: {ANNOTATION_URL}")
        download(ANNOTATION_URL, archive)
    return archive


def extract_annotations(archive: Path, mmpose_root: Path) -> None:
    extract_dir = archive.parent / "annotations_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(extract_dir)

    ann_dir = mmpose_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED_ANNOTATION_COUNTS:
        matches = list(extract_dir.rglob(name))
        if not matches:
            raise FileNotFoundError(f"Missing {name} inside {archive}")
        shutil.copy2(matches[0], ann_dir / name)


def ensure_raw_image_parts(raw_dir: Path, download_images: bool) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    unpacked_root = find_unpacked_image_root(raw_dir)
    if unpacked_root:
        print(f"Using unpacked 300W image root: {unpacked_root}")
        return

    missing = [part for part in RAW_PARTS if not (raw_dir / part).exists()]
    if missing and download_images:
        for part in missing:
            target = raw_dir / part
            print(f"Trying official iBUG download: {RAW_URLS[part]}")
            try:
                download(RAW_URLS[part], target)
            except Exception as exc:  # noqa: BLE001 - re-raised with user instructions.
                if target.exists():
                    target.unlink()
                raise SystemExit(manual_download_message(raw_dir) + f"\n\nDownload error: {exc}") from exc
            if looks_like_html(target):
                target.unlink(missing_ok=True)
                raise SystemExit(manual_download_message(raw_dir))

    missing = [part for part in RAW_PARTS if not (raw_dir / part).exists()]
    if missing:
        raise SystemExit(manual_download_message(raw_dir))

    invalid = [part for part in RAW_PARTS if looks_like_html(raw_dir / part)]
    if invalid:
        raise SystemExit(
            manual_download_message(raw_dir)
            + "\n\nThese files do not look like zip parts: "
            + ", ".join(invalid)
        )


def extract_images(raw_dir: Path, extracted_dir: Path) -> None:
    if any(extracted_dir.rglob("afw")) and any(extracted_dir.rglob("helen")):
        return
    seven_zip = shutil.which("7z") or shutil.which("7za")
    if not seven_zip:
        raise SystemExit("7z/7za was not found. Install p7zip-full in the container and rerun.")
    extracted_dir.mkdir(parents=True, exist_ok=True)
    command = [seven_zip, "x", str(raw_dir / RAW_PARTS[0]), f"-o{extracted_dir}", "-y"]
    subprocess.run(command, check=True)


def find_dir(root: Path, name: str) -> Path:
    candidates = [path for path in root.rglob(name) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Could not find extracted image directory named {name!r} under {root}")
    return sorted(candidates, key=lambda path: len(path.parts))[0]


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source.resolve(), target, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)


def normalize_images(source_root: Path, mmpose_root: Path) -> None:
    image_root = mmpose_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED_IMAGE_DIRS:
        try:
            source = find_dir(source_root, name)
        except FileNotFoundError:
            if name == "Test":
                print("Optional official 300W Test image directory was not found; train/valid splits can still run.")
                continue
            raise
        link_or_copy(source, image_root / name)


def count_annotations(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "images": len(data.get("images", [])),
        "annotations": len(data.get("annotations", [])),
    }


def count_missing_images(annotation_path: Path, image_root: Path) -> int:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    missing = 0
    for image in data.get("images", []):
        if not (image_root / image["file_name"]).exists():
            missing += 1
    return missing


def write_summary(data_dir: Path, report_dir: Path) -> dict[str, Any]:
    mmpose_root = data_dir / "mmpose" / "300w"
    ann_dir = mmpose_root / "annotations"
    image_root = mmpose_root / "images"
    annotation_counts: dict[str, Any] = {}
    for name, expected in EXPECTED_ANNOTATION_COUNTS.items():
        path = ann_dir / name
        counts = count_annotations(path)
        counts["expected_images"] = expected
        counts["missing_images"] = count_missing_images(path, image_root)
        counts["ok"] = counts["images"] == expected and counts["missing_images"] == 0
        annotation_counts[name] = counts

    core_ready = all(
        annotation_counts[name]["ok"]
        for name in CORE_ANNOTATIONS
    )
    all_splits_ready = all(item["ok"] for item in annotation_counts.values())
    summary = {
        "data_root": str(mmpose_root),
        "raw_dir": str(data_dir / "raw"),
        "annotation_counts": annotation_counts,
        "image_dirs": {
            name: str((image_root / name).resolve()) if (image_root / name).exists() else ""
            for name in EXPECTED_IMAGE_DIRS
        },
        "ready": core_ready,
        "all_splits_ready": all_splits_ready,
        "test_split_note": (
            ""
            if all_splits_ready
            else "The Kaggle ibug_300W_large_face_landmark_dataset commonly omits the official Test/ images. "
            "Training and validation are ready when ready=true; official test NME needs Test/ images."
        ),
    }
    out_path = report_dir / "summaries" / "300w_dataset_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download annotations and try official image parts.")
    parser.add_argument("--data-dir", default="data/task4_300w")
    parser.add_argument("--report-dir", default="reports/task4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    raw_dir = data_dir / "raw"
    mmpose_root = data_dir / "mmpose" / "300w"

    if args.download:
        archive = download_annotations(raw_dir)
        extract_annotations(archive, mmpose_root)

    ensure_raw_image_parts(raw_dir, download_images=args.download)
    unpacked_root = find_unpacked_image_root(raw_dir)
    if unpacked_root:
        normalize_images(unpacked_root, mmpose_root)
    else:
        extract_images(raw_dir, data_dir / "extracted")
        normalize_images(data_dir / "extracted", mmpose_root)
    summary = write_summary(data_dir, report_dir)
    if not summary["ready"]:
        raise SystemExit("300W preparation finished but required train/valid images are missing; see the dataset summary.")
    print("300W is ready for Stage2 task 4.x.")


if __name__ == "__main__":
    main()
