#!/usr/bin/env python3
"""Prepare official InsightFace-compatible full MS1MV3 RecordIO data.

The preferred source is the Hugging Face mirror ``gaunernst/ms1mv3-recordio``.
The script creates this layout:

``data/task5_ms1mv3_full_recordio/ms1m-retinaface-t1/{train.rec,train.idx,property,lfw.bin}``

For validation, prefer an official or already aligned 112x112 InsightFace
``lfw.bin``. A bin generated from raw/deepfunneled 250x250 LFW images is useful
only as a smoke check because it is not the standard ArcFace validation target.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_DATASET = "gaunernst/ms1mv3-recordio"
EXPECTED_NUM_CLASSES = 93431
EXPECTED_NUM_IMAGES = 5179510
EXPECTED_IMAGE_SIZE = (112, 112)
TARGET_SUBDIR = "ms1m-retinaface-t1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def safe_file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def size_key(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


def inspect_lfw_bin(path: Path, sample_limit: int = 12000) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {"path": str(path), "exists": False, "aligned_112x112": False}

    try:
        with path.open("rb") as handle:
            try:
                bins, issame_list = pickle.load(handle)
            except UnicodeDecodeError:
                handle.seek(0)
                bins, issame_list = pickle.load(handle, encoding="bytes")
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "readable": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "aligned_112x112": False,
        }

    counts: Counter[str] = Counter()
    decode_errors: list[str] = []
    inspected = min(len(bins), sample_limit)
    for raw in bins[:inspected]:
        try:
            with Image.open(BytesIO(raw)) as image:
                counts[size_key(image.size)] += 1
        except Exception as exc:  # noqa: BLE001
            if len(decode_errors) < 5:
                decode_errors.append(f"{exc.__class__.__name__}: {exc}")

    expected = size_key(EXPECTED_IMAGE_SIZE)
    aligned = bool(inspected > 0 and counts.get(expected, 0) == inspected and not decode_errors)
    return {
        "path": str(path),
        "exists": True,
        "readable": True,
        "size_bytes": path.stat().st_size,
        "pairs": len(issame_list),
        "images": len(bins),
        "positive_pairs": sum(1 for item in issame_list if bool(item)),
        "negative_pairs": sum(1 for item in issame_list if not bool(item)),
        "inspected_images": inspected,
        "image_size_counts": dict(sorted(counts.items())),
        "decode_errors": decode_errors,
        "expected_image_size": expected,
        "aligned_112x112": aligned,
    }


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_pair_label(value: str) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "same"}:
        return True
    if normalized in {"0", "false", "no", "diff", "different"}:
        return False
    return None


def parse_pairs_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "path1": row["path1"],
                    "path2": row["path2"],
                    "same": parse_bool(row["same"]),
                    "fold": int(row.get("fold", 0)),
                }
            )
    return rows


def lfw_image_path(image_root: Path, person: str, index: int) -> Path:
    return image_root / person / f"{person}_{index:04d}.jpg"


def parse_pairs_txt(pairs_path: Path, image_root: Path) -> list[dict[str, Any]]:
    lines = [line.strip() for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for pair_idx, line in enumerate(lines[1:]):
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
            raise ValueError(f"Invalid LFW pair row: {line}")
        rows.append({"path1": str(path1), "path2": str(path2), "same": same, "fold": pair_idx // 600})
    return rows


def parse_yakhyo_ann(path: Path) -> list[dict[str, Any]]:
    """Parse Kaggle yakhyo validation annotations.

    The file layout is:
    ``same relative/path1.jpg relative/path2.jpg``. Some archives include a
    one-line header and some do not, so the parser detects that automatically.
    """
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for line_idx, line in enumerate(lines):
        parts = line.split()
        label = parse_pair_label(parts[0]) if len(parts) == 3 else None
        if label is None:
            if line_idx == 0:
                continue
            raise ValueError(f"Invalid LFW annotation row in {path}: {line}")
        _, path1, path2 = parts
        rows.append({"path1": path1, "path2": path2, "same": label, "fold": len(rows) // 600})
    return rows


def load_lfw_pairs(lfw_dir: Path) -> list[dict[str, Any]]:
    pairs_csv = lfw_dir / "pairs.csv"
    if pairs_csv.exists():
        return parse_pairs_csv(pairs_csv)
    for ann_path in (lfw_dir / "lfw_ann.txt", lfw_dir / "val" / "lfw_ann.txt"):
        if ann_path.exists():
            return parse_yakhyo_ann(ann_path)
    pairs_txt = lfw_dir / "pairs.txt"
    image_root = lfw_dir / "lfw-deepfunneled"
    if pairs_txt.exists() and image_root.exists():
        return parse_pairs_txt(pairs_txt, image_root)
    raise FileNotFoundError(
        f"Missing LFW protocol under {lfw_dir}. Run stage2_task5_3_prepare_lfw.py first."
    )


def resolve_lfw_path(value: str, lfw_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path

    lfw_relative = lfw_dir / path
    if lfw_relative.exists():
        return lfw_relative

    lfw_val_relative = lfw_dir / "val" / path
    if lfw_val_relative.exists():
        return lfw_val_relative

    stage2_candidate = lfw_dir.parent.parent / path
    if stage2_candidate.exists():
        return stage2_candidate

    normalized = value.replace("\\", "/")
    marker = "task5_lfw/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        lfw_candidate = lfw_dir / Path(suffix)
        if lfw_candidate.exists():
            return lfw_candidate
    return path


def create_lfw_bin(lfw_dir: Path, output_path: Path, overwrite: bool = False) -> dict[str, Any]:
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return {"path": str(output_path), "created": False, "pairs": 6000, "size_bytes": output_path.stat().st_size}

    pairs = load_lfw_pairs(lfw_dir)
    bins: list[bytes] = []
    issame_list: list[bool] = []
    missing: list[str] = []
    for item in pairs:
        path1 = resolve_lfw_path(item["path1"], lfw_dir)
        path2 = resolve_lfw_path(item["path2"], lfw_dir)
        pair_missing = False
        if not path1.exists():
            missing.append(str(path1))
            pair_missing = True
        if not path2.exists():
            missing.append(str(path2))
            pair_missing = True
        if pair_missing:
            continue
        bins.append(path1.read_bytes())
        bins.append(path2.read_bytes())
        issame_list.append(bool(item["same"]))
    if missing:
        sample = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing LFW images while creating lfw.bin:\n{sample}")
    if len(issame_list) != 6000:
        raise ValueError(f"Expected 6000 LFW pairs, found {len(issame_list)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump((bins, issame_list), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "path": str(output_path),
        "created": True,
        "source": "task5_lfw_pairs_raw_images",
        "pairs": len(issame_list),
        "positive_pairs": sum(1 for item in issame_list if item),
        "negative_pairs": sum(1 for item in issame_list if not item),
        "size_bytes": output_path.stat().st_size,
    }


def copy_lfw_bin(source_path: Path, output_path: Path, overwrite: bool = False) -> dict[str, Any]:
    if not source_path.exists() or source_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing lfw.bin source: {source_path}")
    if output_path.exists() and output_path.resolve() == source_path.resolve():
        return {
            "path": str(output_path),
            "created": False,
            "source": "provided_lfw_bin",
            "source_path": str(source_path),
            "size_bytes": output_path.stat().st_size,
        }
    if output_path.exists() and not overwrite:
        return {
            "path": str(output_path),
            "created": False,
            "source": "existing_lfw_bin",
            "source_path": str(output_path),
            "size_bytes": output_path.stat().st_size,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    return {
        "path": str(output_path),
        "created": True,
        "source": "provided_lfw_bin",
        "source_path": str(source_path),
        "size_bytes": output_path.stat().st_size,
    }


def build_aligned_image_index(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Missing aligned LFW root: {root}")
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem, path)
    if not index:
        raise FileNotFoundError(f"No aligned LFW images found under {root}")
    return index


def pair_image_stem(value: str) -> str:
    normalized = value.replace("\\", "/")
    return Path(normalized).stem


def encode_aligned_image(path: Path) -> bytes:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != EXPECTED_IMAGE_SIZE:
            image = image.resize(EXPECTED_IMAGE_SIZE, Image.BILINEAR)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def create_lfw_bin_from_aligned_root(
    aligned_root: Path,
    lfw_dir: Path,
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return {
            "path": str(output_path),
            "created": False,
            "source": "existing_lfw_bin",
            "size_bytes": output_path.stat().st_size,
        }

    pairs = load_lfw_pairs(lfw_dir)
    image_index = build_aligned_image_index(aligned_root)
    bins: list[bytes] = []
    issame_list: list[bool] = []
    missing: list[str] = []
    for item in pairs:
        stem1 = pair_image_stem(item["path1"])
        stem2 = pair_image_stem(item["path2"])
        path1 = image_index.get(stem1)
        path2 = image_index.get(stem2)
        if path1 is None:
            missing.append(stem1)
        if path2 is None:
            missing.append(stem2)
        if path1 is None or path2 is None:
            continue
        bins.append(encode_aligned_image(path1))
        bins.append(encode_aligned_image(path2))
        issame_list.append(bool(item["same"]))
    if missing:
        sample = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing aligned LFW images by stem:\n{sample}")
    if len(issame_list) != 6000:
        raise ValueError(f"Expected 6000 LFW pairs, found {len(issame_list)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump((bins, issame_list), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "path": str(output_path),
        "created": True,
        "source": "aligned_lfw_root",
        "aligned_root": str(aligned_root),
        "pairs": len(issame_list),
        "positive_pairs": sum(1 for item in issame_list if item),
        "negative_pairs": sum(1 for item in issame_list if not item),
        "size_bytes": output_path.stat().st_size,
    }


def download_recordio(args: argparse.Namespace, rec_dir: Path) -> dict[str, Any]:
    if not args.download:
        return {"download": False, "message": "Skipped download; validating local files only."}

    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    allow_patterns = ["train.rec", "train.idx", "README.md", "*.json"]
    started = time.time()
    snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(rec_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    return {
        "download": True,
        "dataset": args.dataset,
        "revision": args.revision,
        "seconds": round(time.time() - started, 2),
        "allow_patterns": allow_patterns,
    }


def ensure_property(rec_dir: Path) -> Path:
    property_path = rec_dir / "property"
    content = f"{EXPECTED_NUM_CLASSES},{EXPECTED_IMAGE_SIZE[0]},{EXPECTED_IMAGE_SIZE[1]}\n"
    if not property_path.exists() or property_path.read_text(encoding="utf-8", errors="ignore") != content:
        property_path.write_text(content, encoding="utf-8")
    return property_path


def maybe_import_mxnet() -> tuple[str, str | None]:
    try:
        import numpy as np

        if not hasattr(np, "bool"):
            np.bool = bool  # type: ignore[attr-defined]
        import mxnet as mx  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return "unavailable", f"{exc.__class__.__name__}: {exc}"
    return "available", None


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    rec_dir = data_dir / TARGET_SUBDIR
    summaries_dir = report_dir / "summaries"
    rec_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    download_info = download_recordio(args, rec_dir)
    property_path = ensure_property(rec_dir)
    target_lfw_bin = rec_dir / "lfw.bin"
    if args.lfw_bin_path:
        lfw_bin_info = copy_lfw_bin(Path(args.lfw_bin_path), target_lfw_bin, overwrite=args.overwrite_lfw_bin)
    elif args.aligned_lfw_root:
        lfw_bin_info = create_lfw_bin_from_aligned_root(
            Path(args.aligned_lfw_root),
            Path(args.lfw_dir),
            target_lfw_bin,
            overwrite=args.overwrite_lfw_bin,
        )
    else:
        lfw_bin_info = create_lfw_bin(Path(args.lfw_dir), target_lfw_bin, overwrite=args.overwrite_lfw_bin)
    lfw_bin_inspection = inspect_lfw_bin(target_lfw_bin)
    mxnet_status, mxnet_error = maybe_import_mxnet()

    files = {
        name: {
            "path": str(rec_dir / name),
            "exists": (rec_dir / name).exists(),
            "size_bytes": safe_file_size(rec_dir / name),
        }
        for name in ("train.rec", "train.idx", "property", "lfw.bin")
    }
    train_ready = all(files[name]["exists"] and files[name]["size_bytes"] > 0 for name in ("train.rec", "train.idx", "property"))
    validation_ready = bool(
        files["lfw.bin"]["exists"]
        and files["lfw.bin"]["size_bytes"] > 0
        and lfw_bin_inspection.get("aligned_112x112")
    )
    ready = bool(train_ready and (validation_ready or args.allow_unaligned_lfw_bin))
    validation_warning = None
    if not validation_ready:
        validation_warning = (
            "lfw.bin is not confirmed as 112x112 aligned. Official InsightFace LFW numbers from this bin "
            "should be treated as a smoke check, not the 98.5% acceptance metric."
        )
    summary = {
        "dataset": args.dataset,
        "revision": args.revision,
        "data_root": str(data_dir),
        "recordio_dir": str(rec_dir),
        "expected_num_classes": EXPECTED_NUM_CLASSES,
        "expected_num_images": EXPECTED_NUM_IMAGES,
        "expected_image_size": list(EXPECTED_IMAGE_SIZE),
        "property": property_path.read_text(encoding="utf-8").strip(),
        "files": files,
        "lfw_bin": lfw_bin_info,
        "lfw_bin_inspection": lfw_bin_inspection,
        "train_ready": train_ready,
        "validation_ready": validation_ready,
        "validation_warning": validation_warning,
        "allow_unaligned_lfw_bin": bool(args.allow_unaligned_lfw_bin),
        "mxnet_recordio_reader": mxnet_status,
        "mxnet_error": mxnet_error,
        "download": download_info,
        "ready": ready,
        "note": "Full MS1MV3 RecordIO and model checkpoints are local ignored artifacts and must not be committed.",
    }
    write_json(summaries_dir / "ms1mv3_full_recordio_summary.json", summary)
    if not ready:
        missing = [name for name in ("train.rec", "train.idx", "property") if not files[name]["exists"] or files[name]["size_bytes"] <= 0]
        if missing:
            raise SystemExit(f"MS1MV3 RecordIO is not ready; missing or empty: {missing}")
        if validation_warning:
            print(f"WARNING: {validation_warning}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-dir", default="data/task5_ms1mv3_full_recordio")
    parser.add_argument("--report-dir", default="reports/task5")
    parser.add_argument("--lfw-dir", default="data/task5_lfw")
    parser.add_argument("--lfw-bin-path", default=None, help="Copy an existing official/aligned InsightFace lfw.bin.")
    parser.add_argument("--aligned-lfw-root", default=None, help="Build lfw.bin from 112x112 aligned LFW images.")
    parser.add_argument("--download", action="store_true", help="Download train.rec/train.idx from Hugging Face.")
    parser.add_argument("--overwrite-lfw-bin", action="store_true")
    parser.add_argument("--allow-unaligned-lfw-bin", action="store_true", help="Allow a non-112x112 lfw.bin to count as ready.")
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
