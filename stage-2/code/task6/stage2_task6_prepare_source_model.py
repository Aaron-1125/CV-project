#!/usr/bin/env python3
"""Extract the Task5 first-version cloud checkpoint for Stage2 task 6.x."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


TASK5_BEST = "work_dirs/task5/resnet50_arcface_ms1mv3_dense/best.pth"
TASK5_LAST = "work_dirs/task5/resnet50_arcface_ms1mv3_dense/last.pth"
TASK5_LFW_SUMMARY = "reports/task5/summaries/lfw_eval_summary.json"
TASK5_TRAIN_SUMMARY = "reports/task5/summaries/ms1mv3_dense_train_summary.json"
TASK5_ASSETS = [
    "reports/task5/assets/evaluation/lfw_roc_curve.png",
    "reports/task5/assets/evaluation/lfw_similarity_histogram.png",
    "reports/task5/assets/training/ms1mv3_dense_loss_acc_curve.png",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def extract_member(tar: tarfile.TarFile, member_name: str, output_path: Path) -> bool:
    try:
        member = tar.getmember(member_name)
    except KeyError:
        return False
    if not member.isfile():
        return False
    handle = tar.extractfile(member)
    if handle is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as dst:
        dst.write(handle.read())
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-archive", type=Path, default=Path("reports/task5/task5_cloud_results_8167.tar.gz"))
    parser.add_argument("--out-dir", type=Path, default=Path("work_dirs/task6/source_arcface_8167"))
    parser.add_argument("--source-report-dir", type=Path, default=Path("reports/task6/source_task5"))
    parser.add_argument("--summary-out", type=Path, default=Path("reports/task6/summaries/source_model_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.cloud_archive.exists():
        raise FileNotFoundError(f"Missing Task5 cloud result archive: {args.cloud_archive}")

    extracted: dict[str, str] = {}
    with tarfile.open(args.cloud_archive, "r:gz") as tar:
        names = set(tar.getnames())
        required = [TASK5_BEST, TASK5_LFW_SUMMARY, TASK5_TRAIN_SUMMARY]
        missing = [name for name in required if name not in names]
        if missing:
            raise FileNotFoundError(f"Archive is missing required entries: {missing}")

        best_out = args.out_dir / "best.pth"
        extract_member(tar, TASK5_BEST, best_out)
        extracted[TASK5_BEST] = str(best_out)

        last_out = args.out_dir / "last.pth"
        if extract_member(tar, TASK5_LAST, last_out):
            extracted[TASK5_LAST] = str(last_out)

        lfw_summary_out = args.source_report_dir / "lfw_eval_summary_8167.json"
        train_summary_out = args.source_report_dir / "ms1mv3_dense_train_summary_8167.json"
        extract_member(tar, TASK5_LFW_SUMMARY, lfw_summary_out)
        extract_member(tar, TASK5_TRAIN_SUMMARY, train_summary_out)
        extracted[TASK5_LFW_SUMMARY] = str(lfw_summary_out)
        extracted[TASK5_TRAIN_SUMMARY] = str(train_summary_out)

        for asset in TASK5_ASSETS:
            asset_out = args.source_report_dir / "assets" / Path(asset).name
            if extract_member(tar, asset, asset_out):
                extracted[asset] = str(asset_out)

    lfw_summary = read_json(Path(extracted[TASK5_LFW_SUMMARY]))
    train_summary = read_json(Path(extracted[TASK5_TRAIN_SUMMARY]))
    metrics = lfw_summary.get("metrics", {})
    summary = {
        "source": "Task5 first-version self-contained IResNet50 + ArcFace cloud run",
        "cloud_archive": str(args.cloud_archive),
        "checkpoint": str(args.out_dir / "best.pth"),
        "checkpoint_sha256": sha256(args.out_dir / "best.pth"),
        "checkpoint_size_mb": (args.out_dir / "best.pth").stat().st_size / (1024.0 * 1024.0),
        "source_lfw_accuracy": metrics.get("accuracy", lfw_summary.get("accuracy")),
        "source_roc_auc": metrics.get("roc_auc"),
        "source_pairs": metrics.get("pairs"),
        "source_train_images": train_summary.get("num_images"),
        "source_train_identities": train_summary.get("num_identities"),
        "source_epochs_completed": train_summary.get("epochs_completed"),
        "source_best_lfw_accuracy": train_summary.get("best_lfw_accuracy"),
        "extracted": extracted,
        "note": "This copies only the first-version Task5 cloud checkpoint and summaries into Task6 paths; Task5 work_dirs are not overwritten.",
    }
    write_json(args.summary_out, summary)


if __name__ == "__main__":
    main()
