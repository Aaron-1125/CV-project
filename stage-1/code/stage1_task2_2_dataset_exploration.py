#!/usr/bin/env python3
"""Stage 1 task 2.2: explore CelebA and LFW datasets.

The script downloads/loads CelebA and LFW, writes compact statistics, and
saves visualization images under reports/assets. Large data stays under data/.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CELEBA_ATTR_NAMES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_image_grid(samples: list[tuple[Any, str]], output_path: Path, columns: int = 4) -> None:
    if not samples:
        return
    rows = int(np.ceil(len(samples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3.2))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")
    for ax, (image, title) in zip(axes_arr, samples):
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_bar_chart(labels: list[str], values: list[float], title: str, output_path: Path) -> None:
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.28)))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#2563eb")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_celeba_torchvision(data_dir: Path, download: bool):
    from torchvision.datasets import CelebA

    return CelebA(
        root=str(data_dir),
        split="all",
        target_type=["attr", "identity", "bbox", "landmarks"],
        download=download,
    )


def summarize_celeba_torchvision(dataset: Any, assets_dir: Path, sample_count: int) -> dict[str, Any]:
    attrs = np.asarray(getattr(dataset, "attr"))
    identities = np.asarray(getattr(dataset, "identity")).reshape(-1)
    bbox = np.asarray(getattr(dataset, "bbox"))
    landmarks = np.asarray(getattr(dataset, "landmarks"))

    attr_names = getattr(dataset, "attr_names", CELEBA_ATTR_NAMES)
    positive_rates = attrs.mean(axis=0)
    attr_pairs = sorted(
        zip(attr_names, positive_rates, strict=False),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    top_attrs = [(name, round(float(rate), 4)) for name, rate in attr_pairs[:15]]
    bottom_attrs = [(name, round(float(rate), 4)) for name, rate in attr_pairs[-10:]]

    identity_counts = Counter(int(v) for v in identities)
    top_identities = identity_counts.most_common(15)

    rng = random.Random(20260521)
    sample_indices = rng.sample(range(len(dataset)), min(sample_count, len(dataset)))
    samples = []
    for idx in sample_indices:
        image, target = dataset[idx]
        identity = int(np.asarray(target[1]).reshape(-1)[0])
        samples.append((image, f"idx={idx}\nid={identity}"))

    save_image_grid(samples, assets_dir / "celeba_samples.png")
    save_bar_chart(
        [name for name, _ in top_attrs],
        [rate for _, rate in top_attrs],
        "CelebA Top Attribute Positive Rates",
        assets_dir / "celeba_attribute_top15.png",
    )
    save_bar_chart(
        [str(identity) for identity, _ in top_identities],
        [count for _, count in top_identities],
        "CelebA Top Identity Counts",
        assets_dir / "celeba_identity_top15.png",
    )

    return {
        "status": "ok",
        "source": "torchvision.datasets.CelebA",
        "num_images": len(dataset),
        "num_identities": len(identity_counts),
        "attributes": len(attr_names),
        "bbox_shape": list(bbox.shape),
        "landmarks_shape": list(landmarks.shape),
        "top_attributes_positive_rate": top_attrs,
        "bottom_attributes_positive_rate": bottom_attrs,
        "top_identity_counts": top_identities,
        "assets": {
            "samples": "reports/assets/celeba_samples.png",
            "attribute_top15": "reports/assets/celeba_attribute_top15.png",
            "identity_top15": "reports/assets/celeba_identity_top15.png",
        },
    }


def load_celeba_huggingface(data_dir: Path, download: bool):
    if not download:
        raise RuntimeError("Hugging Face CelebA fallback requires --download.")
    from datasets import load_dataset

    return load_dataset(
        "eurecom-ds/celeba",
        split="train+validation+test",
        cache_dir=str(data_dir / "huggingface_cache"),
    )


def summarize_celeba_huggingface(dataset: Any, assets_dir: Path, sample_count: int) -> dict[str, Any]:
    column_names = set(dataset.column_names)
    attr_column = "attributes" if "attributes" in column_names else None
    identity_column = "identity" if "identity" in column_names else None
    image_column = "image" if "image" in column_names else None

    summary: dict[str, Any] = {
        "status": "ok",
        "source": "datasets.load_dataset('eurecom-ds/celeba', split='train+validation+test')",
        "num_images": dataset.num_rows,
        "columns": dataset.column_names,
    }

    if attr_column:
        attrs = np.asarray(dataset[attr_column])
        positive_rates = attrs.mean(axis=0)
        names = CELEBA_ATTR_NAMES[: attrs.shape[1]]
        attr_pairs = sorted(
            zip(names, positive_rates, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        top_attrs = [(name, round(float(rate), 4)) for name, rate in attr_pairs[:15]]
        save_bar_chart(
            [name for name, _ in top_attrs],
            [rate for _, rate in top_attrs],
            "CelebA Top Attribute Positive Rates",
            assets_dir / "celeba_attribute_top15.png",
        )
        summary["attributes"] = attrs.shape[1]
        summary["top_attributes_positive_rate"] = top_attrs
        summary["assets"] = {"attribute_top15": "reports/assets/celeba_attribute_top15.png"}

    if identity_column:
        identities = np.asarray(dataset[identity_column]).reshape(-1)
        counts = Counter(int(v) for v in identities)
        top_identities = counts.most_common(15)
        save_bar_chart(
            [str(identity) for identity, _ in top_identities],
            [count for _, count in top_identities],
            "CelebA Top Identity Counts",
            assets_dir / "celeba_identity_top15.png",
        )
        summary["num_identities"] = len(counts)
        summary["top_identity_counts"] = top_identities
        summary.setdefault("assets", {})["identity_top15"] = "reports/assets/celeba_identity_top15.png"

    if image_column:
        rng = random.Random(20260521)
        sample_indices = rng.sample(range(dataset.num_rows), min(sample_count, dataset.num_rows))
        samples = []
        for idx in sample_indices:
            row = dataset[int(idx)]
            samples.append((row[image_column], f"idx={idx}"))
        save_image_grid(samples, assets_dir / "celeba_samples.png")
        summary.setdefault("assets", {})["samples"] = "reports/assets/celeba_samples.png"

    return summary


def summarize_celeba(data_dir: Path, assets_dir: Path, download: bool, sample_count: int) -> dict[str, Any]:
    errors: list[str] = []
    try:
        dataset = load_celeba_torchvision(data_dir, download)
        return summarize_celeba_torchvision(dataset, assets_dir, sample_count)
    except Exception as exc:  # noqa: BLE001 - report fallback context for reproducibility.
        errors.append(f"torchvision CelebA failed: {exc}")

    try:
        dataset = load_celeba_huggingface(data_dir, download)
        summary = summarize_celeba_huggingface(dataset, assets_dir, sample_count)
        summary["fallback_errors"] = errors
        return summary
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Hugging Face CelebA failed: {exc}")
        return {"status": "failed", "errors": errors}


def summarize_lfw(data_dir: Path, assets_dir: Path, download: bool, sample_count: int) -> dict[str, Any]:
    from sklearn.datasets import fetch_lfw_pairs, fetch_lfw_people

    lfw_home = data_dir / "lfw"
    people = fetch_lfw_people(
        data_home=str(lfw_home),
        color=True,
        resize=0.5,
        download_if_missing=download,
    )
    pairs = fetch_lfw_pairs(
        data_home=str(lfw_home),
        subset="10_folds",
        color=True,
        resize=0.5,
        download_if_missing=download,
    )

    target_counts = Counter(int(v) for v in people.target)
    top_people = [
        (str(people.target_names[target_id]), int(count))
        for target_id, count in target_counts.most_common(15)
    ]
    pair_counts = Counter(int(v) for v in pairs.target)
    pair_target_counts = {
        str(pairs.target_names[target_id]): int(count)
        for target_id, count in sorted(pair_counts.items())
    }

    rng = random.Random(20260521)
    sample_indices = rng.sample(range(len(people.images)), min(sample_count, len(people.images)))
    samples = []
    for idx in sample_indices:
        image = np.clip(people.images[idx], 0, 255).astype(np.uint8)
        samples.append((image, str(people.target_names[int(people.target[idx])])[:32]))

    save_image_grid(samples, assets_dir / "lfw_samples.png")
    save_bar_chart(
        [name for name, _ in top_people],
        [count for _, count in top_people],
        "LFW Top Identity Counts",
        assets_dir / "lfw_identity_top15.png",
    )

    return {
        "status": "ok",
        "source": "sklearn.datasets.fetch_lfw_people/fetch_lfw_pairs",
        "people_images": int(len(people.images)),
        "people_identities": int(len(people.target_names)),
        "pairs": int(len(pairs.pairs)),
        "pair_target_counts": pair_target_counts,
        "top_identity_counts": top_people,
        "image_shape": list(people.images.shape[1:]),
        "pair_image_shape": list(pairs.pairs.shape[2:]),
        "assets": {
            "samples": "reports/assets/lfw_samples.png",
            "identity_top15": "reports/assets/lfw_identity_top15.png",
        },
    }


def write_markdown_report(report_path: Path, summary: dict[str, Any]) -> None:
    celeba = summary["celeba"]
    lfw = summary["lfw"]
    lines = [
        "# Stage 1 Task 2.2 Dataset Exploration Summary",
        "",
        "## CelebA",
        "",
    ]
    if celeba.get("status") == "ok":
        lines.extend(
            [
                f"- Source: `{celeba.get('source')}`",
                f"- Images: `{celeba.get('num_images')}`",
                f"- Identities: `{celeba.get('num_identities', 'n/a')}`",
                f"- Attributes: `{celeba.get('attributes', 'n/a')}`",
                f"- Sample grid: `reports/assets/celeba_samples.png`",
                "",
                "Top attribute positive rates:",
                "",
            ]
        )
        for name, rate in celeba.get("top_attributes_positive_rate", [])[:10]:
            lines.append(f"- `{name}`: `{rate}`")
    else:
        lines.append("- Status: failed")
        for error in celeba.get("errors", []):
            lines.append(f"- {error}")

    lines.extend(["", "## LFW", ""])
    if lfw.get("status") == "ok":
        lines.extend(
            [
                f"- Source: `{lfw.get('source')}`",
                f"- Images: `{lfw.get('people_images')}`",
                f"- Identities: `{lfw.get('people_identities')}`",
                f"- 10-fold pairs: `{lfw.get('pairs')}`",
                f"- Pair target counts: `{lfw.get('pair_target_counts')}`",
                f"- Sample grid: `reports/assets/lfw_samples.png`",
            ]
        )
    else:
        lines.append("- Status: failed")
        for error in lfw.get("errors", []):
            lines.append(f"- {error}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing datasets.")
    parser.add_argument("--data-dir", default="data", help="Ignored directory for raw datasets.")
    parser.add_argument("--report-dir", default="reports", help="Directory for report artifacts.")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any dataset fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    assets_dir = report_dir / "assets"
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "celeba": summarize_celeba(data_dir, assets_dir, args.download, args.sample_count),
        "lfw": {},
    }
    try:
        summary["lfw"] = summarize_lfw(data_dir, assets_dir, args.download, args.sample_count)
    except Exception as exc:  # noqa: BLE001
        summary["lfw"] = {"status": "failed", "errors": [str(exc)]}

    save_json(report_dir / "stage1_task2_2_dataset_summary.json", summary)
    write_markdown_report(report_dir / "stage1_task2_2_dataset_summary.md", summary)
    print(f"Wrote {report_dir / 'stage1_task2_2_dataset_summary.json'}")
    print(f"Wrote {report_dir / 'stage1_task2_2_dataset_summary.md'}")

    if args.strict and any(part.get("status") != "ok" for part in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
