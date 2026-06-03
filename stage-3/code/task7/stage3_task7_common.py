#!/usr/bin/env python3
"""Shared helpers for Stage3 Task7 StarGAN attribute editing."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import runpy
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SELECTED_ATTRS = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
HAIR_ATTRS = ["Black_Hair", "Blond_Hair", "Brown_Hair"]
OFFICIAL_STARGAN_REF = "94dd002e93a2863d9b987a937b85925b80f7a19f"


def stage3_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cv_project_root() -> Path:
    return stage3_root().parent


def task_root() -> Path:
    return cv_project_root().parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_stage3_path(config_path)
    namespace = runpy.run_path(str(path))
    return {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__")
    }


def resolve_stage3_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return stage3_root() / path


def cfg_get(cfg: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = cfg.get(section, {})
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def cfg_path(cfg: dict[str, Any], section: str, key: str, default: str) -> Path:
    return resolve_stage3_path(str(cfg_get(cfg, section, key, default)))


def report_dir(cfg: dict[str, Any]) -> Path:
    return cfg_path(cfg, "reports", "report_dir", "reports/task7")


def summary_dir(cfg: dict[str, Any]) -> Path:
    return cfg_path(cfg, "reports", "summary_dir", "reports/task7/summaries")


def asset_dir(cfg: dict[str, Any]) -> Path:
    return cfg_path(cfg, "reports", "asset_dir", "reports/task7/assets")


def work_dir(cfg: dict[str, Any]) -> Path:
    return cfg_path(cfg, "train", "work_dir", "work_dirs/task7/stargan_celeba_128_full")


def selected_attrs(cfg: dict[str, Any]) -> list[str]:
    attrs = list(cfg_get(cfg, "model", "selected_attrs", SELECTED_ATTRS))
    validate_selected_attrs(attrs, int(cfg_get(cfg, "model", "c_dim", len(attrs))))
    return attrs


def validate_selected_attrs(attrs: list[str], c_dim: int) -> None:
    if attrs != SELECTED_ATTRS:
        raise ValueError(f"Task7 expects selected_attrs={SELECTED_ATTRS}, got {attrs}")
    if c_dim != len(SELECTED_ATTRS):
        raise ValueError(f"Task7 expects c_dim={len(SELECTED_ATTRS)}, got {c_dim}")


def stargan_repo(cfg: dict[str, Any]) -> Path:
    override = os.environ.get("STARGAN_REPO")
    configured = cfg_get(cfg, "stargan", "repo_path", str(task_root() / "StarGAN"))
    return Path(override or configured).expanduser().resolve()


def ensure_stargan_repo(cfg: dict[str, Any], check_ref: bool = True) -> dict[str, Any]:
    repo = stargan_repo(cfg)
    if not (repo / "main.py").exists() or not (repo / ".git").exists():
        raise FileNotFoundError(
            f"Missing official StarGAN repo at {repo}. Clone yunjey/StarGAN there first."
        )
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    commit = result.stdout.strip()
    expected = str(cfg_get(cfg, "stargan", "ref", OFFICIAL_STARGAN_REF))
    if check_ref and commit != expected:
        raise RuntimeError(f"StarGAN repo is at {commit}, expected {expected}")
    return {
        "repo_url": cfg_get(cfg, "stargan", "repo_url", "https://github.com/yunjey/StarGAN.git"),
        "path": str(repo),
        "commit": commit,
        "expected_commit": expected,
    }


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_command(cmd: list[str], cwd: Path, summary_out: Path | None = None) -> dict[str, Any]:
    started = time.time()
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), text=True)
    summary = {
        "command": cmd,
        "cwd": str(cwd),
        "returncode": result.returncode,
        "seconds": round(time.time() - started, 2),
    }
    if summary_out:
        write_json(summary_out, summary)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return summary


@dataclass
class CelebARow:
    filename: str
    attrs: dict[str, bool]
    original_index: int
    shuffled_rank: int | None = None


def find_celeba_paths(celeba_root: Path) -> tuple[Path, Path]:
    root = celeba_root.expanduser().resolve()
    attr_candidates = [
        root / "list_attr_celeba.txt",
        root / "celeba" / "list_attr_celeba.txt",
        root / "Anno" / "list_attr_celeba.txt",
    ]
    image_candidates = [
        root / "images",
        root / "img_align_celeba",
        root / "celeba" / "images",
        root / "Img" / "img_align_celeba",
    ]
    attr_path = next((p for p in attr_candidates if p.exists()), None)
    image_dir = next((p for p in image_candidates if p.exists() and p.is_dir()), None)
    if attr_path is None:
        raise FileNotFoundError(f"Could not find list_attr_celeba.txt under {root}")
    if image_dir is None:
        raise FileNotFoundError(f"Could not find CelebA image directory under {root}")
    return image_dir, attr_path


def parse_attr_file(attr_path: Path) -> tuple[list[str], list[CelebARow]]:
    lines = [line.strip() for line in attr_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Invalid CelebA attr file: {attr_path}")
    all_attrs = lines[1].split()
    missing = [name for name in SELECTED_ATTRS if name not in all_attrs]
    if missing:
        raise ValueError(f"CelebA attr file misses required attrs: {missing}")
    rows: list[CelebARow] = []
    for idx, line in enumerate(lines[2:]):
        parts = line.split()
        filename = parts[0]
        values = parts[1:]
        if len(values) != len(all_attrs):
            raise ValueError(f"Malformed CelebA attr row for {filename}")
        rows.append(
            CelebARow(
                filename=filename,
                attrs={name: values[attr_idx] == "1" for attr_idx, name in enumerate(all_attrs)},
                original_index=idx,
            )
        )
    return all_attrs, rows


def official_stargan_split(rows: list[CelebARow]) -> tuple[list[CelebARow], list[CelebARow]]:
    shuffled = list(rows)
    random.Random(1234).shuffle(shuffled)
    for rank, row in enumerate(shuffled):
        row.shuffled_rank = rank
    if len(shuffled) < 3:
        return shuffled, shuffled
    test_count = 1999 if len(shuffled) >= 2000 else max(1, len(shuffled) // 4)
    test_rows = shuffled[:test_count]
    train_rows = shuffled[test_count:]
    return train_rows, test_rows


def selected_label(row: CelebARow, attrs: list[str]) -> list[int]:
    return [1 if row.attrs[name] else 0 for name in attrs]


def hair_sum_from_label(label: list[int], attrs: list[str]) -> int:
    return sum(label[attrs.index(name)] for name in HAIR_ATTRS)


def build_target_labels(label: list[int], attrs: list[str]) -> list[dict[str, Any]]:
    validate_selected_attrs(attrs, len(label))
    if hair_sum_from_label(label, attrs) > 1:
        raise ValueError(f"Original label has conflicting hair attrs: {label}")
    targets: list[dict[str, Any]] = []
    hair_indices = [attrs.index(name) for name in HAIR_ATTRS]
    for attr_idx, attr_name in enumerate(attrs):
        target = list(label)
        if attr_name in HAIR_ATTRS:
            for hair_idx in hair_indices:
                target[hair_idx] = 0
            target[attr_idx] = 1
        else:
            target[attr_idx] = 0 if target[attr_idx] else 1
        validate_target_label(target, attrs, direction=attr_name)
        targets.append(
            {
                "direction": attr_name,
                "label": target,
                "attrs": {name: bool(target[idx]) for idx, name in enumerate(attrs)},
            }
        )
    return targets


def validate_target_label(label: list[int], attrs: list[str], direction: str) -> None:
    validate_selected_attrs(attrs, len(label))
    hair_sum = hair_sum_from_label(label, attrs)
    if direction in HAIR_ATTRS and hair_sum != 1:
        raise ValueError(f"Hair edit {direction} must set exactly one hair attr, got {label}")
    if direction not in HAIR_ATTRS and hair_sum > 1:
        raise ValueError(f"Non-hair edit {direction} retains conflicting hair attrs: {label}")


def write_target_labels(samples: list[dict[str, Any]], attrs: list[str], output_base: Path) -> dict[str, str]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        label = sample["label"]
        targets = build_target_labels(label, attrs)
        for target in targets:
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "filename": sample["filename"],
                    "direction": target["direction"],
                    **{f"target_{name}": int(target["attrs"][name]) for name in attrs},
                }
            )
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")
    write_json(json_path, {"selected_attrs": attrs, "records": records})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["sample_id", "filename", "direction"] + [f"target_{name}" for name in attrs]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {csv_path}")
    return {"json": str(json_path), "csv": str(csv_path)}


def select_fixed_samples(rows: list[CelebARow], attrs: list[str], image_dir: Path, count: int) -> list[dict[str, Any]]:
    usable = [
        row for row in rows
        if (image_dir / row.filename).exists() and hair_sum_from_label(selected_label(row, attrs), attrs) <= 1
    ]
    if not usable:
        raise ValueError("No usable CelebA fixed samples found.")
    selected: list[CelebARow] = []
    used: set[str] = set()
    targets = []
    for hair in HAIR_ATTRS:
        for male in [False, True]:
            for young in [False, True]:
                targets.append((hair, male, young))
    for hair, male, young in targets:
        if len(selected) >= count:
            break
        match = next(
            (
                row for row in usable
                if row.filename not in used
                and row.attrs.get(hair, False)
                and row.attrs.get("Male", False) == male
                and row.attrs.get("Young", False) == young
            ),
            None,
        )
        if match:
            selected.append(match)
            used.add(match.filename)
    for row in usable:
        if len(selected) >= count:
            break
        if row.filename not in used:
            selected.append(row)
            used.add(row.filename)
    samples: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        label = selected_label(row, attrs)
        samples.append(
            {
                "sample_id": f"fixed_{idx:02d}",
                "filename": row.filename,
                "source_path": str(image_dir / row.filename),
                "original_index": row.original_index,
                "official_test_rank": row.shuffled_rank,
                "attrs": {name: bool(row.attrs[name]) for name in attrs},
                "label": label,
            }
        )
    return samples


def prepared_image_dir(cfg: dict[str, Any]) -> Path:
    return cfg_path(cfg, "data", "image_dir", "data/celeba/images")


def prepared_attr_path(cfg: dict[str, Any]) -> Path:
    return cfg_path(cfg, "data", "attr_path", "data/celeba/list_attr_celeba.txt")


def fixed_manifest_path(cfg: dict[str, Any]) -> Path:
    return summary_dir(cfg) / "fixed_samples_manifest.json"


def load_fixed_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    path = fixed_manifest_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed sample manifest: {path}")
    return read_json(path)


def copy_or_symlink(source: Path, target: Path, force: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and Path(os.readlink(target)) == source:
            return
        if not force:
            raise FileExistsError(f"{target} already exists. Pass --force to replace it.")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    os.symlink(source, target, target_is_directory=source.is_dir())


def save_source_grid(samples: list[dict[str, Any]], output_path: Path, size: int = 128) -> None:
    images = []
    for sample in samples:
        with Image.open(sample["source_path"]) as handle:
            img = handle.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        images.append((sample["sample_id"], img))
    save_labeled_grid(images, ["source"], output_path, columns=4, cell_size=size)


def save_labeled_grid(items: list[tuple[str, Image.Image]], headers: list[str], output_path: Path, columns: int, cell_size: int) -> None:
    font = ImageFont.load_default()
    header_h = 22
    label_h = 18
    rows = int(np.ceil(len(items) / columns))
    canvas = Image.new("RGB", (columns * cell_size, rows * (cell_size + header_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(items):
        row = idx // columns
        col = idx % columns
        x = col * cell_size
        y = row * (cell_size + header_h + label_h)
        draw.text((x + 4, y + 3), label[:28], fill=(20, 20, 20), font=font)
        canvas.paste(image.resize((cell_size, cell_size), Image.Resampling.BICUBIC), (x, y + header_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    print(f"Wrote {output_path}")


def import_stargan_generator(repo: Path):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model import Generator  # type: ignore

    return Generator


def load_generator(cfg: dict[str, Any], checkpoint: Path, device: str):
    import torch

    repo = stargan_repo(cfg)
    Generator = import_stargan_generator(repo)
    generator = Generator(
        int(cfg_get(cfg, "model", "g_conv_dim", 64)),
        int(cfg_get(cfg, "model", "c_dim", 5)),
        int(cfg_get(cfg, "model", "g_repeat_num", 6)),
    )
    try:
        state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint), map_location=device)
    generator.load_state_dict(state)
    generator.to(device)
    generator.eval()
    return generator


def image_transform(cfg: dict[str, Any], train: bool = False):
    from torchvision import transforms as T

    ops: list[Any] = []
    if train:
        ops.append(T.RandomHorizontalFlip())
    ops.extend(
        [
            T.CenterCrop(int(cfg_get(cfg, "model", "celeba_crop_size", 178))),
            T.Resize(int(cfg_get(cfg, "model", "image_size", 128))),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return T.Compose(ops)


def denorm_tensor(tensor):
    return tensor.add(1).div(2).clamp(0, 1)


def tensor_to_pil(tensor) -> Image.Image:
    import torch

    array = denorm_tensor(tensor.detach().cpu()).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def generate_fixed_samples(
    cfg: dict[str, Any],
    checkpoint: Path,
    run_name: str,
    output_dir: Path,
    device: str | None = None,
) -> dict[str, Any]:
    import torch

    attrs = selected_attrs(cfg)
    device = device or str(cfg_get(cfg, "evaluation", "device", "cuda:0"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    manifest = load_fixed_manifest(cfg)
    samples = manifest["samples"]
    image_dir = Path(manifest["image_dir"])
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing StarGAN generator checkpoint: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)
    label_paths = write_target_labels(samples, attrs, output_dir / f"{run_name}_target_labels")

    transform = image_transform(cfg, train=False)
    generator = load_generator(cfg, checkpoint, device)

    source_tensors = []
    source_images: list[Image.Image] = []
    for sample in samples:
        with Image.open(image_dir / sample["filename"]) as handle:
            image = handle.convert("RGB")
        source_tensors.append(transform(image))
        source_images.append(image.resize((int(cfg_get(cfg, "model", "image_size", 128)),) * 2, Image.Resampling.BICUBIC))
    x_real = torch.stack(source_tensors).to(device)

    columns = ["source"] + attrs
    cell = int(cfg_get(cfg, "model", "image_size", 128))
    header_h = 24
    row_label_w = 72
    grid = Image.new("RGB", (row_label_w + len(columns) * cell, header_h + len(samples) * cell), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    for col_idx, title in enumerate(columns):
        draw.text((row_label_w + col_idx * cell + 4, 6), title, fill=(20, 20, 20), font=font)

    generated_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample_idx, sample in enumerate(samples):
            y = header_h + sample_idx * cell
            draw.text((4, y + 6), sample["sample_id"], fill=(20, 20, 20), font=font)
            grid.paste(source_images[sample_idx], (row_label_w, y))
            targets = build_target_labels(sample["label"], attrs)
            for target_idx, target in enumerate(targets):
                label_tensor = torch.tensor([target["label"]], dtype=torch.float32, device=device)
                fake = generator(x_real[sample_idx : sample_idx + 1], label_tensor)[0]
                fake_image = tensor_to_pil(fake)
                direction = target["direction"]
                image_path = output_dir / "images" / f"{sample['sample_id']}_{direction}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                fake_image.save(image_path, quality=95)
                grid.paste(fake_image, (row_label_w + (target_idx + 1) * cell, y))
                generated_records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "filename": sample["filename"],
                        "direction": direction,
                        "image_path": str(image_path),
                        "target_label": target["label"],
                    }
                )
    grid_path = output_dir / f"{run_name}_fixed_grid.jpg"
    grid.save(grid_path, quality=95)
    summary = {
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "grid": str(grid_path),
        "target_labels": label_paths,
        "generated": generated_records,
        "samples": len(samples),
        "directions": attrs,
        "device": device,
    }
    write_json(output_dir / f"{run_name}_fixed_summary.json", summary)
    return summary


def generate_eval_images(
    cfg: dict[str, Any],
    checkpoint: Path,
    output_dir: Path,
    limit: int,
    device: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    import torch

    attrs = selected_attrs(cfg)
    device = device or str(cfg_get(cfg, "evaluation", "device", "cuda:0"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    batch_size = batch_size or int(cfg_get(cfg, "evaluation", "batch_size", 64))
    image_dir = prepared_image_dir(cfg)
    attr_path = prepared_attr_path(cfg)
    _, rows = parse_attr_file(attr_path)
    _, test_rows = official_stargan_split(rows)
    usable = [
        row for row in test_rows
        if (image_dir / row.filename).exists() and hair_sum_from_label(selected_label(row, attrs), attrs) <= 1
    ][:limit]
    if not usable:
        raise ValueError("No usable evaluation images found.")
    eval_samples = [
        {
            "sample_id": f"eval_{idx:04d}",
            "filename": row.filename,
            "source_path": str(image_dir / row.filename),
            "original_index": row.original_index,
            "official_test_rank": row.shuffled_rank,
            "attrs": {name: bool(row.attrs[name]) for name in attrs},
            "label": selected_label(row, attrs),
        }
        for idx, row in enumerate(usable)
    ]
    target_label_paths = write_target_labels(eval_samples, attrs, output_dir / "eval_target_labels")
    transform = image_transform(cfg, train=False)
    generator = load_generator(cfg, checkpoint, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_dir = output_dir / "real"
    fake_root = output_dir / "generated"
    real_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(usable), batch_size):
            batch_rows = usable[start : start + batch_size]
            tensors = []
            for row in batch_rows:
                with Image.open(image_dir / row.filename) as handle:
                    image = handle.convert("RGB")
                tensors.append(transform(image))
                real_path = real_dir / row.filename
                if not real_path.exists():
                    image.resize((int(cfg_get(cfg, "model", "image_size", 128)),) * 2, Image.Resampling.BICUBIC).save(real_path, quality=95)
            x_real = torch.stack(tensors).to(device)
            for direction in attrs:
                direction_dir = fake_root / direction
                direction_dir.mkdir(parents=True, exist_ok=True)
                labels = []
                for row in batch_rows:
                    label = selected_label(row, attrs)
                    target = next(item for item in build_target_labels(label, attrs) if item["direction"] == direction)
                    labels.append(target["label"])
                label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
                fake_batch = generator(x_real, label_tensor)
                for idx, row in enumerate(batch_rows):
                    fake_path = direction_dir / row.filename
                    tensor_to_pil(fake_batch[idx]).save(fake_path, quality=95)
                    records.append(
                        {
                            "filename": row.filename,
                            "source_path": str(real_dir / row.filename),
                            "generated_path": str(fake_path),
                            "direction": direction,
                            "target_label": labels[idx],
                            "source_label": selected_label(row, attrs),
                        }
                    )
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "image_dir": str(image_dir),
        "attr_path": str(attr_path),
        "output_dir": str(output_dir),
        "images": len(usable),
        "directions": attrs,
        "target_labels": target_label_paths,
        "records": records,
    }
    write_json(output_dir / "eval_generation_manifest.json", manifest)
    return manifest


def checkpoint_path_for_iters(cfg: dict[str, Any], iters: int) -> Path:
    return work_dir(cfg) / "models" / f"{iters}-G.ckpt"


def model_dir(cfg: dict[str, Any]) -> Path:
    return work_dir(cfg) / "models"


def sorted_generator_checkpoints(cfg: dict[str, Any]) -> list[Path]:
    paths = list(model_dir(cfg).glob("*-G.ckpt"))
    def iter_num(path: Path) -> int:
        try:
            return int(path.name.split("-")[0])
        except ValueError:
            return -1
    return sorted(paths, key=iter_num)
