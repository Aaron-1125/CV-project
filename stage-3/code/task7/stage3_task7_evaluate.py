#!/usr/bin/env python3
"""Evaluate Stage3 Task7 StarGAN with attribute success, identity retention, FID, and IS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from stage3_task7_common import (
    HAIR_ATTRS,
    asset_dir,
    build_target_labels,
    cfg_get,
    checkpoint_path_for_iters,
    generate_eval_images,
    load_config,
    official_stargan_split,
    parse_attr_file,
    prepared_attr_path,
    prepared_image_dir,
    selected_attrs,
    selected_label,
    summary_dir,
    write_json,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except ModuleNotFoundError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = object  # type: ignore
    models = None  # type: ignore
    transforms = None  # type: ignore


def require_torch() -> None:
    if torch is None or nn is None or DataLoader is None or models is None or transforms is None:
        raise ModuleNotFoundError(
            "stage3_task7_evaluate.py requires torch and torchvision. "
            "Run it inside the project deep-learning environment or install Task7 requirements."
        )


class CelebAAttrDataset(Dataset):
    def __init__(self, rows, image_dir: Path, attrs: list[str], image_size: int) -> None:
        self.rows = rows
        self.image_dir = image_dir
        self.attrs = attrs
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(self.image_dir / row.filename) as handle:
            image = handle.convert("RGB")
        label = torch.tensor(selected_label(row, self.attrs), dtype=torch.float32)
        return self.transform(image), label


class GeneratedImageDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int) -> None:
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as handle:
            image = handle.convert("RGB")
        return self.transform(image), str(self.paths[idx])


def build_classifier(cfg: dict[str, Any]) -> nn.Module:
    require_torch()
    weights = None
    if bool(cfg_get(cfg, "evaluation", "attribute_classifier_pretrained", False)):
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except Exception:
            weights = None
    try:
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    return model


def train_or_load_classifier(cfg: dict[str, Any], device: str, force_train: bool = False) -> tuple[nn.Module, dict[str, Any]]:
    require_torch()
    attrs = selected_attrs(cfg)
    ckpt_dir = Path(cfg_get(cfg, "evaluation", "attribute_classifier_dir", "work_dirs/task7/attribute_classifier"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path(__file__).resolve().parents[2] / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "resnet18_5attrs.pt"
    summary_path = summary_dir(cfg) / "attribute_classifier_summary.json"
    model = build_classifier(cfg).to(device)
    if ckpt_path.exists() and not force_train:
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        summary = state.get("summary", {"checkpoint": str(ckpt_path), "loaded": True})
        write_json(summary_path, summary)
        return model, summary

    image_dir = prepared_image_dir(cfg)
    attr_path = prepared_attr_path(cfg)
    _, rows = parse_attr_file(attr_path)
    train_rows, test_rows = official_stargan_split(rows)
    image_size = int(cfg_get(cfg, "model", "image_size", 128))
    train_ds = CelebAAttrDataset(train_rows, image_dir, attrs, image_size)
    test_ds = CelebAAttrDataset(test_rows, image_dir, attrs, image_size)
    batch_size = int(cfg_get(cfg, "evaluation", "attribute_classifier_batch_size", 256))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg_get(cfg, "evaluation", "attribute_classifier_lr", 0.001)))
    loss_fn = nn.BCEWithLogitsLoss()
    epochs = int(cfg_get(cfg, "evaluation", "attribute_classifier_epochs", 3))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate_classifier_on_loader(model, test_loader, device, attrs)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        print(f"attr classifier epoch {epoch}: loss={history[-1]['train_loss']:.4f} exact={metrics['exact_match_accuracy']:.4f}")
    summary = {
        "checkpoint": str(ckpt_path),
        "epochs": epochs,
        "train_images": len(train_ds),
        "test_images": len(test_ds),
        "history": history,
        "final": history[-1] if history else {},
    }
    torch.save({"model": model.state_dict(), "summary": summary}, ckpt_path)
    write_json(summary_path, summary)
    model.eval()
    return model, summary


def evaluate_classifier_on_loader(model: nn.Module, loader: DataLoader, device: str, attrs: list[str]) -> dict[str, Any]:
    threshold = 0.5
    total = 0
    exact = 0
    per_attr_correct = np.zeros(len(attrs), dtype=np.float64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = (torch.sigmoid(model(images)) >= threshold).float()
            per_attr_correct += (pred == labels).float().sum(dim=0).cpu().numpy()
            exact += int((pred == labels).all(dim=1).sum().cpu())
            total += int(labels.size(0))
    return {
        "exact_match_accuracy": exact / max(1, total),
        "per_attr_accuracy": {name: float(per_attr_correct[idx] / max(1, total)) for idx, name in enumerate(attrs)},
    }


def predict_generated_attrs(cfg: dict[str, Any], model: nn.Module, manifest: dict[str, Any], device: str) -> dict[str, Any]:
    attrs = selected_attrs(cfg)
    image_size = int(cfg_get(cfg, "model", "image_size", 128))
    paths = [Path(record["generated_path"]) for record in manifest["records"]]
    dataset = GeneratedImageDataset(paths, image_size)
    loader = DataLoader(dataset, batch_size=int(cfg_get(cfg, "evaluation", "batch_size", 64)), shuffle=False)
    predictions: dict[str, list[int]] = {}
    threshold = float(cfg_get(cfg, "evaluation", "attribute_threshold", 0.5))
    with torch.no_grad():
        for images, batch_paths in loader:
            logits = model(images.to(device))
            pred = (torch.sigmoid(logits) >= threshold).int().cpu().numpy()
            for path, row in zip(batch_paths, pred):
                predictions[path] = [int(x) for x in row.tolist()]

    per_direction: dict[str, dict[str, Any]] = {}
    records = []
    for record in manifest["records"]:
        direction = record["direction"]
        target = [int(v) for v in record["target_label"]]
        pred = predictions[record["generated_path"]]
        attr_idx = attrs.index(direction)
        primary_success = pred[attr_idx] == target[attr_idx]
        if direction in HAIR_ATTRS:
            hair_idxs = [attrs.index(name) for name in HAIR_ATTRS]
            primary_success = primary_success and sum(pred[idx] for idx in hair_idxs) == 1
        strict_success = pred == target
        records.append({**record, "predicted_label": pred, "primary_success": bool(primary_success), "strict_success": bool(strict_success)})
        bucket = per_direction.setdefault(direction, {"total": 0, "primary_success": 0, "strict_success": 0})
        bucket["total"] += 1
        bucket["primary_success"] += int(primary_success)
        bucket["strict_success"] += int(strict_success)
    for direction, bucket in per_direction.items():
        total = max(1, bucket["total"])
        bucket["primary_success_rate"] = bucket["primary_success"] / total
        bucket["strict_success_rate"] = bucket["strict_success"] / total
    summary = {"per_direction": per_direction, "records": records}
    write_json(summary_dir(cfg) / "attribute_success_summary.json", summary)
    return summary


def evaluate_identity(cfg: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    require_torch()
    try:
        import insightface
    except Exception as exc:
        summary = {"available": False, "reason": repr(exc)}
        write_json(summary_dir(cfg) / "identity_retention_summary.json", summary)
        return summary

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    app = insightface.app.FaceAnalysis(name=str(cfg_get(cfg, "evaluation", "identity_model", "buffalo_l")), providers=providers)
    ctx_id = 0 if torch.cuda.is_available() else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    source_cache: dict[str, Optional[np.ndarray]] = {}
    similarities = []
    no_source = 0
    no_generated = 0
    records = []
    for record in manifest["records"]:
        source_path = record["source_path"]
        generated_path = record["generated_path"]
        if source_path not in source_cache:
            source_cache[source_path] = face_embedding(app, source_path)
        source_emb = source_cache[source_path]
        gen_emb = face_embedding(app, generated_path)
        if source_emb is None:
            no_source += 1
        if gen_emb is None:
            no_generated += 1
        sim = None
        if source_emb is not None and gen_emb is not None:
            sim = float(np.dot(source_emb, gen_emb) / (np.linalg.norm(source_emb) * np.linalg.norm(gen_emb) + 1e-12))
            similarities.append(sim)
        records.append({**record, "identity_cosine": sim})
    sims = np.asarray(similarities, dtype=np.float64)
    summary = {
        "available": True,
        "records": records,
        "pairs": len(manifest["records"]),
        "valid_pairs": int(len(similarities)),
        "no_source_face": no_source,
        "no_generated_face": no_generated,
        "mean": float(np.mean(sims)) if len(sims) else None,
        "median": float(np.median(sims)) if len(sims) else None,
        "p10": float(np.percentile(sims, 10)) if len(sims) else None,
        "warning_threshold": float(cfg_get(cfg, "evaluation", "identity_similarity_warning", 0.35)),
    }
    write_json(summary_dir(cfg) / "identity_retention_summary.json", summary)
    return summary


def face_embedding(app, path: str) -> Optional[np.ndarray]:
    import cv2

    image = cv2.imread(path)
    if image is None:
        return None
    faces = app.get(image)
    if not faces:
        return None
    face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
    return np.asarray(face.embedding, dtype=np.float32)


def evaluate_fid_is(cfg: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    require_torch()
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
    except Exception as exc:
        summary = {"available": False, "reason": repr(exc)}
        write_json(summary_dir(cfg) / "fid_is_summary.json", summary)
        return summary

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    image_size = int(cfg_get(cfg, "model", "image_size", 128))
    real_paths = sorted({Path(record["source_path"]) for record in manifest["records"]})
    fake_paths = [Path(record["generated_path"]) for record in manifest["records"]]
    try:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        update_metric_images(fid, real_paths, image_size, real=True, device=device)
        update_metric_images(fid, fake_paths, image_size, real=False, device=device)
        fid_value = float(fid.compute().detach().cpu())
        inception = InceptionScore(normalize=True).to(device)
        update_inception_images(inception, fake_paths, image_size, device=device)
        is_mean, is_std = inception.compute()
        per_direction = {}
        for direction in selected_attrs(cfg):
            direction_fake_paths = [Path(record["generated_path"]) for record in manifest["records"] if record["direction"] == direction]
            if not direction_fake_paths:
                continue
            direction_fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            update_metric_images(direction_fid, real_paths, image_size, real=True, device=device)
            update_metric_images(direction_fid, direction_fake_paths, image_size, real=False, device=device)
            direction_is = InceptionScore(normalize=True).to(device)
            update_inception_images(direction_is, direction_fake_paths, image_size, device=device)
            direction_is_mean, direction_is_std = direction_is.compute()
            per_direction[direction] = {
                "fid": float(direction_fid.compute().detach().cpu()),
                "inception_score_mean": float(direction_is_mean.detach().cpu()),
                "inception_score_std": float(direction_is_std.detach().cpu()),
                "generated_images": len(direction_fake_paths),
            }
        summary = {
            "available": True,
            "fid": fid_value,
            "inception_score_mean": float(is_mean.detach().cpu()),
            "inception_score_std": float(is_std.detach().cpu()),
            "real_images": len(real_paths),
            "generated_images": len(fake_paths),
            "per_direction": per_direction,
        }
    except Exception as exc:
        summary = {"available": False, "reason": repr(exc)}
    write_json(summary_dir(cfg) / "fid_is_summary.json", summary)
    return summary


def image_batch(paths: list[Path], image_size: int) -> torch.Tensor:
    require_torch()
    tensors = []
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    for path in paths:
        with Image.open(path) as handle:
            tensors.append(transform(handle.convert("RGB")))
    return torch.stack(tensors)


def update_metric_images(metric, paths: list[Path], image_size: int, real: bool, device: str) -> None:
    for start in range(0, len(paths), 32):
        batch = image_batch(paths[start : start + 32], image_size).to(device)
        metric.update(batch, real=real)


def update_inception_images(metric, paths: list[Path], image_size: int, device: str) -> None:
    for start in range(0, len(paths), 32):
        batch = image_batch(paths[start : start + 32], image_size).to(device)
        metric.update(batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task7_stargan/a800_full.py")
    parser.add_argument("--test-iters", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-count", type=int, default=None)
    parser.add_argument("--force-train-attribute-classifier", action="store_true")
    parser.add_argument("--skip-identity", action="store_true")
    parser.add_argument("--skip-fid-is", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_torch()
    cfg = load_config(args.config)
    device = args.device or str(cfg_get(cfg, "evaluation", "device", "cuda:0"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    test_iters = args.test_iters or int(cfg_get(cfg, "train", "final_test_iters", cfg_get(cfg, "train", "num_iters", 200000)))
    checkpoint = checkpoint_path_for_iters(cfg, test_iters)
    eval_count = args.eval_count or int(cfg_get(cfg, "data", "eval_sample_count", 512))
    eval_dir = asset_dir(cfg) / "evaluation" / f"iter_{test_iters}"
    manifest = generate_eval_images(cfg, checkpoint, eval_dir, eval_count, device=device)
    classifier, classifier_summary = train_or_load_classifier(cfg, device, force_train=args.force_train_attribute_classifier)
    attr_summary = predict_generated_attrs(cfg, classifier, manifest, device)
    identity_summary = {"available": False, "reason": "skipped"} if args.skip_identity else evaluate_identity(cfg, manifest)
    if args.skip_identity:
        write_json(summary_dir(cfg) / "identity_retention_summary.json", identity_summary)
    fid_is_summary = {"available": False, "reason": "skipped"} if args.skip_fid_is else evaluate_fid_is(cfg, manifest)
    if args.skip_fid_is:
        write_json(summary_dir(cfg) / "fid_is_summary.json", fid_is_summary)
    write_json(
        summary_dir(cfg) / "task7_evaluation_summary.json",
        {
            "test_iters": test_iters,
            "checkpoint": str(checkpoint),
            "eval_generation_manifest": str(eval_dir / "eval_generation_manifest.json"),
            "attribute_classifier": classifier_summary,
            "attribute_success": attr_summary["per_direction"],
            "identity_retention": {k: v for k, v in identity_summary.items() if k != "records"},
            "fid_is": fid_is_summary,
        },
    )


if __name__ == "__main__":
    main()
