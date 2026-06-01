#!/usr/bin/env python3
"""Run the official InsightFace ArcFace Torch pipeline for Stage2 task 5.x."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any


def load_config(path: str):
    from mmengine.config import Config

    return Config.fromfile(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def size_key(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


def inspect_lfw_bin(path: Path, expected_size: tuple[int, int] = (112, 112), sample_limit: int = 12000) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {"path": str(path), "exists": False, "aligned_112x112": False}

    from PIL import Image

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
            "readable": False,
            "size_bytes": path.stat().st_size,
            "error": f"{exc.__class__.__name__}: {exc}",
            "aligned_112x112": False,
        }

    counts: Counter[str] = Counter()
    inspected = min(len(bins), sample_limit)
    decode_errors: list[str] = []
    for raw in bins[:inspected]:
        try:
            with Image.open(BytesIO(raw)) as image:
                counts[size_key(image.size)] += 1
        except Exception as exc:  # noqa: BLE001
            if len(decode_errors) < 5:
                decode_errors.append(f"{exc.__class__.__name__}: {exc}")

    expected = size_key(expected_size)
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


def run_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


def capture_command(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=str(cwd) if cwd else None, text=True).strip()


def ensure_repo(repo_url: str, ref: str, external_dir: Path) -> dict[str, Any]:
    external_dir.parent.mkdir(parents=True, exist_ok=True)
    arcface_root = external_dir / "recognition" / "arcface_torch"
    if not (external_dir / ".git").exists():
        if arcface_root.exists():
            print(
                "WARNING: InsightFace source exists without .git metadata; using the existing local copy "
                f"at {external_dir}.",
                flush=True,
            )
            return {"repo_url": repo_url, "ref": ref, "path": str(external_dir), "commit": "unknown-local-copy"}
        run_command(["git", "clone", "--depth", "1", "--branch", ref, repo_url, str(external_dir)])
    else:
        try:
            run_command(["git", "fetch", "--depth", "1", "origin", ref], cwd=external_dir)
            run_command(["git", "checkout", "FETCH_HEAD"], cwd=external_dir)
        except subprocess.CalledProcessError as exc:
            if not arcface_root.exists():
                raise
            print(
                "WARNING: git fetch for InsightFace failed; using the existing local clone "
                f"at {external_dir}. Original error code: {exc.returncode}",
                flush=True,
            )
    try:
        commit = capture_command(["git", "rev-parse", "HEAD"], cwd=external_dir)
    except subprocess.CalledProcessError:
        commit = "unknown-local-copy"
    return {"repo_url": repo_url, "ref": ref, "path": str(external_dir), "commit": commit}


def arcface_dir(cfg: Any) -> Path:
    return Path(cfg.insightface.external_dir) / cfg.insightface.arcface_subdir


def patch_verification_interp(arcface_root: Path) -> dict[str, Any]:
    """Patch official LFW verification for newer SciPy duplicate-x handling."""
    verification_path = arcface_root / "eval" / "verification.py"
    marker = "# Stage2 patch: deduplicate FAR values before scipy interpolation."
    if not verification_path.exists():
        raise FileNotFoundError(f"Missing official verification.py: {verification_path}")
    text = verification_path.read_text(encoding="utf-8")
    if marker in text:
        return {"path": str(verification_path), "applied": False, "reason": "already patched"}

    old = """        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
"""
    new = f"""        if np.max(far_train) >= far_target:
            {marker}
            unique_far, unique_indices = np.unique(far_train, return_index=True)
            unique_thresholds = thresholds[unique_indices]
            order = np.argsort(unique_far)
            unique_far = unique_far[order]
            unique_thresholds = unique_thresholds[order]
            if unique_far.size < 2:
                threshold = unique_thresholds[0]
            else:
                f = interpolate.interp1d(
                    unique_far,
                    unique_thresholds,
                    kind='slinear',
                    bounds_error=False,
                    fill_value=(unique_thresholds[0], unique_thresholds[-1]))
                threshold = f(far_target)
        else:
            threshold = 0.0
"""
    if old not in text:
        raise RuntimeError("Could not find the expected scipy interp1d block in official verification.py")
    verification_path.write_text(text.replace(old, new), encoding="utf-8")
    return {"path": str(verification_path), "applied": True, "reason": "patched duplicate FAR handling"}


def official_config_text(cfg: Any, rec_dir: Path, output_dir: Path) -> str:
    official = cfg.official
    return f'''from easydict import EasyDict as edict

config = edict()
config.margin_list = {tuple(official.margin_list)!r}
config.network = {official.network!r}
config.resume = {bool(official.resume)!r}
config.output = {str(output_dir).replace(os.sep, "/")!r}
config.embedding_size = {int(official.embedding_size)}
config.sample_rate = {float(official.sample_rate)}
config.fp16 = {bool(official.fp16)!r}
config.momentum = {float(official.momentum)}
config.weight_decay = {float(official.weight_decay)}
config.batch_size = {int(official.batch_size)}
config.lr = {float(official.lr)}
config.verbose = {int(official.verbose)}
config.dali = {bool(official.dali)!r}
config.dali_aug = {bool(getattr(official, "dali_aug", False))!r}
config.optimizer = {getattr(official, "optimizer", "sgd")!r}
config.num_workers = {int(getattr(official, "num_workers", 8))}
config.rec = {str(rec_dir).replace(os.sep, "/")!r}
config.num_classes = {int(official.num_classes)}
config.num_image = {int(official.num_image)}
config.num_epoch = {int(official.num_epoch)}
config.warmup_epoch = {int(official.warmup_epoch)}
config.val_targets = {list(official.val_targets)!r}
'''


def write_official_config(cfg: Any) -> Path:
    rec_dir = Path(cfg.data.rec).resolve()
    output_dir = Path(cfg.official.output).resolve()
    target = arcface_dir(cfg) / "configs" / f"{cfg.insightface.generated_config_name}.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(official_config_text(cfg, rec_dir, output_dir), encoding="utf-8")
    return target


def validate_recordio_layout(cfg: Any) -> dict[str, Any]:
    rec_dir = Path(cfg.data.rec)
    required = ["train.rec", "train.idx", "property", "lfw.bin"]
    files = {
        name: {
            "path": str(rec_dir / name),
            "exists": (rec_dir / name).exists(),
            "size_bytes": (rec_dir / name).stat().st_size if (rec_dir / name).exists() else 0,
        }
        for name in required
    }
    ready = all(item["exists"] and item["size_bytes"] > 0 for item in files.values())
    if not ready:
        missing = [name for name, item in files.items() if not item["exists"] or item["size_bytes"] <= 0]
        raise FileNotFoundError(f"RecordIO layout is not ready under {rec_dir}; missing or empty: {missing}")
    lfw_inspection = inspect_lfw_bin(rec_dir / "lfw.bin")
    validation_ready = bool(lfw_inspection.get("aligned_112x112"))
    validation_warning = None
    if not validation_ready:
        validation_warning = (
            "lfw.bin is not confirmed as 112x112 aligned. Official training can run, but this LFW metric "
            "is not the acceptance metric until the bin is replaced with an aligned validation target."
        )
        print(f"WARNING: {validation_warning}")
    return {
        "recordio_dir": str(rec_dir),
        "files": files,
        "lfw_bin_inspection": lfw_inspection,
        "validation_ready": validation_ready,
        "validation_warning": validation_warning,
        "ready": ready,
    }


def parse_lfw_metrics_from_text(text: str) -> dict[str, Any]:
    accuracies = [float(value) for value in re.findall(r"\[lfw\]\[\d+\]Accuracy-Flip:\s*([0-9.]+)\+-", text)]
    highest = [float(value) for value in re.findall(r"\[lfw\]\[\d+\]Accuracy-Highest:\s*([0-9.]+)", text)]
    xnorm = [float(value) for value in re.findall(r"\[lfw\]\[\d+\]XNorm:\s*([0-9.]+)", text)]
    return {
        "lfw_accuracy_history": accuracies,
        "lfw_highest_history": highest,
        "lfw_xnorm_history": xnorm,
        "best_lfw_accuracy": max(highest or accuracies) if (highest or accuracies) else None,
        "last_lfw_accuracy": accuracies[-1] if accuracies else None,
    }


def collect_training_logs(output_dir: Path) -> str:
    chunks: list[str] = []
    for path in sorted(output_dir.glob("*.log")) + sorted(output_dir.glob("*.txt")):
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


def setup(args: argparse.Namespace, cfg: Any) -> dict[str, Any]:
    repo = ensure_repo(cfg.insightface.repo_url, args.insightface_ref or cfg.insightface.ref, Path(cfg.insightface.external_dir))
    verification_patch = patch_verification_interp(arcface_dir(cfg))
    config_path = write_official_config(cfg)
    layout = validate_recordio_layout(cfg)
    summary = {
        "task": cfg.task_name,
        "repo": repo,
        "verification_patch": verification_patch,
        "official_config": str(config_path),
        "recordio": layout,
        "output_dir": str(Path(cfg.official.output)),
    }
    write_json(Path(args.summary_out or cfg.train.summary_out), summary)
    return summary


def train(args: argparse.Namespace, cfg: Any) -> dict[str, Any]:
    setup_summary = setup(args, cfg)
    official_name = f"configs/{cfg.insightface.generated_config_name}.py"
    output_dir = Path(cfg.official.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    run_command([sys.executable, "train_v2.py", official_name], cwd=arcface_dir(cfg), env=env)
    elapsed = round(time.time() - started, 2)
    logs = collect_training_logs(output_dir)
    metrics = parse_lfw_metrics_from_text(logs)
    model_path = output_dir / "model.pt"
    checkpoint_path = output_dir / "checkpoint_gpu_0.pt"
    summary = {
        **setup_summary,
        "seconds": elapsed,
        "model_path": str(model_path),
        "checkpoint_path": str(checkpoint_path),
        "model_exists": model_path.exists(),
        "checkpoint_exists": checkpoint_path.exists(),
        "metrics": metrics,
        "target_lfw_accuracy": float(cfg.train.target_lfw_accuracy),
        "target_met": bool((metrics.get("best_lfw_accuracy") or 0.0) >= float(cfg.train.target_lfw_accuracy)),
        "note": "Official InsightFace does not save a separate best checkpoint; model.pt is the latest rank-0 backbone.",
    }
    write_json(Path(args.summary_out or cfg.train.summary_out), summary)
    return summary


def eval_summary(args: argparse.Namespace, cfg: Any) -> dict[str, Any]:
    output_dir = Path(cfg.official.output)
    logs = collect_training_logs(output_dir)
    metrics = parse_lfw_metrics_from_text(logs)
    checkpoint = Path(args.checkpoint) if args.checkpoint else output_dir / "model.pt"
    summary = {
        "task": cfg.task_name,
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "official_log_dir": str(output_dir),
        "metrics": metrics,
        "accuracy": metrics.get("best_lfw_accuracy"),
        "target_lfw_accuracy": float(cfg.train.target_lfw_accuracy),
        "target_met": bool((metrics.get("best_lfw_accuracy") or 0.0) >= float(cfg.train.target_lfw_accuracy)),
        "note": "This summary is parsed from official InsightFace LFW validation logs.",
    }
    write_json(Path(args.summary_out or cfg.train.eval_summary_out), summary)
    return summary


def load_bin_payload(path: Path) -> tuple[list[bytes], list[bool]]:
    with path.open("rb") as handle:
        try:
            bins, issame_list = pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            bins, issame_list = pickle.load(handle, encoding="bytes")
    return bins, [bool(item) for item in issame_list]


def image_bytes_to_tensor(raw: bytes, image_size: tuple[int, int], flip: bool = False):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(BytesIO(raw)) as image:
        image = image.convert("RGB")
        if image.size != image_size:
            image = image.resize(image_size, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    if flip:
        array = array[:, ::-1, :].copy()
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    return ((tensor / 255.0) - 0.5) / 0.5


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def evaluate_lfw_bin(
    cfg: Any,
    checkpoint: Path,
    bin_path: Path,
    batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    import numpy as np
    import sklearn.preprocessing
    import torch

    root = arcface_dir(cfg).resolve()
    sys.path.insert(0, str(root))
    from backbones import get_model  # type: ignore
    from eval import verification  # type: ignore

    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    backbone = get_model(
        cfg.official.network,
        dropout=0.0,
        fp16=bool(cfg.official.fp16 and device.type == "cuda"),
        num_features=int(cfg.official.embedding_size),
    )
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict_backbone" in state:
        state = state["state_dict_backbone"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload in {checkpoint}")
    backbone.load_state_dict(strip_module_prefix(state), strict=True)
    backbone.to(device)
    backbone.eval()

    bins, issame_list = load_bin_payload(bin_path)
    image_size = tuple(cfg.data.expected_image_size)
    embeddings_by_flip: list[np.ndarray] = []
    started = time.time()
    for flip in (False, True):
        chunks: list[np.ndarray] = []
        for start in range(0, len(bins), batch_size):
            batch = bins[start : start + batch_size]
            tensor = torch.stack([image_bytes_to_tensor(raw, image_size, flip=flip) for raw in batch]).to(device)
            with torch.no_grad():
                output = backbone(tensor).detach().cpu().numpy()
            chunks.append(output)
        embeddings_by_flip.append(np.concatenate(chunks, axis=0))

    combined = embeddings_by_flip[0] + embeddings_by_flip[1]
    xnorm = float(np.mean(np.linalg.norm(combined, axis=1)))
    combined = sklearn.preprocessing.normalize(combined)
    _, _, accuracy, val, val_std, far = verification.evaluate(combined, issame_list, nrof_folds=10)
    elapsed = round(time.time() - started, 2)
    return {
        "pairs": len(issame_list),
        "images": len(bins),
        "accuracy": float(np.mean(accuracy)),
        "accuracy_std": float(np.std(accuracy)),
        "fold_accuracies": [float(item) for item in accuracy],
        "val_at_far_1e-3": float(val),
        "val_std": float(val_std),
        "far": float(far),
        "xnorm": xnorm,
        "seconds": elapsed,
        "batch_size": batch_size,
        "device": str(device),
    }


def eval_bin(args: argparse.Namespace, cfg: Any) -> dict[str, Any]:
    repo = ensure_repo(cfg.insightface.repo_url, args.insightface_ref or cfg.insightface.ref, Path(cfg.insightface.external_dir))
    verification_patch = patch_verification_interp(arcface_dir(cfg))
    bin_path = Path(args.bin_path) if args.bin_path else Path(cfg.data.rec) / f"{args.target_name}.bin"
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg.official.output) / "model.pt"
    inspection = inspect_lfw_bin(bin_path)
    metrics = evaluate_lfw_bin(
        cfg=cfg,
        checkpoint=checkpoint,
        bin_path=bin_path,
        batch_size=int(args.batch_size),
        device_name=args.device,
    )
    summary = {
        "task": cfg.task_name,
        "repo": repo,
        "verification_patch": verification_patch,
        "target_name": args.target_name,
        "bin_path": str(bin_path),
        "bin_inspection": inspection,
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "metrics": metrics,
        "accuracy": metrics["accuracy"],
        "target_lfw_accuracy": float(cfg.train.target_lfw_accuracy),
        "target_met": bool(metrics["accuracy"] >= float(cfg.train.target_lfw_accuracy)),
        "note": "This is a direct post-training evaluation of model.pt on the selected InsightFace-format validation bin.",
    }
    write_json(Path(args.summary_out or cfg.train.eval_summary_out), summary)
    return summary


def cleanup_external(args: argparse.Namespace, cfg: Any) -> None:
    target = Path(cfg.insightface.external_dir)
    if target.exists():
        resolved = target.resolve()
        stage2 = Path.cwd().resolve()
        if stage2.name != "stage-2":
            raise RuntimeError("Run cleanup-external from the stage-2 directory.")
        if stage2 not in resolved.parents:
            raise RuntimeError(f"Refusing to delete outside stage-2: {resolved}")
        shutil.rmtree(resolved)
        print(f"Deleted {resolved}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "train", "eval-summary", "eval-bin", "cleanup-external"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", default="configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py")
        sub.add_argument("--insightface-ref", default=None)
        sub.add_argument("--summary-out", default=None)
        sub.add_argument("--checkpoint", default=None)
        sub.add_argument("--bin-path", default=None)
        sub.add_argument("--target-name", default="lfw")
        sub.add_argument("--batch-size", default=256, type=int)
        sub.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.command == "setup":
        setup(args, cfg)
    elif args.command == "train":
        train(args, cfg)
    elif args.command == "eval-summary":
        eval_summary(args, cfg)
    elif args.command == "eval-bin":
        eval_bin(args, cfg)
    elif args.command == "cleanup-external":
        cleanup_external(args, cfg)


if __name__ == "__main__":
    main()
