#!/usr/bin/env python3
"""Run the official InsightFace ArcFace Torch pipeline for Stage2 task 5.x."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
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


def run_command(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


def capture_command(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=str(cwd) if cwd else None, text=True).strip()


def ensure_repo(repo_url: str, ref: str, external_dir: Path) -> dict[str, Any]:
    external_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (external_dir / ".git").exists():
        run_command(["git", "clone", "--depth", "1", "--branch", ref, repo_url, str(external_dir)])
    else:
        run_command(["git", "fetch", "--depth", "1", "origin", ref], cwd=external_dir)
        run_command(["git", "checkout", "FETCH_HEAD"], cwd=external_dir)
    commit = capture_command(["git", "rev-parse", "HEAD"], cwd=external_dir)
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
    return {"recordio_dir": str(rec_dir), "files": files, "ready": ready}


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
    for name in ("setup", "train", "eval-summary", "cleanup-external"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", default="configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py")
        sub.add_argument("--insightface-ref", default=None)
        sub.add_argument("--summary-out", default=None)
        sub.add_argument("--checkpoint", default=None)
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
    elif args.command == "cleanup-external":
        cleanup_external(args, cfg)


if __name__ == "__main__":
    main()
