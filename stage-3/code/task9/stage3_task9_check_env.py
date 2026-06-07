#!/usr/bin/env python3
"""Check Stage3 Task9 runtime dependencies."""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
from typing import Any, Dict

from stage3_task9_common import (
    INSTALL_HINT,
    dependency_status,
    ensure_task9_dirs,
    env_summary_path,
    load_config,
    now_ts,
    python_version_summary,
    summary_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task9_effects/a800_mediapipe_face_effects.py")
    parser.add_argument("--allow-missing", action="store_true", help="Write summary but do not exit non-zero for missing dependencies.")
    return parser.parse_args()


def torch_cuda_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"torch_available": False, "cuda_available": False}
    try:
        import torch  # type: ignore

        info["torch_available"] = True
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
    except Exception as exc:
        info["torch_error"] = "{}: {}".format(type(exc).__name__, str(exc))
    return info


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_task9_dirs(cfg)
    deps = dependency_status(["cv2", "mediapipe", "numpy", "PIL", "tqdm", "matplotlib", "torch"])
    missing_required = [name for name in ["cv2", "mediapipe", "numpy", "PIL"] if not deps.get(name, {}).get("available")]
    payload = {
        "task": cfg.get("task_name"),
        "checked_at": now_ts(),
        "python": python_version_summary(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "env": {
            "CELEBA_ROOT": os.environ.get("CELEBA_ROOT"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "dependencies": deps,
        "gpu": torch_cuda_info(),
        "mediapipe_required": True,
        "install_hint": INSTALL_HINT if missing_required else None,
        "missing_required_dependencies": missing_required,
        "ready": not missing_required,
    }
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    write_json(env_summary_path(cfg), payload)
    print("Task9 environment ready: {}".format(payload["ready"]))
    if missing_required:
        print("Missing required dependencies: {}".format(", ".join(missing_required)))
        print("Install with: {}".format(INSTALL_HINT))
        if not args.allow_missing:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
