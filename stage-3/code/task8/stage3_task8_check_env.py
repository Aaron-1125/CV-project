#!/usr/bin/env python3
"""Check the external 3DDFA_V2 repo and Task8 runtime dependencies."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

from stage3_task8_common import (
    choose_backend,
    dependency_status,
    env_summary_path,
    inspect_3ddfa_repo,
    load_config,
    now_ts,
    summary_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task8_3dface/a800_3ddfa_v2.py")
    parser.add_argument("--backend", choices=["auto", "onnx", "pth"], default="auto")
    parser.add_argument("--allow-missing", action="store_true", help="Write summary but do not exit non-zero for missing repo/weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    repo_info = inspect_3ddfa_repo(cfg)
    backend, backend_reason = choose_backend(cfg, args.backend, repo_info)
    deps = dependency_status(
        [
            "cv2",
            "yaml",
            "numpy",
            "torch",
            "onnxruntime",
            "PIL",
            "matplotlib",
            "trimesh",
            "pyrender",
            "pytorch3d",
        ]
    )
    payload = {
        "task": cfg.get("task_name"),
        "checked_at": now_ts(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "env": {
            "3DDFA_REPO": os.environ.get("3DDFA_REPO"),
            "CELEBA_ROOT": os.environ.get("CELEBA_ROOT"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "repo": repo_info,
        "requested_backend": args.backend,
        "selected_backend": backend,
        "backend_reason": backend_reason,
        "dependencies": deps,
        "ready": bool(repo_info.get("ready")),
    }
    write_json(env_summary_path(cfg), payload)
    print("3DDFA_V2 repo: {}".format(repo_info.get("repo_path")))
    print("Repo ready: {}".format(repo_info.get("ready")))
    print("Commit: {}".format(repo_info.get("commit") or "N/A"))
    print("Selected backend: {} ({})".format(backend, backend_reason))
    if not repo_info.get("ready"):
        print("ERROR: {}".format(repo_info.get("hint")), file=sys.stderr)
        if not args.allow_missing:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
