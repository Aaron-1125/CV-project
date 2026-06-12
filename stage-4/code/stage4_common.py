#!/usr/bin/env python3
"""Shared helpers for Stage4 project integration."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import runpy
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stage4_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return stage4_root().parent


def stage3_root() -> Path:
    return repo_root() / "stage-3"


def task9_code_dir() -> Path:
    return stage3_root() / "code" / "task9"


def stage4_code_dir() -> Path:
    return stage4_root() / "code"


def stage4_report_dir() -> Path:
    return stage4_root() / "reports"


def stage4_asset_dir() -> Path:
    return stage4_report_dir() / "assets"


def stage4_video_dir() -> Path:
    return stage4_asset_dir() / "videos"


def stage4_summary_dir() -> Path:
    return stage4_report_dir() / "summaries"


def stage4_summary_path() -> Path:
    return stage4_summary_dir() / "stage4_integration_summary.json"


def stage4_report_path() -> Path:
    return stage4_report_dir() / "stage4_project_integration_report.md"


def default_stage4_config_path() -> Path:
    return stage4_root() / "configs" / "stage4_app_config.py"


def ensure_stage4_dirs() -> None:
    for path in [
        stage4_code_dir(),
        stage4_root() / "configs",
        stage4_report_dir(),
        stage4_asset_dir(),
        stage4_video_dir(),
        stage4_summary_dir(),
    ]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_repo_path(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return repo_root() / path


def resolve_stage4_path(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return stage4_root() / path


def resolve_cwd_or_repo_path(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return repo_root() / path


def add_task9_to_path() -> Path:
    task9_dir = task9_code_dir()
    task9_text = str(task9_dir)
    if task9_text not in sys.path:
        sys.path.insert(0, task9_text)
    return task9_dir


def load_python_config(config_path: Optional[Any] = None) -> Dict[str, Any]:
    path = default_stage4_config_path() if config_path is None else resolve_cwd_or_repo_path(config_path)
    if not path.exists():
        path = resolve_stage4_path(config_path)
    namespace = runpy.run_path(str(path))
    return {key: value for key, value in namespace.items() if not key.startswith("__")}


def write_json(path_value: Any, payload: Any) -> None:
    target = Path(path_value)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path_value: Any, text: str) -> None:
    target = Path(path_value)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def read_json(path_value: Any) -> Any:
    return json.loads(Path(path_value).read_text(encoding="utf-8"))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def module_status(names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    status: Dict[str, Dict[str, Any]] = {}
    for name in names:
        entry: Dict[str, Any] = {"available": importlib.util.find_spec(name) is not None}
        if entry["available"]:
            try:
                module = __import__(name)
                entry["version"] = getattr(module, "__version__", "unknown")
                entry["path"] = getattr(module, "__file__", "builtin")
            except Exception as exc:
                entry["import_error"] = "{}: {}".format(type(exc).__name__, exc)
        status[name] = entry
    return status


def python_summary() -> Dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }


def rel_to_repo(path_value: Any) -> str:
    path = Path(path_value)
    try:
        return os.path.relpath(str(path), str(repo_root()))
    except Exception:
        return str(path)
