#!/usr/bin/env python3
"""Runtime path helpers for source and PyInstaller macOS app modes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List


APP_NAME = "Stage4FaceEffects"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def source_stage4_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_repo_root() -> Path:
    return source_stage4_root().parent


def app_base_dir() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
    return source_repo_root()


def resource_path(relative_path: str | Path) -> Path:
    path = Path(relative_path).expanduser()
    if path.is_absolute():
        return path
    return app_base_dir() / path


def bundled_or_source_path(relative_path: str | Path) -> Path:
    bundled = resource_path(relative_path)
    if bundled.exists():
        return bundled
    return source_repo_root() / Path(relative_path)


def user_data_dir() -> Path:
    if is_frozen():
        return Path.home() / "Documents" / APP_NAME
    return source_stage4_root() / "reports"


def current_executable_command(mode: str) -> List[str]:
    flag = mode if mode.startswith("--") else "--{}".format(mode)
    if is_frozen():
        return [str(Path(sys.executable).resolve()), flag]
    return [sys.executable, str(source_stage4_root() / "code" / "stage4_app_main.py"), flag]


def source_script_command(script_name: str) -> List[str]:
    return [sys.executable, str(source_stage4_root() / "code" / script_name)]
