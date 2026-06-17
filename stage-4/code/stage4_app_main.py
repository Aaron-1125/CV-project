#!/usr/bin/env python3
"""Unified Stage4 app entry for source runs and PyInstaller bundles."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


HELP_TEXT = """Stage4FaceEffects unified entry

Usage:
  python stage-4/code/stage4_app_main.py --gui [--safe]
  python stage-4/code/stage4_app_main.py --run-cli <stage4_run_cli.py args>
  python stage-4/code/stage4_app_main.py --live-worker <stage4_live_camera_worker.py args>
  python stage-4/code/stage4_app_main.py --write-report
  python stage-4/code/stage4_app_main.py --check-env

When packaged with PyInstaller, the GUI starts child processes through this
same executable with --run-cli or --live-worker.
"""


def dispatch_module(module_name: str, argv: List[str]) -> int:
    sys.argv = ["{}.py".format(module_name)] + list(argv)
    module = importlib.import_module(module_name)
    result = module.main()
    return int(result or 0)


def main(argv: List[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        args = ["--gui"]

    mode = args[0]
    rest = args[1:]

    if mode in ("-h", "--help"):
        print(HELP_TEXT)
        return 0
    if mode == "--gui":
        return dispatch_module("stage4_desktop_app", rest)
    if mode == "--run-cli":
        if "--image" in rest:
            return dispatch_module("stage4_process_image_cli", rest)
        return dispatch_module("stage4_run_cli", rest)
    if mode == "--live-worker":
        return dispatch_module("stage4_live_camera_worker", rest)
    if mode == "--write-report":
        return dispatch_module("stage4_write_report", rest)
    if mode == "--check-env":
        return dispatch_module("stage4_run_cli", ["--check-env"] + rest)

    print("Unknown Stage4 app mode: {}".format(mode), file=sys.stderr)
    print(HELP_TEXT, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
