#!/usr/bin/env python3
"""Capture real Windows terminal windows for the Stage2 week 2 report.

This script intentionally does not render fake terminal images. It opens
visible PowerShell windows, runs the requested commands, captures the actual
window pixels with PIL ImageGrab, then closes the windows.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from PIL import ImageGrab


CREATE_NEW_CONSOLE = 0x00000010
SW_RESTORE = 9

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()


def ps_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def enum_windows() -> Iterable[int]:
    handles: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def callback(hwnd: int, _lparam: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            handles.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return handles


def window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def find_window(title_fragment: str, timeout_s: float = 20.0) -> int:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        matches = [hwnd for hwnd in enum_windows() if title_fragment in window_title(hwnd)]
        if matches:
            return matches[0]
        time.sleep(0.25)
    raise TimeoutError(f"Could not find terminal window containing title: {title_fragment}")


def wait_for_marker(marker: Path, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if marker.exists():
            return
        time.sleep(0.5)
    raise TimeoutError(f"Command marker was not written: {marker}")


def capture_window(hwnd: int, out_path: Path) -> None:
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    time.sleep(1.0)
    rect = ctypes.wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("GetWindowRect failed")
    bbox = (rect.left, rect.top, rect.right, rect.bottom)
    image = ImageGrab.grab(bbox=bbox)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def write_script(script_path: Path, title: str, cwd: Path, body: str, marker: Path) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    text = f"""$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = {ps_quote(title)}
Set-Location -LiteralPath {ps_quote(cwd)}
Clear-Host
Write-Host 'Stage2 week2 terminal run'
Write-Host 'Task: {title}'
Write-Host 'Working directory:' (Get-Location).Path
Write-Host ''
{body}
Set-Content -LiteralPath {ps_quote(marker)} -Value ((Get-Date).ToString('s'))
Start-Sleep -Seconds 90
"""
    script_path.write_text(text, encoding="utf-8")


def launch_and_capture(
    *,
    title: str,
    cwd: Path,
    body: str,
    out_path: Path,
    work_dir: Path,
    timeout_s: float,
) -> None:
    safe_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in title).strip("_")
    script_path = work_dir / f"{safe_name}.ps1"
    marker = work_dir / f"{safe_name}.done"
    if marker.exists():
        marker.unlink()
    write_script(script_path, title, cwd, body, marker)
    process = subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
        ],
        cwd=str(cwd),
        creationflags=CREATE_NEW_CONSOLE,
        text=True,
    )
    try:
        hwnd = find_window(title)
        wait_for_marker(marker, timeout_s=timeout_s)
        capture_window(hwnd, out_path)
    finally:
        terminate_process_tree(process)


def build_commands(repo_root: Path, stage2_root: Path) -> list[dict[str, object]]:
    return [
        {
            "title": "Stage2 Week2 - Environment Check",
            "cwd": repo_root,
            "out": "terminal_environment_check.png",
            "timeout": 180.0,
            "body": """
Write-Host '$ docker compose config --services'
docker compose config --services
Write-Host ''
Write-Host '$ docker compose run --rm -w /workspace stage2-gpu python docker/verify_environment.py'
docker compose run --rm -w /workspace stage2-gpu python docker/verify_environment.py
""",
        },
        {
            "title": "Stage2 Week2 - Task5 Cloud Summary",
            "cwd": repo_root,
            "out": "terminal_task5_cloud_sync.png",
            "timeout": 60.0,
            "body": r"""
Write-Host '$ python - read Task5 cloud summaries'
@'
import json
from pathlib import Path
train = json.loads(Path("stage-2/reports/task5/summaries/ms1mv3_dense_train_summary.json").read_text(encoding="utf-8"))
eval_ = json.loads(Path("stage-2/reports/task5/summaries/lfw_eval_summary.json").read_text(encoding="utf-8"))
print("Task5 cloud result synchronized to reports/task5")
print("num_images:", train["num_images"])
print("num_identities:", train["num_identities"])
print("epochs_completed:", train["epochs_completed"])
print("best_lfw_accuracy:", train["best_lfw_accuracy"])
print("lfw_eval_accuracy:", eval_["metrics"]["accuracy"])
print("target_met:", eval_["target_met"])
print("training_plot:", "stage-2/reports/task5/assets/training/ms1mv3_dense_loss_acc_curve.png")
'@ | python -
""",
        },
        {
            "title": "Stage2 Week2 - Task6 Results",
            "cwd": repo_root,
            "out": "terminal_task6_eval_summary.png",
            "timeout": 60.0,
            "body": r"""
Write-Host '$ python - read Task6 quantization and ONNX summaries'
@'
import json
from pathlib import Path
quant = json.loads(Path("stage-2/reports/task6/summaries/quantization_summary.json").read_text(encoding="utf-8"))
onnx = json.loads(Path("stage-2/reports/task6/summaries/onnx_summary.json").read_text(encoding="utf-8"))
print("FP32 LFW accuracy:", quant["fp32"]["metrics"]["accuracy"])
print("Dynamic INT8 LFW accuracy:", quant["dynamic_quantized"]["metrics"]["accuracy"])
print("FP32 model size MB:", round(quant["fp32"]["model_size_mb"], 2))
print("Dynamic INT8 model size MB:", round(quant["dynamic_quantized"]["model_size_mb"], 2))
print("ONNX LFW accuracy:", onnx["onnx"]["accuracy"])
print("ONNX mean cosine:", onnx["consistency"]["mean_cosine"])
print("ONNX max abs diff:", onnx["consistency"]["max_abs_diff"])
'@ | python -
""",
        },
        {
            "title": "Stage2 Week2 - Demo Training",
            "cwd": stage2_root,
            "out": "terminal_demo_training.png",
            "timeout": 420.0,
            "body": """
Write-Host '$ docker compose run --rm -w /workspace/stage-2 stage2-gpu python code/task5/stage2_task5_run_arcface.py train ...'
docker compose run --rm -w /workspace/stage-2 stage2-gpu python code/task5/stage2_task5_run_arcface.py train --config work_dirs/task5/weekly_demo_train/weekly_demo_config.py --work-dir work_dirs/task5/weekly_demo_train/real_terminal_run --summary-out reports/task5/logs/weekly_demo_train_summary_real_terminal.json --loss-plot-out reports/task5/logs/weekly_demo_train_curve_real_terminal.png --device cuda:0
""",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--stage2-root", type=Path, default=Path.cwd() / "stage-2")
    parser.add_argument("--out-dir", type=Path, default=Path("stage-2/reports/assets/weekly/week2"))
    parser.add_argument("--work-dir", type=Path, default=Path("stage-2/work_dirs/task6/weekly_terminal_capture"))
    return parser.parse_args()


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("This real terminal capture helper is Windows-only.")
    args = parse_args()
    repo_root = args.repo_root.resolve()
    stage2_root = args.stage2_root.resolve()
    out_dir = args.out_dir.resolve()
    work_dir = args.work_dir.resolve()
    for command in build_commands(repo_root, stage2_root):
        out_path = out_dir / str(command["out"])
        print(f"Opening real terminal: {command['title']}")
        launch_and_capture(
            title=str(command["title"]),
            cwd=command["cwd"],  # type: ignore[arg-type]
            body=str(command["body"]),
            out_path=out_path,
            work_dir=work_dir,
            timeout_s=float(command["timeout"]),
        )
        print(f"Captured {out_path}")


if __name__ == "__main__":
    main()
