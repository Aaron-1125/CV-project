#!/usr/bin/env python3
"""Capture real Terminal screenshots for the week 1 report on macOS.

The script opens Terminal, runs compact verification commands, captures the
screen, crops the Terminal window area, and writes PNGs under reports/assets.
It is intentionally macOS-only because it uses osascript and screencapture.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


TERMINAL_BOUNDS_POINTS = (60, 70, 1180, 760)
CAPTURE_PADDING_PX = 12


@dataclass(frozen=True)
class CaptureCommand:
    name: str
    title: str
    command: str


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def applescript_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def crop_terminal(full_path: Path, output_path: Path) -> None:
    image = Image.open(full_path)
    scale = image.width / 1280
    left, top, right, bottom = TERMINAL_BOUNDS_POINTS
    crop_box = (
        max(0, int(left * scale) - CAPTURE_PADDING_PX),
        max(0, int(top * scale) - CAPTURE_PADDING_PX),
        min(image.width, int(right * scale) + CAPTURE_PADDING_PX),
        min(image.height, int(bottom * scale) + CAPTURE_PADDING_PX),
    )
    trim_terminal_blank_bottom(image.crop(crop_box)).save(output_path)


def trim_terminal_blank_bottom(image: Image.Image) -> Image.Image:
    """Remove unused blank rows below the last visible Terminal output line."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    left = max(20, int(width * 0.04))
    right = min(width - 20, int(width * 0.96))
    top = max(60, int(height * 0.06))
    bottom_limit = max(top + 1, height - max(24, int(height * 0.04)))
    pixels = rgb.load()

    last_content_y = top
    for y in range(bottom_limit - 1, top - 1, -1):
        for x in range(left, right, 3):
            r, g, b = pixels[x, y]
            if (r + g + b) / 3 < 215:
                last_content_y = y
                break
        if last_content_y == y:
            break

    margin = max(52, int(height * 0.055))
    crop_bottom = min(height, last_content_y + margin)
    min_bottom = min(height, top + int(height * 0.22))
    return image.crop((0, 0, width, max(crop_bottom, min_bottom)))


def start_terminal(command: str, done_path: Path, cwd: Path) -> None:
    done_path.unlink(missing_ok=True)
    wrapped = (
        "clear; "
        "printf '\\033]0;Codex Week1 Screenshot\\007'; "
        f"cd {shell_quote(str(cwd))}; "
        f"{command}; "
        f"touch {shell_quote(str(done_path))}; "
        "printf '\\n[done] Screenshot captured from real Terminal output.\\n'; "
        "sleep 8; exit"
    )
    left, top, right, bottom = TERMINAL_BOUNDS_POINTS
    script = "\n".join(
        [
            'tell application "Terminal"',
            "  activate",
            f"  do script {applescript_quote(wrapped)}",
            f"  set bounds of front window to {{{left}, {top}, {right}, {bottom}}}",
            "end tell",
        ]
    )
    run(["osascript", "-e", script])


def wait_for_done(done_path: Path, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if done_path.exists():
            time.sleep(0.8)
            return
        time.sleep(0.3)
    raise TimeoutError(f"Terminal command did not finish within {timeout}s: {done_path}")


def capture_one(item: CaptureCommand, output_dir: Path, cwd: Path) -> Path:
    output_path = output_dir / f"terminal_{item.name}.png"
    full_path = output_dir / f"terminal_{item.name}_full.png"
    done_path = output_dir / f".terminal_{item.name}.done"

    start_terminal(item.command, done_path, cwd)
    wait_for_done(done_path)
    run(["screencapture", "-x", str(full_path)])
    crop_terminal(full_path, output_path)
    full_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)
    print(f"Wrote {output_path}")
    return output_path


def build_commands(python_bin: str) -> list[CaptureCommand]:
    py = shell_quote(python_bin)
    return [
        CaptureCommand(
            name="dataset_exploration",
            title="Dataset Exploration",
            command=(
                "printf '$ python demo/stage1_task2_2_dataset_exploration.py --download --data-dir data --report-dir reports\\n\\n'; "
                f"{py} demo/stage1_task2_2_dataset_exploration.py --download --data-dir data --report-dir reports; "
                "printf '\\n$ python -c \"print dataset summary\"\\n'; "
                f"{py} -c \"import json; d=json.load(open('reports/stage1_task2_2_dataset_summary.json')); "
                "print('CelebA images:', d['celeba']['num_images']); "
                "print('CelebA identities:', d['celeba'].get('num_identities')); "
                "print('LFW images:', d['lfw']['people_images']); "
                "print('LFW identities:', d['lfw']['people_identities']); "
                "print('LFW pairs:', d['lfw']['pairs'])\""
            ),
        ),
        CaptureCommand(
            name="mmdet_detection",
            title="MMDetection Summary",
            command=(
                "printf '$ python -c \"summarize mmdet_face_detection_summary.json\"\\n\\n'; "
                f"{py} -c \"import json, pathlib; d=json.load(open('reports/assets/detection/mmdet_face_detection_summary.json')); "
                "print('model:', d['model']); print('prompt:', d['texts']); "
                "[print(pathlib.Path(i['image']).name, 'faces=', i['num_detections'], 'label=', i['detections'][0]['label']) for i in d['images']]\""
            ),
        ),
        CaptureCommand(
            name="lfw_verification",
            title="LFW Verification Summary",
            command=(
                "printf '$ python -c \"summarize lfw_insightface_verification_summary.json\"\\n\\n'; "
                f"{py} -c \"import json, pathlib; d=json.load(open('reports/assets/evaluation/lfw_insightface_verification_summary.json')); "
                "print('valid_pairs:', d['valid_pairs']); print('failed_pairs:', d['failed_pairs']); "
                "print('mean_accuracy:', d['mean_accuracy']); print('std_accuracy:', d['std_accuracy']); print('AUC:', d['auc']); "
                "[print(pathlib.Path(i['image']).name, 'landmark_faces=', i['num_faces']) for i in d['landmark_visualizations']]\""
            ),
        ),
        CaptureCommand(
            name="assets_tree",
            title="Assets Tree",
            command=(
                "printf '$ find reports/assets -maxdepth 3 -type f | sort\\n\\n'; "
                "find reports/assets -maxdepth 3 -type f ! -name '.DS_Store' | sort | sed 's#reports/assets/##'"
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/assets/weekly/week1"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON", "python"))
    return parser.parse_args()


def main() -> None:
    if not shutil.which("osascript") or not shutil.which("screencapture"):
        raise SystemExit("This screenshot helper requires macOS osascript and screencapture.")

    args = parse_args()
    cwd = Path.cwd()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for item in build_commands(args.python_bin):
        capture_one(item, args.out_dir, cwd)


if __name__ == "__main__":
    main()
