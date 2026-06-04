#!/usr/bin/env python3
"""Run official 3DDFA_V2 demo.py and collect Task8 reconstruction outputs."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from stage3_task8_common import (
    DEFAULT_OFFICIAL_OUTPUTS,
    cfg_get,
    changed_files,
    choose_backend,
    choose_mode,
    ensure_3ddfa_repo,
    file_snapshot,
    official_results_dir,
    prepare_summary_path,
    python_executable,
    read_json,
    reconstruction_dir,
    reconstruction_summary_path,
    scan_official_outputs,
    summary_dir,
    three_ddfa_repo,
    truncate_text,
    write_json,
    load_config,
)


OUTPUT_TARGET_NAMES = {
    "2d_sparse": "official_2d_sparse",
    "3d": "official_3d_overlay",
    "pose": "official_pose",
    "obj": "official_mesh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task8_3dface/a800_3ddfa_v2.py")
    parser.add_argument("--runner", choices=["official_subprocess", "python_api"], default=None)
    parser.add_argument("--backend", choices=["auto", "onnx", "pth"], default=None)
    parser.add_argument("--mode", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--outputs", nargs="+", choices=["2d_sparse", "2d_dense", "3d", "depth", "pncc", "pose", "uv_tex", "ply", "obj"], default=None)
    parser.add_argument("--python", default=None, help="Python executable used for official_subprocess mode.")
    parser.add_argument("--force", action="store_true", help="Replace each sample reconstruction directory before running.")
    return parser.parse_args()


def show_flag_value(cfg: Dict[str, Any]) -> str:
    return "true" if bool(cfg_get(cfg, "reconstruction", "show_flag", False)) else "false"


def official_config_arg(cfg: Dict[str, Any]) -> str:
    configured = str(cfg_get(cfg, "third_party", "official_config", "configs/mb1_120x120.yml"))
    return configured


def build_official_command(
    cfg: Dict[str, Any],
    image_path: Path,
    opt: str,
    backend: str,
    mode: str,
    python_bin: str,
) -> List[str]:
    cmd = [
        python_bin,
        "demo.py",
        "-c",
        official_config_arg(cfg),
        "-f",
        str(image_path),
        "-m",
        mode,
        "-o",
        opt,
        "--show_flag",
        show_flag_value(cfg),
    ]
    if backend == "onnx":
        cmd.append("--onnx")
    return cmd


def run_subprocess(cmd: List[str], cwd: Path) -> Dict[str, Any]:
    started = time.time()
    result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return {
        "command": cmd,
        "cwd": str(cwd),
        "returncode": result.returncode,
        "seconds": round(time.time() - started, 3),
        "stdout": truncate_text(result.stdout),
        "stderr": truncate_text(result.stderr),
    }


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def run_python_api_demo(
    cfg: Dict[str, Any],
    repo: Path,
    image_path: Path,
    opt: str,
    backend: str,
    mode: str,
) -> Dict[str, Any]:
    started = time.time()
    stdout = io.StringIO()
    stderr = io.StringIO()
    pseudo_command = [
        "python_api",
        "demo.py::main",
        "-c",
        official_config_arg(cfg),
        "-f",
        str(image_path),
        "-m",
        mode,
        "-o",
        opt,
        "--show_flag",
        show_flag_value(cfg),
    ]
    if backend == "onnx":
        pseudo_command.append("--onnx")
    old_sys_path = list(sys.path)
    returncode = 0
    try:
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        with pushd(repo), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            namespace = runpy.run_path(str(repo / "demo.py"))
            namespace["main"](
                SimpleNamespace(
                    config=official_config_arg(cfg),
                    img_fp=str(image_path),
                    mode=mode,
                    opt=opt,
                    show_flag=False,
                    onnx=(backend == "onnx"),
                )
            )
    except SystemExit as exc:
        returncode = int(exc.code or 0) if isinstance(exc.code, int) else 1
    except Exception as exc:
        returncode = 1
        stderr.write("{}: {}\n".format(type(exc).__name__, str(exc)))
    finally:
        sys.path = old_sys_path
    return {
        "command": pseudo_command,
        "cwd": str(repo),
        "returncode": returncode,
        "seconds": round(time.time() - started, 3),
        "stdout": truncate_text(stdout.getvalue()),
        "stderr": truncate_text(stderr.getvalue()),
    }


def attempt_plan(initial_backend: str, initial_mode: str, repo_info: Dict[str, Any], cfg: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    attempts: List[Tuple[str, str, str]] = [(initial_backend, initial_mode, "primary")]
    allow_backend = bool(cfg_get(cfg, "reconstruction", "allow_backend_fallback", True))
    allow_cpu = bool(cfg_get(cfg, "reconstruction", "allow_cpu_fallback", True))
    has_pth = bool(repo_info.get("preferred_checkpoint_exists"))
    if allow_backend and initial_backend == "onnx" and has_pth:
        attempts.append(("pth", initial_mode, "fallback from ONNX to PyTorch pth"))
    if allow_cpu and initial_mode == "gpu":
        attempts.append((initial_backend, "cpu", "fallback from gpu to cpu"))
        if allow_backend and initial_backend == "onnx" and has_pth:
            attempts.append(("pth", "cpu", "fallback from ONNX/gpu to PyTorch pth/cpu"))
    deduped: List[Tuple[str, str, str]] = []
    seen = set()
    for backend, mode, reason in attempts:
        key = (backend, mode)
        if key not in seen:
            deduped.append((backend, mode, reason))
            seen.add(key)
    return deduped


def archive_outputs(files: List[Path], sample_dir: Path, opt: str) -> List[str]:
    archived: List[str] = []
    base_name = OUTPUT_TARGET_NAMES.get(opt, "official_{}".format(opt))
    for idx, source in enumerate(files):
        suffix = source.suffix or (".obj" if opt == "obj" else ".jpg")
        target_name = "{}{}".format(base_name, suffix) if idx == 0 else "{}_{:02d}{}".format(base_name, idx + 1, suffix)
        target = sample_dir / target_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
        archived.append(str(target))
    return archived


def run_one_official_output(
    cfg: Dict[str, Any],
    repo: Path,
    repo_info: Dict[str, Any],
    image_path: Path,
    sample_dir: Path,
    opt: str,
    runner: str,
    backend: str,
    mode: str,
    python_bin: str,
) -> Dict[str, Any]:
    results_dir = official_results_dir(repo)
    results_dir.mkdir(parents=True, exist_ok=True)
    basename = image_path.stem
    attempts_summary = []
    archived_paths: List[str] = []
    for attempt_backend, attempt_mode, reason in attempt_plan(backend, mode, repo_info, cfg):
        before = file_snapshot(results_dir, basename)
        if runner == "official_subprocess":
            cmd = build_official_command(cfg, image_path, opt, attempt_backend, attempt_mode, python_bin)
            result = run_subprocess(cmd, repo)
        else:
            result = run_python_api_demo(cfg, repo, image_path, opt, attempt_backend, attempt_mode)
        after = file_snapshot(results_dir, basename)
        new_or_changed = changed_files(before, after)
        outputs = scan_official_outputs(results_dir, basename, opt, before)
        archived_paths = archive_outputs(outputs, sample_dir, opt) if outputs else []
        attempt_record = {
            "backend": attempt_backend,
            "mode": attempt_mode,
            "reason": reason,
            "result": result,
            "official_changed_files": [str(p) for p in new_or_changed],
            "matched_official_outputs": [str(p) for p in outputs],
            "archived_outputs": archived_paths,
            "success": result["returncode"] == 0 and bool(archived_paths),
        }
        attempts_summary.append(attempt_record)
        if attempt_record["success"]:
            return {
                "opt": opt,
                "success": True,
                "backend": attempt_backend,
                "mode": attempt_mode,
                "paths": archived_paths,
                "attempts": attempts_summary,
            }
    return {
        "opt": opt,
        "success": False,
        "backend": backend,
        "mode": mode,
        "paths": archived_paths,
        "attempts": attempts_summary,
        "failure_reason": "Official demo did not return a matched {} output.".format(opt),
    }


def load_prepared_samples(cfg: Dict[str, Any], max_samples: Optional[int]) -> List[Dict[str, Any]]:
    path = prepare_summary_path(cfg)
    if not path.exists():
        raise FileNotFoundError("Missing prepared samples summary: {}. Run stage3_task8_prepare_samples.py first.".format(path))
    prepare = read_json(path)
    samples = list(prepare.get("samples", []))
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("No prepared samples found in {}".format(path))
    for sample in samples:
        staged = Path(sample["staged_path"])
        if not staged.exists():
            raise FileNotFoundError("Prepared input image is missing: {}".format(staged))
    return samples


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runner = args.runner or str(cfg_get(cfg, "reconstruction", "runner", "official_subprocess"))
    backend_request = args.backend or str(cfg_get(cfg, "reconstruction", "backend", "auto"))
    outputs = args.outputs or list(cfg_get(cfg, "reconstruction", "official_outputs", DEFAULT_OFFICIAL_OUTPUTS))
    repo_info = ensure_3ddfa_repo(cfg)
    repo = three_ddfa_repo(cfg)
    backend, backend_reason = choose_backend(cfg, backend_request, repo_info)
    mode, mode_reason = choose_mode(cfg, args.mode, backend)
    python_bin = args.python or python_executable()
    samples = load_prepared_samples(cfg, args.max_samples)
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    reconstruction_dir(cfg).mkdir(parents=True, exist_ok=True)

    records = []
    for sample in samples:
        sample_id = sample["sample_id"]
        image_path = Path(sample["staged_path"]).absolute()
        out_dir = reconstruction_dir(cfg) / sample_id
        if args.force and out_dir.exists():
            shutil.rmtree(str(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_record: Dict[str, Any] = {
            "sample_id": sample_id,
            "input_image": str(image_path),
            "output_dir": str(out_dir),
            "runner": runner,
            "requested_backend": backend_request,
            "selected_backend": backend,
            "backend_reason": backend_reason,
            "selected_mode": mode,
            "mode_reason": mode_reason,
            "outputs": {},
        }
        print("Running official 3DDFA_V2 for {} ({})".format(sample_id, image_path.name))
        for opt in outputs:
            result = run_one_official_output(
                cfg=cfg,
                repo=repo,
                repo_info=repo_info,
                image_path=image_path,
                sample_dir=out_dir,
                opt=opt,
                runner=runner,
                backend=backend,
                mode=mode,
                python_bin=python_bin,
            )
            sample_record["outputs"][opt] = result
            if not result.get("success"):
                print("  {}: FAILED ({})".format(opt, result.get("failure_reason", "unknown")))
            else:
                print("  {}: {}".format(opt, ", ".join(result.get("paths", []))))
        required = ["2d_sparse", "3d", "pose", "obj"]
        sample_record["success"] = all(sample_record["outputs"].get(opt, {}).get("success") for opt in required if opt in outputs)
        sample_record["obj_path"] = (sample_record["outputs"].get("obj", {}).get("paths") or [None])[0]
        sample_record["landmark_path"] = (sample_record["outputs"].get("2d_sparse", {}).get("paths") or [None])[0]
        sample_record["overlay_path"] = (sample_record["outputs"].get("3d", {}).get("paths") or [None])[0]
        sample_record["pose_path"] = (sample_record["outputs"].get("pose", {}).get("paths") or [None])[0]
        records.append(sample_record)

    ready = any(row.get("obj_path") for row in records)
    payload = {
        "task": cfg.get("task_name"),
        "ready": ready,
        "runner": runner,
        "repo": repo_info,
        "requested_backend": backend_request,
        "selected_backend": backend,
        "backend_reason": backend_reason,
        "selected_mode": mode,
        "mode_reason": mode_reason,
        "official_outputs": outputs,
        "sample_count": len(records),
        "records": records,
    }
    write_json(reconstruction_summary_path(cfg), payload)
    if not ready:
        raise SystemExit("No official 3DDFA_V2 .obj output was collected. See {}".format(reconstruction_summary_path(cfg)))


if __name__ == "__main__":
    main()
