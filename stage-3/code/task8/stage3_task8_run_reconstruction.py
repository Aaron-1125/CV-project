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
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--outputs", nargs="+", choices=["2d_sparse", "2d_dense", "3d", "depth", "pncc", "pose", "uv_tex", "ply", "obj"], default=None)
    parser.add_argument("--python", default=None, help="Python executable used for official_subprocess mode.")
    parser.add_argument("--force", action="store_true", help="Replace each sample reconstruction directory before running.")
    parser.add_argument("--resume", action="store_true", default=None, help="Load previous reconstruction summary and update records incrementally.")
    parser.add_argument("--skip-existing", action="store_true", default=None, help="Skip samples whose archived outputs already match the current input.")
    parser.add_argument("--continue-on-error", action="store_true", default=None, help="Record per-sample failures and continue with remaining samples.")
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
        suffix = source.suffix or (".obj" if opt == "obj" else ".ply" if opt == "ply" else ".jpg")
        target_name = "{}{}".format(base_name, suffix) if idx == 0 else "{}_{:02d}{}".format(base_name, idx + 1, suffix)
        target = sample_dir / target_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
        archived.append(str(target))
    return archived


def expected_output_path(sample_dir: Path, opt: str) -> Path:
    suffix = ".obj" if opt == "obj" else ".ply" if opt == "ply" else ".jpg"
    return sample_dir / "{}{}".format(OUTPUT_TARGET_NAMES.get(opt, "official_{}".format(opt)), suffix)


def sample_meta_path(sample_dir: Path) -> Path:
    return sample_dir / "sample_meta.json"


def read_sample_meta(sample_dir: Path) -> Dict[str, Any]:
    path = sample_meta_path(sample_dir)
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def existing_outputs_complete(sample_dir: Path, image_path: Path, outputs: List[str]) -> bool:
    meta = read_sample_meta(sample_dir)
    if meta.get("input_filename") != image_path.name:
        return False
    return all(expected_output_path(sample_dir, opt).exists() for opt in outputs)


def skipped_record(
    sample: Dict[str, Any],
    image_path: Path,
    sample_dir: Path,
    outputs: List[str],
    runner: str,
    backend_request: str,
    backend: str,
    backend_reason: str,
    mode: str,
    mode_reason: str,
) -> Dict[str, Any]:
    output_records = {}
    for opt in outputs:
        path = expected_output_path(sample_dir, opt)
        output_records[opt] = {
            "opt": opt,
            "success": path.exists(),
            "paths": [str(path)] if path.exists() else [],
            "skipped": True,
        }
    record: Dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "input_image": str(image_path),
        "source_path": sample.get("source_path"),
        "output_dir": str(sample_dir),
        "runner": runner,
        "requested_backend": backend_request,
        "selected_backend": backend,
        "backend_reason": backend_reason,
        "selected_mode": mode,
        "mode_reason": mode_reason,
        "outputs": output_records,
        "success": all(row.get("success") for row in output_records.values()),
        "skipped": True,
        "status": "skipped_existing",
    }
    attach_primary_paths(record)
    return record


def attach_primary_paths(sample_record: Dict[str, Any]) -> None:
    sample_record["obj_path"] = (sample_record["outputs"].get("obj", {}).get("paths") or [None])[0]
    sample_record["landmark_path"] = (sample_record["outputs"].get("2d_sparse", {}).get("paths") or [None])[0]
    sample_record["overlay_path"] = (sample_record["outputs"].get("3d", {}).get("paths") or [None])[0]
    sample_record["pose_path"] = (sample_record["outputs"].get("pose", {}).get("paths") or [None])[0]


def write_sample_meta(sample_dir: Path, sample_record: Dict[str, Any], outputs: List[str]) -> None:
    payload = {
        "sample_id": sample_record.get("sample_id"),
        "input_image": sample_record.get("input_image"),
        "input_filename": Path(str(sample_record.get("input_image", ""))).name,
        "source_path": sample_record.get("source_path"),
        "success": sample_record.get("success", False),
        "status": sample_record.get("status"),
        "official_outputs": outputs,
        "obj_path": sample_record.get("obj_path"),
        "landmark_path": sample_record.get("landmark_path"),
        "overlay_path": sample_record.get("overlay_path"),
        "pose_path": sample_record.get("pose_path"),
    }
    write_json(sample_meta_path(sample_dir), payload)


def sample_failure_reason(sample_record: Dict[str, Any]) -> str:
    if sample_record.get("failure_reason"):
        return str(sample_record["failure_reason"])
    failed = [
        "{}: {}".format(opt, row.get("failure_reason", "failed"))
        for opt, row in sample_record.get("outputs", {}).items()
        if not row.get("success")
    ]
    return "; ".join(failed) if failed else "unknown"


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


def load_prepared_samples(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = prepare_summary_path(cfg)
    if not path.exists():
        raise FileNotFoundError("Missing prepared samples summary: {}. Run stage3_task8_prepare_samples.py first.".format(path))
    prepare = read_json(path)
    samples = list(prepare.get("samples", []))
    if not samples:
        raise ValueError("No prepared samples found in {}".format(path))
    for sample in samples:
        staged = Path(sample["staged_path"])
        if not staged.exists():
            raise FileNotFoundError("Prepared input image is missing: {}".format(staged))
    return samples


def select_work_items(samples: List[Dict[str, Any]], start_index: int, end_index: Optional[int], max_samples: Optional[int]) -> List[Dict[str, Any]]:
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    end = len(samples) if end_index is None else min(end_index, len(samples))
    if end < start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index")
    selected = samples[start_index:end]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def previous_records_by_id(cfg: Dict[str, Any], resume: bool) -> Dict[str, Dict[str, Any]]:
    path = reconstruction_summary_path(cfg)
    if not resume or not path.exists():
        return {}
    try:
        previous = read_json(path)
    except Exception:
        return {}
    return {str(row.get("sample_id")): row for row in previous.get("records", []) if row.get("sample_id")}


def build_summary_payload(
    cfg: Dict[str, Any],
    repo_info: Dict[str, Any],
    runner: str,
    backend_request: str,
    backend: str,
    backend_reason: str,
    mode: str,
    mode_reason: str,
    outputs: List[str],
    all_samples: List[Dict[str, Any]],
    records_by_id: Dict[str, Dict[str, Any]],
    start_index: int,
    end_index: Optional[int],
    max_samples: Optional[int],
) -> Dict[str, Any]:
    ordered_records = [records_by_id[row["sample_id"]] for row in all_samples if row["sample_id"] in records_by_id]
    success_count = sum(1 for row in ordered_records if row.get("success"))
    failure_records = [row for row in ordered_records if row.get("status") == "failed" or (row.get("success") is False and row.get("status") != "pending")]
    skipped_count = sum(1 for row in ordered_records if row.get("skipped"))
    sample_count = len(all_samples)
    success_rate = float(success_count) / float(sample_count) if sample_count else 0.0
    return {
        "task": cfg.get("task_name"),
        "ready": success_count > 0,
        "runner": runner,
        "repo": repo_info,
        "requested_backend": backend_request,
        "selected_backend": backend,
        "backend_reason": backend_reason,
        "selected_mode": mode,
        "mode_reason": mode_reason,
        "official_outputs": outputs,
        "sample_count": sample_count,
        "record_count": len(ordered_records),
        "processed_count": len(ordered_records),
        "pending_count": max(sample_count - len(ordered_records), 0),
        "success_count": success_count,
        "failure_count": len(failure_records),
        "skipped_count": skipped_count,
        "success_rate": success_rate,
        "start_index": start_index,
        "end_index": end_index,
        "max_samples": max_samples,
        "failed_records": failure_records,
        "records": ordered_records,
    }


def write_progress_summary(
    cfg: Dict[str, Any],
    repo_info: Dict[str, Any],
    runner: str,
    backend_request: str,
    backend: str,
    backend_reason: str,
    mode: str,
    mode_reason: str,
    outputs: List[str],
    all_samples: List[Dict[str, Any]],
    records_by_id: Dict[str, Dict[str, Any]],
    start_index: int,
    end_index: Optional[int],
    max_samples: Optional[int],
) -> Dict[str, Any]:
    payload = build_summary_payload(
        cfg,
        repo_info,
        runner,
        backend_request,
        backend,
        backend_reason,
        mode,
        mode_reason,
        outputs,
        all_samples,
        records_by_id,
        start_index,
        end_index,
        max_samples,
    )
    write_json(reconstruction_summary_path(cfg), payload)
    return payload


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runner = args.runner or str(cfg_get(cfg, "reconstruction", "runner", "official_subprocess"))
    backend_request = args.backend or str(cfg_get(cfg, "reconstruction", "backend", "auto"))
    outputs = args.outputs or list(cfg_get(cfg, "reconstruction", "official_outputs", DEFAULT_OFFICIAL_OUTPUTS))
    resume = bool(cfg_get(cfg, "reconstruction", "resume", True)) if args.resume is None else bool(args.resume)
    skip_existing = bool(cfg_get(cfg, "reconstruction", "skip_existing", True)) if args.skip_existing is None else bool(args.skip_existing)
    continue_on_error = bool(cfg_get(cfg, "reconstruction", "continue_on_error", True)) if args.continue_on_error is None else bool(args.continue_on_error)
    configured_max = cfg_get(cfg, "reconstruction", "max_samples", None)
    max_samples = args.max_samples if args.max_samples is not None else configured_max
    if max_samples is not None:
        max_samples = int(max_samples)
    repo_info = ensure_3ddfa_repo(cfg)
    repo = three_ddfa_repo(cfg)
    backend, backend_reason = choose_backend(cfg, backend_request, repo_info)
    mode, mode_reason = choose_mode(cfg, args.mode, backend)
    python_bin = args.python or python_executable()
    all_samples = load_prepared_samples(cfg)
    samples = select_work_items(all_samples, args.start_index, args.end_index, max_samples)
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    reconstruction_dir(cfg).mkdir(parents=True, exist_ok=True)

    records_by_id = previous_records_by_id(cfg, resume)
    for sample in samples:
        sample_id = sample["sample_id"]
        image_path = Path(sample["staged_path"]).absolute()
        out_dir = reconstruction_dir(cfg) / sample_id
        if args.force and out_dir.exists():
            shutil.rmtree(str(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        if skip_existing and existing_outputs_complete(out_dir, image_path, outputs):
            sample_record = skipped_record(
                sample,
                image_path,
                out_dir,
                outputs,
                runner,
                backend_request,
                backend,
                backend_reason,
                mode,
                mode_reason,
            )
            records_by_id[sample_id] = sample_record
            print("Skipping {} because archived official outputs already match {}".format(sample_id, image_path.name))
            write_progress_summary(
                cfg,
                repo_info,
                runner,
                backend_request,
                backend,
                backend_reason,
                mode,
                mode_reason,
                outputs,
                all_samples,
                records_by_id,
                args.start_index,
                args.end_index,
                max_samples,
            )
            continue
        sample_record: Dict[str, Any] = {
            "sample_id": sample_id,
            "input_image": str(image_path),
            "source_path": sample.get("source_path"),
            "output_dir": str(out_dir),
            "runner": runner,
            "requested_backend": backend_request,
            "selected_backend": backend,
            "backend_reason": backend_reason,
            "selected_mode": mode,
            "mode_reason": mode_reason,
            "outputs": {},
            "success": False,
            "skipped": False,
            "status": "running",
        }
        print("Running official 3DDFA_V2 for {} ({})".format(sample_id, image_path.name))
        try:
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
            sample_record["status"] = "success" if sample_record["success"] else "failed"
            if not sample_record["success"]:
                sample_record["failure_reason"] = sample_failure_reason(sample_record)
        except Exception as exc:
            sample_record["success"] = False
            sample_record["status"] = "failed"
            sample_record["failure_reason"] = "{}: {}".format(type(exc).__name__, str(exc))
            if not continue_on_error:
                records_by_id[sample_id] = sample_record
                write_sample_meta(out_dir, sample_record, outputs)
                write_progress_summary(
                    cfg,
                    repo_info,
                    runner,
                    backend_request,
                    backend,
                    backend_reason,
                    mode,
                    mode_reason,
                    outputs,
                    all_samples,
                    records_by_id,
                    args.start_index,
                    args.end_index,
                    max_samples,
                )
                raise
        attach_primary_paths(sample_record)
        records_by_id[sample_id] = sample_record
        write_sample_meta(out_dir, sample_record, outputs)
        write_progress_summary(
            cfg,
            repo_info,
            runner,
            backend_request,
            backend,
            backend_reason,
            mode,
            mode_reason,
            outputs,
            all_samples,
            records_by_id,
            args.start_index,
            args.end_index,
            max_samples,
        )

    payload = write_progress_summary(
        cfg,
        repo_info,
        runner,
        backend_request,
        backend,
        backend_reason,
        mode,
        mode_reason,
        outputs,
        all_samples,
        records_by_id,
        args.start_index,
        args.end_index,
        max_samples,
    )
    if not payload["ready"] and not continue_on_error:
        raise SystemExit("No official 3DDFA_V2 .obj output was collected. See {}".format(reconstruction_summary_path(cfg)))


if __name__ == "__main__":
    main()
