#!/usr/bin/env python3
"""Train or test official StarGAN through the Stage3 Task7 wrapper."""

from __future__ import annotations

import argparse
import sys

from stage3_task7_common import (
    asset_dir,
    cfg_get,
    checkpoint_path_for_iters,
    ensure_stargan_repo,
    generate_fixed_samples,
    load_config,
    prepared_attr_path,
    prepared_image_dir,
    run_command,
    selected_attrs,
    stargan_repo,
    summary_dir,
    work_dir,
    write_json,
)


def str_bool(value: bool) -> str:
    return "true" if value else "false"


def build_train_command(cfg: dict, resume_iters: int | None) -> list[str]:
    attrs = selected_attrs(cfg)
    wd = work_dir(cfg)
    cmd = [
        sys.executable,
        "main.py",
        "--mode", "train",
        "--dataset", str(cfg_get(cfg, "model", "dataset", "CelebA")),
        "--image_size", str(cfg_get(cfg, "model", "image_size", 128)),
        "--celeba_crop_size", str(cfg_get(cfg, "model", "celeba_crop_size", 178)),
        "--c_dim", str(cfg_get(cfg, "model", "c_dim", 5)),
        "--g_conv_dim", str(cfg_get(cfg, "model", "g_conv_dim", 64)),
        "--d_conv_dim", str(cfg_get(cfg, "model", "d_conv_dim", 64)),
        "--g_repeat_num", str(cfg_get(cfg, "model", "g_repeat_num", 6)),
        "--d_repeat_num", str(cfg_get(cfg, "model", "d_repeat_num", 6)),
        "--lambda_cls", str(cfg_get(cfg, "model", "lambda_cls", 1)),
        "--lambda_rec", str(cfg_get(cfg, "model", "lambda_rec", 10)),
        "--lambda_gp", str(cfg_get(cfg, "model", "lambda_gp", 10)),
        "--batch_size", str(cfg_get(cfg, "train", "batch_size", 16)),
        "--num_iters", str(cfg_get(cfg, "train", "num_iters", 200000)),
        "--num_iters_decay", str(cfg_get(cfg, "train", "num_iters_decay", 100000)),
        "--g_lr", str(cfg_get(cfg, "train", "g_lr", 0.0001)),
        "--d_lr", str(cfg_get(cfg, "train", "d_lr", 0.0001)),
        "--n_critic", str(cfg_get(cfg, "train", "n_critic", 5)),
        "--beta1", str(cfg_get(cfg, "train", "beta1", 0.5)),
        "--beta2", str(cfg_get(cfg, "train", "beta2", 0.999)),
        "--num_workers", str(cfg_get(cfg, "train", "num_workers", 8)),
        "--use_tensorboard", str_bool(bool(cfg_get(cfg, "train", "use_tensorboard", False))),
        "--celeba_image_dir", str(prepared_image_dir(cfg)),
        "--attr_path", str(prepared_attr_path(cfg)),
        "--log_dir", str(wd / "logs"),
        "--model_save_dir", str(wd / "models"),
        "--sample_dir", str(wd / "samples"),
        "--result_dir", str(wd / "results"),
        "--log_step", str(cfg_get(cfg, "train", "log_step", 100)),
        "--sample_step", str(cfg_get(cfg, "train", "sample_step", 5000)),
        "--model_save_step", str(cfg_get(cfg, "train", "model_save_step", 10000)),
        "--lr_update_step", str(cfg_get(cfg, "train", "lr_update_step", 1000)),
        "--selected_attrs",
        *attrs,
    ]
    if resume_iters is not None:
        cmd.extend(["--resume_iters", str(resume_iters)])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/task7_stargan/a800_full.py")
    train.add_argument("--resume-iters", type=int, default=None)
    test = subparsers.add_parser("test")
    test.add_argument("--config", default="configs/task7_stargan/a800_full.py")
    test.add_argument("--test-iters", type=int, default=None)
    test.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    repo = ensure_stargan_repo(cfg)
    if args.mode == "train":
        wd = work_dir(cfg)
        for subdir in ["logs", "models", "samples", "results"]:
            (wd / subdir).mkdir(parents=True, exist_ok=True)
        cmd = build_train_command(cfg, args.resume_iters)
        summary = run_command(cmd, cwd=stargan_repo(cfg), summary_out=summary_dir(cfg) / "stargan_train_command_summary.json")
        summary["resume_iters"] = args.resume_iters
        summary["work_dir"] = str(wd)
        summary["stargan_repo"] = repo
        write_json(summary_dir(cfg) / "stargan_train_summary.json", summary)
    else:
        test_iters = args.test_iters or int(cfg_get(cfg, "train", "final_test_iters", cfg_get(cfg, "train", "num_iters", 200000)))
        checkpoint = checkpoint_path_for_iters(cfg, test_iters)
        output_dir = asset_dir(cfg) / "self_trained" / f"iter_{test_iters}"
        fixed = generate_fixed_samples(cfg, checkpoint, f"self_trained_{test_iters}", output_dir, device=args.device)
        write_json(
            summary_dir(cfg) / "self_trained_test_summary.json",
            {
                "ready": True,
                "test_iters": test_iters,
                "checkpoint": str(checkpoint),
                "fixed_grid": fixed["grid"],
                "fixed_summary": str(output_dir / f"self_trained_{test_iters}_fixed_summary.json"),
            },
        )


if __name__ == "__main__":
    main()

