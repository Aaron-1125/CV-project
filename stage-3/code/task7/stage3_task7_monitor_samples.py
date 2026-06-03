#!/usr/bin/env python3
"""Generate fixed-sample monitoring grids for saved StarGAN checkpoints."""

from __future__ import annotations

import argparse

from stage3_task7_common import (
    asset_dir,
    generate_fixed_samples,
    load_config,
    sorted_generator_checkpoints,
    summary_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task7_stargan/a800_full.py")
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def checkpoint_iter(path) -> int:
    return int(path.name.split("-")[0])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoints = sorted_generator_checkpoints(cfg)
    if not checkpoints:
        raise FileNotFoundError("No *-G.ckpt files found in the Task7 model dir.")
    if args.latest:
        checkpoints = checkpoints[-1:]
    elif not args.all_checkpoints:
        checkpoints = [path for path in checkpoints if checkpoint_iter(path) % 10000 == 0]
    records = []
    for checkpoint in checkpoints:
        iters = checkpoint_iter(checkpoint)
        output_dir = asset_dir(cfg) / "monitor" / f"iter_{iters}"
        summary = generate_fixed_samples(cfg, checkpoint, f"monitor_{iters}", output_dir, device=args.device)
        records.append({"iters": iters, "checkpoint": str(checkpoint), "grid": summary["grid"]})
    write_json(
        summary_dir(cfg) / "monitor_samples_summary.json",
        {
            "checkpoints": records,
            "count": len(records),
        },
    )


if __name__ == "__main__":
    main()

