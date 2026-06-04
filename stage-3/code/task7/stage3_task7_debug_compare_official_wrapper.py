#!/usr/bin/env python3
"""Debug one wrapper StarGAN generation against the official test flow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from stage3_task7_common import (
    asset_dir,
    cfg_get,
    checkpoint_path_for_iters,
    create_official_target_tensors,
    image_transform,
    load_config,
    load_fixed_manifest,
    load_generator,
    safe_save_generated_image,
    selected_attrs,
    stargan_repo,
    tensor_label_to_ints,
    validate_target_label,
    write_json,
)
from stage3_task7_pretrained_sanity import ensure_pretrained_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task7_stargan/a800_full.py")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-kind", choices=["pretrained", "self"], default="pretrained")
    parser.add_argument("--test-iters", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--direction", default="Black_Hair")
    parser.add_argument("--pretrained-zip", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_checkpoint(cfg: Dict[str, Any], args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.checkpoint_kind == "pretrained":
        return ensure_pretrained_checkpoint(cfg, args.pretrained_zip)
    test_iters = args.test_iters or int(cfg_get(cfg, "train", "final_test_iters", cfg_get(cfg, "train", "num_iters", 200000)))
    return checkpoint_path_for_iters(cfg, test_iters)


def inspect_checkpoint(cfg: Dict[str, Any], checkpoint: Path) -> Dict[str, Any]:
    import torch

    repo = stargan_repo(cfg)
    if str(repo) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(repo))
    from model import Generator  # type: ignore

    generator = Generator(
        int(cfg_get(cfg, "model", "g_conv_dim", 64)),
        int(cfg_get(cfg, "model", "c_dim", 5)),
        int(cfg_get(cfg, "model", "g_repeat_num", 6)),
    )
    try:
        state = torch.load(str(checkpoint), map_location=lambda storage, loc: storage, weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint), map_location=lambda storage, loc: storage)
    model_state = generator.state_dict()
    missing = [key for key in model_state if key not in state]
    unexpected = [key for key in state if key not in model_state]
    shape_diffs = [
        {
            "key": key,
            "checkpoint_shape": list(state[key].shape),
            "model_shape": list(model_state[key].shape),
        }
        for key in state
        if key in model_state and tuple(state[key].shape) != tuple(model_state[key].shape)
    ]
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_keys": len(state),
        "model_keys": len(model_state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_diffs": shape_diffs,
    }


def save_denorm_tensor(tensor, path: Path) -> None:
    from torchvision.utils import save_image

    from stage3_task7_common import denorm_tensor

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denorm_tensor(tensor.detach().cpu()), str(path), nrow=1, padding=0)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    import torch

    cfg = load_config(args.config)
    attrs = selected_attrs(cfg)
    if args.direction not in attrs:
        raise ValueError(f"--direction must be one of {attrs}")
    manifest = load_fixed_manifest(cfg)
    samples = manifest["samples"]
    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise IndexError(f"--sample-index must be between 0 and {len(samples) - 1}")
    sample = samples[args.sample_index]
    image_dir = Path(manifest["image_dir"])
    checkpoint = resolve_checkpoint(cfg, args)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    device = args.device or str(cfg_get(cfg, "evaluation", "device", "cuda:0"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    output_dir = asset_dir(cfg) / "debug_official_wrapper" / checkpoint.stem / sample["sample_id"] / args.direction
    transform = image_transform(cfg, train=False)
    with Image.open(image_dir / sample["filename"]) as handle:
        image = handle.convert("RGB")
    x_real = transform(image).unsqueeze(0).to(device)
    c_org = torch.tensor([sample["label"]], dtype=torch.float32)
    target_tensors = create_official_target_tensors(c_org, attrs, device)
    direction_idx = attrs.index(args.direction)
    target_tensor = target_tensors[direction_idx]
    target_label = tensor_label_to_ints(target_tensor[0])
    validate_target_label(target_label, attrs, direction=args.direction)

    generator = load_generator(cfg, checkpoint, device)
    with torch.no_grad():
        fake = generator(x_real, target_tensor)[0]

    input_path = output_dir / "input_after_transform_denorm.jpg"
    fake_path = output_dir / "wrapper_fake.jpg"
    save_denorm_tensor(x_real[0], input_path)
    safe_save_generated_image(fake, fake_path)
    target_json = output_dir / "target_label.json"
    payload = {
        "sample": sample,
        "direction": args.direction,
        "selected_attrs": attrs,
        "source_label": sample["label"],
        "target_label": target_label,
        "target_tensor": target_tensor.detach().cpu().tolist(),
        "input_after_transform_denorm": str(input_path),
        "wrapper_fake": str(fake_path),
        "generator_mode": "train",
        "official_alignment": {
            "transform": "CenterCrop -> Resize -> ToTensor -> Normalize(0.5, 0.5)",
            "labels": "Mirrors Solver.create_labels for CelebA.",
            "save": "torchvision.utils.save_image(denorm(...), nrow=1, padding=0).",
        },
        "checkpoint_check": inspect_checkpoint(cfg, checkpoint),
    }
    write_json(target_json, payload)
    print(f"Checkpoint: {checkpoint}")
    print(f"Target tensor: {payload['target_tensor']}")
    print(f"Wrote {fake_path}")


if __name__ == "__main__":
    main()
