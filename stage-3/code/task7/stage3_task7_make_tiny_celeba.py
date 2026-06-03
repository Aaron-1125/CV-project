#!/usr/bin/env python3
"""Create a tiny CelebA-like dataset for local Task7 smoke tests."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

from stage3_task7_common import SELECTED_ATTRS, write_json


ALL_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/tiny_celeba"))
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    image_dir = args.out_dir / "img_align_celeba"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, dict[str, int]]] = []
    hair_cycle = ["Black_Hair", "Blond_Hair", "Brown_Hair", None]
    for idx in range(args.count):
        filename = f"{idx + 1:06d}.jpg"
        attrs = {name: -1 for name in ALL_ATTRS}
        hair = hair_cycle[idx % len(hair_cycle)]
        if hair:
            attrs[hair] = 1
        attrs["Male"] = 1 if idx % 2 else -1
        attrs["Young"] = 1 if (idx // 2) % 2 else -1
        attrs["Smiling"] = 1 if idx % 3 == 0 else -1
        base = (180 + (idx * 13) % 45, 150 + (idx * 17) % 65, 135 + (idx * 19) % 80)
        image = Image.new("RGB", (178, 218), base)
        draw = ImageDraw.Draw(image)
        face = (60, 58, 118, 144)
        draw.ellipse(face, fill=(230, 190, 165), outline=(80, 60, 50), width=2)
        hair_color = {
            "Black_Hair": (30, 25, 25),
            "Blond_Hair": (220, 190, 95),
            "Brown_Hair": (100, 65, 35),
            None: (150, 150, 150),
        }[hair]
        draw.pieslice((48, 42, 130, 110), 180, 360, fill=hair_color)
        draw.ellipse((73, 90, 78, 95), fill=(20, 20, 20))
        draw.ellipse((100, 90, 105, 95), fill=(20, 20, 20))
        draw.arc((78, 103, 104, 124), 0, 180, fill=(120, 45, 60), width=2)
        if attrs["Male"] == 1:
            draw.rectangle((78, 132, 104, 138), fill=(70, 40, 30))
        image.save(image_dir / filename, quality=95)
        rows.append((filename, attrs))

    attr_path = args.out_dir / "list_attr_celeba.txt"
    with attr_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(rows)}\n")
        handle.write(" ".join(ALL_ATTRS) + "\n")
        for filename, attrs in rows:
            values = [str(attrs[name]) for name in ALL_ATTRS]
            handle.write(filename + " " + " ".join(values) + "\n")
    write_json(
        args.out_dir / "tiny_celeba_summary.json",
        {
            "images": len(rows),
            "image_dir": str(image_dir),
            "attr_path": str(attr_path),
            "selected_attrs": SELECTED_ATTRS,
        },
    )


if __name__ == "__main__":
    main()
