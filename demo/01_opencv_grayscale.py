#!/usr/bin/env python3
"""Simple OpenCV image processing demo.

Reads a sample image, converts it to grayscale, and saves the result.
"""

from __future__ import annotations

from pathlib import Path

import cv2


def main() -> None:
    input_path = Path("sample_inputs/01036a162ec6e859bb81218ad79dc1aa.jpg")
    output_dir = Path("outputs")
    output_path = output_dir / "opencv_grayscale_sample.jpg"

    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    output_dir.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), gray)
    if not ok:
        raise RuntimeError(f"Could not save grayscale image: {output_path}")

    print(f"input: {input_path}")
    print(f"bgr_shape: {image.shape}")
    print(f"gray_shape: {gray.shape}")
    print(f"gray_mean: {gray.mean():.2f}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()

