#!/usr/bin/env python3
"""Verify the Docker CV environment for stage 1 task 1.3."""

from __future__ import annotations

import platform
import sys

import cv2
import mmcv
import mmdet
import mmengine
import numpy as np
import onnxruntime
import torch
import torchvision


def main() -> None:
    print("python", sys.version.split()[0])
    print("platform", platform.platform())
    print("torch", torch.__version__)
    print("torchvision", torchvision.__version__)
    print("opencv", cv2.__version__)
    print("onnxruntime", onnxruntime.__version__)
    print("mmengine", mmengine.__version__)
    print("mmcv", mmcv.__version__)
    print("mmdet", mmdet.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("mps_available", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, image.shape[1], dtype=np.uint8)
    image[:, :, 1] = np.linspace(255, 0, image.shape[0], dtype=np.uint8)[:, None]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("sample_image", "synthetic_gradient")
    print("bgr_shape", image.shape)
    print("gray_shape", gray.shape)
    print("gray_mean", round(float(np.mean(gray)), 2))


if __name__ == "__main__":
    main()
