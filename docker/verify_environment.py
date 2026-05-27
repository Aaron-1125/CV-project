#!/usr/bin/env python3
"""Verify the unified Docker CV environment for all project stages."""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> None:
    import cv2
    import datasets
    import huggingface_hub
    import insightface
    import mmcv
    import mmdet
    import mmengine
    import numpy as np
    import onnxruntime
    import pyarrow
    import reportlab
    import torch
    import torchmetrics
    import torchvision
    import transformers
    import webdataset
    from mmengine.config import Config

    print("python", sys.version.split()[0])
    print("platform", platform.platform())
    print("torch", torch.__version__)
    print("torchvision", torchvision.__version__)
    print("opencv", cv2.__version__)
    print("onnxruntime", onnxruntime.__version__)
    print("insightface", getattr(insightface, "__version__", "unknown"))
    print("datasets", datasets.__version__)
    print("huggingface_hub", huggingface_hub.__version__)
    print("pyarrow", pyarrow.__version__)
    print("torchmetrics", torchmetrics.__version__)
    print("transformers", transformers.__version__)
    print("webdataset", webdataset.__version__)
    print("reportlab", reportlab.Version)
    print("mmengine", mmengine.__version__)
    print("mmcv", mmcv.__version__)
    print("mmdet", mmdet.__version__)
    try:
        import mmpose
    except ImportError:
        mmpose = None
    print("mmpose", mmpose.__version__ if mmpose else "not installed")
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

    stage2_config = Path("stage-2/configs/mmdet/ssd300_widerface_smoke.py")
    if stage2_config.exists():
        cfg = Config.fromfile(stage2_config)
        assert cfg.model.bbox_head.num_classes == 1
        print("stage2_config", stage2_config)
    task4_config = Path("stage-2/configs/task4_mmpose/td-hm_hrnetv2-w18_300w_full_gpu.py")
    if task4_config.exists():
        cfg = Config.fromfile(task4_config)
        assert cfg.model.head.out_channels == 68
        print("stage2_task4_config", task4_config)
    task5_config = Path("stage-2/configs/task5_arcface/resnet50_arcface_ms1mv3_subset_gpu.py")
    if task5_config.exists():
        cfg = Config.fromfile(task5_config)
        assert cfg.model.embedding_size == 512
        assert cfg.loss.margin == 0.5
        print("stage2_task5_config", task5_config)
    print("unified docker environment ok")


if __name__ == "__main__":
    main()
