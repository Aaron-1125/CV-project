#!/usr/bin/env python3
"""Verify the unified Docker CV environment for all project stages."""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> None:
    import cv2
    import datasets
    import easydict
    import hf_transfer
    import hf_xet
    import huggingface_hub
    import insightface
    import mmcv
    import mmdet
    import mmengine
    import numpy as np
    import onnx
    import onnxruntime
    import pyarrow
    import reportlab
    import torch
    import torchmetrics
    import torchvision
    import transformers
    import webdataset
    import tensorboard
    from mmengine.config import Config

    print("python", sys.version.split()[0])
    print("platform", platform.platform())
    print("torch", torch.__version__)
    print("torchvision", torchvision.__version__)
    print("opencv", cv2.__version__)
    print("onnx", onnx.__version__)
    print("onnxruntime", onnxruntime.__version__)
    print("insightface", getattr(insightface, "__version__", "unknown"))
    print("datasets", datasets.__version__)
    print("easydict", easydict.__version__ if hasattr(easydict, "__version__") else "installed")
    print("huggingface_hub", huggingface_hub.__version__)
    print("hf_transfer", getattr(hf_transfer, "__version__", "installed"))
    print("hf_xet", getattr(hf_xet, "__version__", "installed"))
    print("pyarrow", pyarrow.__version__)
    print("torchmetrics", torchmetrics.__version__)
    print("transformers", transformers.__version__)
    print("webdataset", webdataset.__version__)
    print("tensorboard", tensorboard.__version__)
    print("reportlab", reportlab.Version)
    try:
        import mxnet
    except Exception as exc:  # noqa: BLE001 - Py3.10 Docker can skip MXNet; AutoDL Py3.8 installs it.
        mxnet = None
        print("mxnet", f"not available ({exc.__class__.__name__}: {exc})")
    else:
        print("mxnet", mxnet.__version__)
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
    task5_insightface_config = Path("stage-2/configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py")
    if task5_insightface_config.exists():
        cfg = Config.fromfile(task5_insightface_config)
        assert cfg.official.num_classes == 93431
        assert cfg.official.num_image == 5179510
        assert cfg.official.network == "r50"
        print("stage2_task5_insightface_config", task5_insightface_config)
    task6_source_script = Path("stage-2/code/task6/stage2_task6_prepare_source_model.py")
    task6_onnx_script = Path("stage-2/code/task6/stage2_task6_3_export_onnx.py")
    if task6_source_script.exists() and task6_onnx_script.exists():
        print("stage2_task6_scripts", "available")
    print("unified docker environment ok")


if __name__ == "__main__":
    main()
