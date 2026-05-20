# 阶段一任务1.1：环境配置与验证报告

日期：2026-05-19

## 任务目标

配置 Python/Anaconda 开发环境，确认 PyTorch、OpenCV、Jupyter 等基础工具可用，并明确 MMDetection 的安装与运行策略。

## 本机环境

- 操作系统：macOS-26.3.1-arm64-arm-64bit
- Conda 环境路径：`/Users/aaron/Documents/字节实习/.conda/bytedance-cv`
- Python：`3.11.15`
- Jupyter kernel：`bytedance-cv`
- PyTorch 后端：
  - Apple MPS：可用
  - CUDA：不可用

## 已验证依赖版本

| 组件 | 版本/状态 |
| --- | --- |
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| torchaudio | 2.11.0 |
| opencv-python | 4.13.0 |
| onnxruntime | 1.24.4 |
| insightface | 0.7.3 |
| matplotlib | 3.10.8 |
| numpy | 2.4.4 |
| mmdet | 未安装 |
| mmcv | 未安装 |
| mmengine | 未安装 |

## MMDetection 策略

本机轻量环境不安装 `mmdet/mmcv/mmengine`；MMDetection 已在任务1.3的 Docker 镜像 `bytedance-cv:stage1` 中统一配置。

原因：

- 当前本机是 Apple Silicon/macOS，MMCV/MMDetection 的本地安装和编译更容易遇到平台与二进制包兼容问题。
- 现有 `.conda/bytedance-cv` 已经可稳定运行 PyTorch、OpenCV、InsightFace 和 Jupyter，不应为了 MMDetection 破坏当前可用环境。
- MMDetection 官方推荐通过 MIM 安装 `mmengine`、`mmcv>=2.0.0` 和 `mmdet`；MMCV 官方文档也提醒不要在同一环境同时安装 `mmcv` 和 `mmcv-lite`。

参考：

- [MMDetection Get Started](https://mmdetection.readthedocs.io/en/main/get_started.html)
- [MMCV Installation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)

## 验证命令与结果

### 1. 核心依赖版本检查

```bash
conda run --no-capture-output --prefix "/Users/aaron/Documents/字节实习/.conda/bytedance-cv" \
  python -c "import torch, torchvision, torchaudio, cv2, onnxruntime, insightface, matplotlib, numpy; print(torch.__version__, cv2.__version__)"
```

结果：核心依赖均可正常 import，版本见上表。

### 2. PyTorch 最小训练闭环

```bash
conda run --no-capture-output --prefix "/Users/aaron/Documents/字节实习/.conda/bytedance-cv" \
  python demo/01_pytorch_minimal_training.py \
  --dataset fake \
  --epochs 1 \
  --batch-size 32 \
  --fake-train-size 64 \
  --fake-val-size 32 \
  --num-workers 0
```

结果：

```text
Using device: mps
epoch=1 train_loss=2.4593 train_acc=0.078 val_loss=2.2779 val_acc=0.156
Saved metrics to outputs/minimal_training_metrics.csv
```

说明：训练、评估、指标保存链路正常。

### 3. OpenCV 图像读取与灰度化

```bash
conda run --no-capture-output --prefix "/Users/aaron/Documents/字节实习/.conda/bytedance-cv" \
  python -c "from pathlib import Path; import cv2; p=Path('sample_inputs/01036a162ec6e859bb81218ad79dc1aa.jpg'); img=cv2.imread(str(p)); assert img is not None; gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); print(img.shape, gray.shape, gray.dtype)"
```

结果：

```text
bgr_shape (1279, 1706, 3)
gray_shape (1279, 1706)
gray_dtype uint8
gray_mean 107.57
```

说明：OpenCV 图像读取、颜色空间转换和基础数组处理正常。

### 4. Jupyter kernel 检查

```bash
jupyter kernelspec list
```

结果：

```text
Available kernels:
  bytedance-cv    /Users/aaron/Library/Jupyter/kernels/bytedance-cv
  python3         /opt/anaconda3/share/jupyter/kernels/python3
```

说明：Jupyter 中可以选择 `bytedance-cv` 内核进行实验。

## 验收结论

任务1.1已完成：

- 本机 Anaconda 隔离环境可用。
- PyTorch 训练闭环可用，并能使用 Apple MPS。
- OpenCV 基础图像处理可用。
- Jupyter Notebook/Lab 实验内核可用。
- MMDetection 不在本机轻量环境安装，已按 Docker 路线处理。

下一步等待确认后进入任务1.2：学习 Git 基本操作，创建 GitHub 仓库，用于代码版本控制和提交。
