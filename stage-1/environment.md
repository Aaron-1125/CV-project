# 环境配置文档

## 1. 环境概览

| 项目 | 配置 |
| --- | --- |
| 操作系统 | macOS arm64 |
| 本机环境管理 | Anaconda |
| 本机 Python 环境 | `/Users/aaron/Documents/字节实习/.conda/bytedance-cv` |
| Python 版本 | 3.11.15 |
| Docker 镜像 | `bytedance-cv:stage1` |
| Docker 镜像平台 | linux/amd64 |
| Docker Engine | 29.4.3 linux/aarch64 |
| Docker Compose | v5.1.3 |

## 2. 本机依赖版本

| 依赖 | 版本 |
| --- | --- |
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| torchaudio | 2.11.0 |
| opencv-python | 4.13.0 |
| onnxruntime | 1.24.4 |
| insightface | 0.7.3 |
| matplotlib | 3.10.8 |
| numpy | 2.4.4 |
| gdown | 6.0.0 |
| datasets | 4.8.5 |
| huggingface_hub | 0.36.2 |
| transformers | 4.37.2 |

设备后端：

- Apple MPS：可用
- CUDA：不可用

## 3. Docker 镜像依赖版本

| 依赖 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| torch | 2.0.0+cpu |
| torchvision | 0.15.1+cpu |
| opencv-python | 4.11.0 |
| onnxruntime | 1.23.2 |
| mmengine | 0.10.7 |
| mmcv | 2.0.0 |
| mmdet | 3.3.0 |
| gdown | 需按 `docker/requirements-docker.txt` 安装 |
| datasets | 需按 `docker/requirements-docker.txt` 安装 |
| huggingface_hub | 需按 `docker/requirements-docker.txt` 安装 |
| transformers | 需按 `docker/requirements-docker.txt` 安装 |

## 4. 环境配置步骤

### 4.1 本机 Conda 环境

```bash
conda env create --prefix ./.conda/bytedance-cv --file environment.yml
conda activate "/Users/aaron/Documents/字节实习/.conda/bytedance-cv"
```

如需更新环境：

```bash
conda env update --prefix "/Users/aaron/Documents/字节实习/.conda/bytedance-cv" --file environment.yml --prune
```

### 4.2 Jupyter Kernel

```bash
conda run --prefix "/Users/aaron/Documents/字节实习/.conda/bytedance-cv" \
  python -m ipykernel install --user \
  --name bytedance-cv \
  --display-name "Python (bytedance-cv)"
```

### 4.3 Docker 环境

```bash
docker build --platform linux/amd64 -t bytedance-cv:stage1 -f docker/Dockerfile .
```

## 5. 验证命令

### 5.1 本机环境验证

```bash
conda run --no-capture-output --prefix "/Users/aaron/Documents/字节实习/.conda/bytedance-cv" \
  python -c "import torch, cv2, onnxruntime, insightface, datasets, transformers; print(torch.__version__, cv2.__version__, onnxruntime.__version__, insightface.__version__, datasets.__version__, transformers.__version__)"
```

### 5.2 Docker 环境验证

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1
```

### 5.3 Hello World 程序

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1 python code/00_hello_docker.py
```

### 5.4 OpenCV 图像处理程序

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1 python code/01_opencv_grayscale.py
```

### 5.5 PyTorch 训练验证

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1 \
  python code/01_pytorch_minimal_training.py \
  --dataset fake \
  --epochs 1
```

### 5.6 阶段一人脸识别基础任务

```bash
python code/stage1_task2_2_dataset_exploration.py --download --data-dir data --report-dir reports
python code/stage1_task2_3_mmdet_face_detection.py --input-dir ../../../sample_inputs --out-dir reports/assets
python code/stage1_task2_4_landmarks_and_lfw_eval.py --download --data-dir data --out-dir reports/assets --landmark-input-dir ../../../sample_inputs
```

## 6. 验收结果

| 验收项 | 状态 |
| --- | --- |
| 本机 Conda 环境 | 通过 |
| Jupyter Kernel | 通过 |
| Docker 镜像构建 | 通过 |
| Docker 环境验证 | 通过 |
| Hello World 程序 | 通过 |
| OpenCV 灰度化程序 | 通过 |
| PyTorch smoke test | 通过 |
| MMDetection 依赖验证 | 通过 |
| CelebA/LFW 数据探索脚本 | 待完整数据下载完成后复核 |
| MMDetection 开放词人脸检测脚本 | 待 checkpoint 下载完成后复核 |
| InsightFace 关键点与 LFW 验证脚本 | 待 LFW 下载完成后复核 |

说明：本机 Conda 环境用于轻量开发；Docker 镜像用于统一复现 PyTorch、OpenCV 和 MMDetection 环境。
