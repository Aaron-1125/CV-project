# 阶段一任务1.3：Docker 项目环境配置报告

日期：2026-05-20

## 任务目标

创建统一 Docker 项目环境，用于复现阶段一 CV 实验依赖，并将 PyTorch、OpenCV、MMDetection 放入同一个镜像。

## 镜像配置

| 项目 | 配置 |
| --- | --- |
| 镜像名称 | `bytedance-cv:stage1` |
| 平台 | `linux/amd64` |
| 基础镜像 | `python:3.10-slim` |
| Python | 3.10 |
| PyTorch | 2.0.0+cpu |
| torchvision | 0.15.1+cpu |
| OpenCV | 4.11.0 |
| MMEngine | 0.10.7 |
| MMCV | 2.0.0 |
| MMDetection | 3.3.0 |

说明：Apple Silicon 本机仍保留轻量 Conda 环境；MMDetection 相关实验统一使用 Docker 镜像运行。

## 已提交文件

- `docker/Dockerfile`
- `docker/requirements-docker.txt`
- `docker/verify_environment.py`
- `docker-compose.yml`
- `demo/00_hello_docker.py`

## 使用命令

构建镜像：

```bash
docker build --platform linux/amd64 -t bytedance-cv:stage1 -f docker/Dockerfile .
```

验证环境：

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1
```

运行 Hello World：

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1 python demo/00_hello_docker.py
```

运行 OpenCV 灰度化程序：

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1 python demo/01_opencv_grayscale.py
```

运行 PyTorch smoke test：

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1 \
  python demo/01_pytorch_minimal_training.py \
  --dataset fake \
  --epochs 1 \
  --batch-size 32 \
  --fake-train-size 64 \
  --fake-val-size 32 \
  --num-workers 0
```

启动 JupyterLab：

```bash
docker compose up jupyter
```

访问地址：

```text
http://localhost:8888/lab?token=bytedance-cv
```

## 验收结论

任务1.3已完成：

- Dockerfile 已创建。
- Docker Compose 配置已创建。
- 统一镜像已包含 PyTorch、OpenCV、MMEngine、MMCV、MMDetection。
- 环境验证、Hello World、OpenCV 灰度化、PyTorch smoke test 已通过。

下一步等待确认后，再继续后续任务点。
