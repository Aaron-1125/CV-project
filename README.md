# CV Project

阶段一基础环境与工具交付仓库。

## 交付内容

- 环境配置文档：`environment.md`
- Conda 环境定义：`environment.yml`
- Docker 镜像配置：`docker/Dockerfile`
- Docker Compose 配置：`docker-compose.yml`
- Docker Hello World：`demo/00_hello_docker.py`
- OpenCV 灰度化 demo：`demo/01_opencv_grayscale.py`
- PyTorch smoke test：`demo/01_pytorch_minimal_training.py`
- 环境验证脚本：`docker/verify_environment.py`
- 验收报告：`notes/`

## Docker 运行

构建镜像：

```bash
docker build --platform linux/amd64 -t bytedance-cv:stage1 -f docker/Dockerfile .
```

运行环境验证：

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1
```

运行 Hello World：

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1 python demo/00_hello_docker.py
```

运行 OpenCV 图像处理 demo：

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1 python demo/01_opencv_grayscale.py
```

运行 PyTorch 最小训练 smoke test：

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1 \
  python demo/01_pytorch_minimal_training.py \
  --dataset fake \
  --epochs 1
```

说明：Docker 镜像 `bytedance-cv:stage1` 已统一包含 PyTorch、OpenCV、MMEngine、MMCV 和 MMDetection。
