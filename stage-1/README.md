# CV Project

阶段一基础环境、工具学习与人脸识别基础交付仓库。

## 交付内容

- 环境配置文档：`environment.md`
- Conda 环境定义：`environment.yml`
- Docker 镜像配置：`docker/Dockerfile`
- Docker Compose 配置：`docker-compose.yml`
- Docker Hello World：`code/00_hello_docker.py`
- OpenCV 灰度化 demo：`code/01_opencv_grayscale.py`
- PyTorch smoke test：`code/01_pytorch_minimal_training.py`
- CelebA/LFW 数据集探索：`code/stage1_task2_2_dataset_exploration.py`
- MMDetection 开放词人脸检测：`code/stage1_task2_3_mmdet_face_detection.py`
- InsightFace 关键点定位与 LFW 验证：`code/stage1_task2_4_landmarks_and_lfw_eval.py`
- 环境验证脚本：`docker/verify_environment.py`
- 验收报告：`reports/`

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
docker run --platform linux/amd64 --rm bytedance-cv:stage1 python code/00_hello_docker.py
```

运行 OpenCV 图像处理 demo：

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1 python code/01_opencv_grayscale.py
```

运行 PyTorch 最小训练 smoke test：

```bash
docker run --platform linux/amd64 --rm bytedance-cv:stage1 \
  python code/01_pytorch_minimal_training.py \
  --dataset fake \
  --epochs 1
```

## 阶段一 2.x 运行顺序

以下命令默认在 `stage-1/` 目录下运行。原始数据会进入 `data/`，模型权重会进入 `checkpoints/`，这两个目录已被 Git 忽略。

1. 完整下载并探索 CelebA/LFW：

```bash
python code/stage1_task2_2_dataset_exploration.py \
  --download \
  --data-dir data \
  --report-dir reports
```

2. 使用 MMDetection GroundingDINO 开放词模型检测人脸：

```bash
python code/stage1_task2_3_mmdet_face_detection.py \
  --input-dir ../../../sample_inputs \
  --out-dir reports/assets \
  --checkpoint-dir checkpoints/mmdet \
  --texts "face . human face ." \
  --score-thr 0.25
```

如果本机环境没有 MMDetection，可使用 Docker 镜像运行，并额外挂载样例图目录：

```bash
docker run --platform linux/amd64 --rm \
  -v "$PWD":/workspace \
  -v "/Users/aaron/Documents/字节实习/sample_inputs":/sample_inputs:ro \
  -w /workspace \
  bytedance-cv:stage1 \
  sh -lc "python -m pip install 'huggingface_hub>=0.19,<1.0' transformers==4.37.2 >/tmp/pip-mmdet.log && \
  python code/stage1_task2_3_mmdet_face_detection.py \
    --input-dir /sample_inputs \
    --out-dir reports/assets \
    --checkpoint-dir checkpoints/mmdet \
    --texts 'face . human face .' \
    --score-thr 0.25 \
    --device cpu"
```

3. 使用 InsightFace 关键点定位并在 LFW 10-fold pairs 上验证：

```bash
python code/stage1_task2_4_landmarks_and_lfw_eval.py \
  --download \
  --data-dir data \
  --out-dir reports/assets \
  --landmark-input-dir ../../../sample_inputs
```

阶段报告入口：

```text
reports/stage1_face_basics_dataset_report.md
```
