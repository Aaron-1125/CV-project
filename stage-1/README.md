# CV Project

阶段一基础环境、工具学习与人脸识别基础交付仓库。

## 交付内容

- 环境配置文档：`environment.md`
- Conda 环境定义：`environment.yml`
- Docker 镜像配置：`docker/Dockerfile`
- Docker Compose 配置：`docker-compose.yml`
- Docker Hello World：`demo/00_hello_docker.py`
- OpenCV 灰度化 demo：`demo/01_opencv_grayscale.py`
- PyTorch smoke test：`demo/01_pytorch_minimal_training.py`
- CelebA/LFW 数据集探索：`demo/stage1_task2_2_dataset_exploration.py`
- MMDetection 开放词人脸检测：`demo/stage1_task2_3_mmdet_face_detection.py`
- InsightFace 关键点定位与 LFW 验证：`demo/stage1_task2_4_landmarks_and_lfw_eval.py`
- 环境验证脚本：`docker/verify_environment.py`
- 验收报告、结果图：`reports/`
- 第一周周报：`reports/weekly/week1_report_2026-05-21.md`
- 第一周周报 PDF：`reports/weekly/week1_report_2026-05-21.pdf`

## 目录约定

- `reports/` 存放可提交的报告、摘要 JSON 和交付说明。
- `reports/assets/dataset/` 存放 CelebA/LFW 数据分布和样本网格。
- `reports/assets/inputs/public_lfw/` 存放从 LFW 公开数据集导出的检测/关键点测试图。
- `reports/assets/detection/` 存放 MMDetection 人脸检测结果图和摘要 JSON。
- `reports/assets/landmarks/` 存放 InsightFace 关键点可视化结果图。
- `reports/assets/evaluation/` 存放 LFW 验证 ROC、分数分布图和评估摘要 JSON。
- `reports/assets/weekly/` 存放周报中引用的真实终端截图。
- `reports/weekly/` 存放每周周报 Markdown 源文件和导出的 PDF。
- `data/` 存放原始数据集和缓存，`checkpoints/` 存放模型权重，二者不提交 Git。

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

## 阶段一 2.x 运行顺序

以下命令默认在 `stage-1/` 目录下运行。原始数据会进入 `data/`，模型权重会进入 `checkpoints/`，这两个目录已被 Git 忽略。

1. 完整下载并探索 CelebA/LFW：

```bash
python demo/stage1_task2_2_dataset_exploration.py \
  --download \
  --data-dir data \
  --report-dir reports
```

第 1 步会从 LFW 公开数据集中导出检测/关键点测试图到 `reports/assets/inputs/public_lfw/`。

2. 使用 MMDetection GroundingDINO 开放词模型检测公开 LFW 测试图：

```bash
python demo/stage1_task2_3_mmdet_face_detection.py \
  --input-dir reports/assets/inputs/public_lfw \
  --out-dir reports/assets/detection \
  --checkpoint-dir checkpoints/mmdet \
  --texts "face . human face ." \
  --score-thr 0.25
```

如果本机环境没有 MMDetection，可使用 Docker 镜像运行：

```bash
docker run --platform linux/amd64 --rm \
  -v "$PWD":/workspace \
  -w /workspace \
  bytedance-cv:stage1 \
  sh -lc "python -m pip install 'huggingface_hub>=0.19,<1.0' transformers==4.37.2 >/tmp/pip-mmdet.log && \
  python demo/stage1_task2_3_mmdet_face_detection.py \
    --input-dir reports/assets/inputs/public_lfw \
    --out-dir reports/assets/detection \
    --checkpoint-dir checkpoints/mmdet \
    --texts 'face . human face .' \
    --score-thr 0.25 \
    --device cpu"
```

3. 使用 InsightFace 关键点定位并在 LFW 10-fold pairs 上验证：

```bash
python demo/stage1_task2_4_landmarks_and_lfw_eval.py \
  --download \
  --data-dir data \
  --out-dir reports/assets/evaluation \
  --landmark-input-dir reports/assets/inputs/public_lfw \
  --landmark-out-dir reports/assets/landmarks
```

阶段报告入口：

```text
reports/stage1_face_basics_dataset_report.md
```

## 第一周周报 PDF

生成真实终端截图：

```bash
python demo/capture_week1_terminal_screenshots.py \
  --python-bin /Users/aaron/Documents/字节实习/.conda/bytedance-cv/bin/python
```

导出 PDF：

```bash
python demo/export_weekly_report_pdf.py \
  --source reports/weekly/week1_report_2026-05-21.md \
  --output reports/weekly/week1_report_2026-05-21.pdf
```

渲染检查：

```bash
mkdir -p tmp/pdfs
pdftoppm -png reports/weekly/week1_report_2026-05-21.pdf tmp/pdfs/week1_report
```
