# CV Project Stage 2

阶段二任务 3.x：人脸检测模型训练。交付物只放在本目录内，原始数据、模型权重和训练输出由 `.gitignore` 排除。

## 目录约定

- `code/prepare/`：WIDER FACE 下载、VOC 转换、smoke 子集和公开验证图导出。
- `code/train/`：MMDetection Runner 训练/测试封装。
- `code/evaluate/`：IoU=0.5 AP、precision、recall 统计和检测结果可视化。
- `configs/mmdet/`：SSD300 WIDER FACE smoke/full 配置。
- `reports/assets/dataset/`：数据集样本和分布图。
- `reports/assets/training/`：训练曲线与训练相关图。
- `reports/assets/evaluation/`：评估图表。
- `reports/assets/inputs/wider_val/`：公开 WIDER FACE validation 测试图。
- `reports/assets/detection/`：检测结果可视化。
- `reports/summaries/`：数据、训练、评估摘要 JSON。
- `data/`、`checkpoints/`、`work_dirs/`：本地数据/权重/训练输出，不提交 Git。

## 运行顺序

准备 WIDER FACE：

```bash
python code/prepare/stage2_task3_2_prepare_widerface.py \
  --download \
  --data-dir data \
  --report-dir reports \
  --smoke-train 128 \
  --smoke-val 64
```

Docker 命令从 `CV project/` 项目根目录运行，stage-2 任务通过 `-w /workspace/stage-2` 指定工作目录。

根目录保留两套 Docker 环境：

- `docker/Dockerfile` / `bytedance-cv:project`：CPU/Mac 友好的 smoke 环境，用于快速验证链路。
- `docker/Dockerfile.gpu` / `bytedance-cv:gpu`：NVIDIA GPU 环境，用于 CUDA 加速的完整 WIDER FACE 训练。

构建 Docker 环境：

```bash
cd "/Users/aaron/Documents/字节实习/task/CV project"
docker build --platform linux/amd64 -t bytedance-cv:project -f docker/Dockerfile .
```

构建并验证 NVIDIA GPU 环境：

```bash
docker compose build stage2-gpu
docker compose run --rm stage2-gpu python /workspace/docker/verify_environment.py
```

GPU 验证输出中 `cuda_available` 应为 `True`。

Smoke 训练：

```bash
docker run --platform linux/amd64 --rm \
  -v "$PWD":/workspace \
  -w /workspace/stage-2 \
  bytedance-cv:project \
  python code/train/stage2_task3_2_run_mmdet.py train \
    --config configs/mmdet/ssd300_widerface_smoke.py \
    --work-dir work_dirs/ssd300_widerface_smoke \
    --summary-out reports/summaries/widerface_smoke_train_summary.json \
    --loss-plot-out reports/assets/training/smoke_loss_curve.png
```

Smoke 评估与检测可视化：

```bash
docker run --platform linux/amd64 --rm \
  -v "$PWD":/workspace \
  -w /workspace/stage-2 \
  bytedance-cv:project \
  python code/evaluate/stage2_task3_3_evaluate_widerface.py \
    --config configs/mmdet/ssd300_widerface_smoke.py \
    --checkpoint work_dirs/ssd300_widerface_smoke/epoch_1.pth \
    --data-root data/WIDERFace \
    --ann-file smoke_val.txt \
    --split val \
    --input-dir reports/assets/inputs/wider_val \
    --out-dir reports/assets/detection \
    --summary-out reports/summaries/widerface_smoke_eval_summary.json \
    --device cpu \
    --score-thr 0.05 \
    --iou-thr 0.5 \
    --visualize-count 4 \
    --vis-top-k 5 \
    --metrics-plot-out reports/assets/evaluation/widerface_smoke_eval_metrics.png
```

可视化说明：

- `reports/assets/inputs/wider_val/` 中默认只导出 4 张公开 WIDER FACE validation 图片，用于报告展示；训练规模不是 4 张。
- 当前 smoke 实验实际使用 `128` 张 train、`64` 张 val；全量训练配置使用 WIDER FACE train `12,337` 张。
- `input_XX_<image_id>.jpg` 与 `detection_XX_<image_id>.jpg` 一一对应，检测图是在同一张 input 原图上叠加框生成。
- 检测图中橙色框为 WIDER FACE GT 标注，绿色框为 smoke 模型预测框；绿色框质量差是 1 epoch 短训和随机初始化导致的预期现象。

GPU 全量训练入口：

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/train/stage2_task3_2_run_mmdet.py train \
    --config configs/mmdet/ssd300_widerface_full_gpu.py \
    --work-dir work_dirs/ssd300_widerface_full_gpu \
    --summary-out reports/summaries/widerface_full_train_summary.json \
    --loss-plot-out reports/assets/training/full_loss_curve.png
```

GPU 全量评估与检测可视化：

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/evaluate/stage2_task3_3_evaluate_widerface.py \
    --config configs/mmdet/ssd300_widerface_full_gpu.py \
    --checkpoint work_dirs/ssd300_widerface_full_gpu/epoch_24.pth \
    --data-root data/WIDERFace \
    --ann-file val.txt \
    --split val \
    --input-dir reports/assets/inputs/wider_val \
    --out-dir reports/assets/detection \
    --summary-out reports/summaries/widerface_full_eval_summary.json \
    --device cuda:0 \
    --score-thr 0.05 \
    --iou-thr 0.5 \
    --visualize-count 4 \
    --vis-top-k 20 \
    --metrics-plot-out reports/assets/evaluation/widerface_full_eval_metrics.png
```

## 报告入口

- `reports/task3_1_detection_algorithms.md`
- `reports/stage2_task3_face_detection_training_report.md`
