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

## Stage2 Task 4.x

Task 4.x deliverables are isolated under `reports/task4/` and do not overwrite the task 3.x WIDER FACE outputs.

- Task 4.1 report: `reports/task4/task4_1_landmark_algorithms.md`
- Task 4.x report: `reports/task4/stage2_task4_landmark_alignment_report.md`
- Task 4.x code: `code/task4/`
- Task 4.x config: `configs/task4_mmpose/`
- 8GB fallback config: `configs/task4_mmpose/td-hm_hrnetv2-w18_300w_full_gpu_bs16.py`
- Task 4.x data: `data/task4_300w/`
- Task 4.x work dirs: `work_dirs/task4/`

Prepare 300W:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task4/stage2_task4_2_prepare_300w.py \
    --download \
    --data-dir data/task4_300w \
    --report-dir reports/task4
```

The prepare script accepts either the official multipart archive or an unpacked
Kaggle `ibug_300W_large_face_landmark_dataset` directory placed under
`data/task4_300w/raw/`. The Kaggle package does not include the official
`Test/` images, so training and valid/common/challenge evaluation can run while
the official test split is recorded as skipped.

Train HRNetv2-W18:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task4/stage2_task4_run_mmpose.py train \
    --config configs/task4_mmpose/td-hm_hrnetv2-w18_300w_full_gpu.py \
    --work-dir work_dirs/task4/hrnetv2_w18_300w_full \
    --summary-out reports/task4/summaries/300w_full_train_summary.json \
    --loss-plot-out reports/task4/assets/training/300w_full_loss_curve.png \
    --device cuda:0
```

If the 8GB GPU runs out of memory with batch size 32, rerun the same command with
`--config configs/task4_mmpose/td-hm_hrnetv2-w18_300w_full_gpu_bs16.py`; that
fallback uses batch size 16 and Adam lr `6.25e-5` while keeping full 300W data
and 60 epochs.

Evaluate and align:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task4/stage2_task4_run_mmpose.py test \
    --config configs/task4_mmpose/td-hm_hrnetv2-w18_300w_full_gpu.py \
    --checkpoint work_dirs/task4/hrnetv2_w18_300w_full/best.pth \
    --summary-out reports/task4/summaries/300w_full_eval_summary.json \
    --metrics-plot-out reports/task4/assets/evaluation/300w_nme_metrics.png \
    --device cuda:0

docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task4/stage2_task4_3_align_faces.py \
    --config configs/task4_mmpose/td-hm_hrnetv2-w18_300w_full_gpu.py \
    --checkpoint work_dirs/task4/hrnetv2_w18_300w_full/best.pth \
    --data-dir data/task4_300w/mmpose/300w \
    --out-dir reports/task4/assets/alignment \
    --summary-out reports/task4/summaries/300w_alignment_summary.json \
    --visualize-count 8 \
    --device cuda:0
```

## Stage2 Task 5.x

Task 5.x deliverables are isolated under `reports/task5/` and do not overwrite
task 3.x detection outputs or task 4.x landmark/alignment outputs.

- Task 5.1 report: `reports/task5/task5_1_face_recognition_algorithms.md`
- Task 5.x report: `reports/task5/stage2_task5_arcface_training_report.md`
- Task 5.x code: `code/task5/`
- Task 5.x config: `configs/task5_arcface/`
- Task 5.x data: `data/task5_ms1mv3_dense/` and `data/task5_lfw/`
- Task 5.x work dirs: `work_dirs/task5/`

AutoDL A800 80GB recommendation: run the dense 800k MS1MV3 subset directly.
Use at least a 150-200 GB data disk. The config
`configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py` is tuned for one
A800/A100 80GB GPU with batch size 512, 12 workers, AMP, 60 epochs, and a
36-hour training budget. It saves `best.pth` and `last.pth` only to avoid
filling the cloud data disk with per-epoch checkpoints.

Prepare MS1MV3 dense 800k:

```bash
python code/task5/stage2_task5_2_prepare_ms1mv3.py \
  --dataset gaunernst/ms1mv3-wds-gz \
  --data-dir data/task5_ms1mv3_dense \
  --report-dir reports/task5 \
  --mode subset \
  --max-images 800000 \
  --max-identities 20000 \
  --images-per-identity-cap 80 \
  --output-tag ms1mv3_dense \
  --max-stream-retries 20
```

Prepare LFW:

```bash
python code/task5/stage2_task5_3_prepare_lfw.py \
  --data-dir data/task5_lfw \
  --report-dir reports/task5
```

Train ResNet50/IResNet50 + ArcFace:

```bash
python code/task5/stage2_task5_run_arcface.py train \
  --config configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py \
  --work-dir work_dirs/task5/resnet50_arcface_ms1mv3_dense \
  --summary-out reports/task5/summaries/ms1mv3_dense_train_summary.json \
  --loss-plot-out reports/task5/assets/training/ms1mv3_dense_loss_acc_curve.png \
  --device cuda:0
```

Evaluate LFW:

```bash
python code/task5/stage2_task5_run_arcface.py eval-lfw \
  --config configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py \
  --checkpoint work_dirs/task5/resnet50_arcface_ms1mv3_dense/best.pth \
  --lfw-dir data/task5_lfw \
  --summary-out reports/task5/summaries/lfw_eval_summary.json \
  --roc-plot-out reports/task5/assets/evaluation/lfw_roc_curve.png \
  --device cuda:0
```

If 800k still misses the 98.5% LFW target, rerun preparation with 1M+ images
and resume the same work dir.
