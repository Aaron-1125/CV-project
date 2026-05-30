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
- Task 5.x data: `data/task5_ms1mv3_full_recordio/` and `data/task5_lfw/`
- Task 5.x work dirs: `work_dirs/task5/`

The previous dense 800k custom trainer result is kept as a failed baseline
(`81.67%` LFW). The main route now uses the official InsightFace
`recognition/arcface_torch` pipeline with full MS1MV3 RecordIO.

Prepare LFW:

```bash
python code/task5/stage2_task5_3_prepare_lfw.py \
  --data-dir data/task5_lfw \
  --report-dir reports/task5
```

Prepare full MS1MV3 RecordIO and official LFW validation bin:

```bash
python code/task5/stage2_task5_4_prepare_ms1mv3_recordio.py \
  --download \
  --dataset gaunernst/ms1mv3-recordio \
  --data-dir data/task5_ms1mv3_full_recordio \
  --lfw-dir data/task5_lfw \
  --report-dir reports/task5
```

Set up the runtime-cloned official InsightFace source:

```bash
python code/task5/stage2_task5_5_run_insightface.py setup \
  --config configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py \
  --summary-out reports/task5/summaries/insightface_full_setup_summary.json
```

Train official ResNet50 + ArcFace on full MS1MV3:

```bash
python code/task5/stage2_task5_5_run_insightface.py train \
  --config configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py \
  --summary-out reports/task5/summaries/insightface_full_train_summary.json
```

Parse official LFW validation:

```bash
python code/task5/stage2_task5_5_run_insightface.py eval-summary \
  --config configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py \
  --checkpoint work_dirs/task5/insightface_ms1mv3_r50_full/model.pt \
  --summary-out reports/task5/summaries/insightface_full_lfw_eval_summary.json
```

Success requires `accuracy >= 0.985` and `target_met: true`.

## Stage2 Task 6.x

Task 6.x deliverables are isolated under `reports/task6/` and use the Task 5
first-version self-contained `IResNet50 + ArcFace` checkpoint from
`reports/task5/task5_cloud_results_8167.tar.gz`. The optional 6.4 ByteNN task is
not included.

- Task 6.1 report: `reports/task6/task6_1_optimization_methods.md`
- Task 6.x report: `reports/task6/stage2_task6_model_optimization_report.md`
- Task 6.x code: `code/task6/`
- Task 6.x summaries: `reports/task6/summaries/`
- Task 6.x models and ONNX artifacts: `work_dirs/task6/`
- Weekly report PDF: `reports/weekly/week2_report_2026-05-28.pdf`

Prepare the Task 5 first-version cloud checkpoint for Task 6:

```bash
python code/task6/stage2_task6_prepare_source_model.py \
  --cloud-archive reports/task5/task5_cloud_results_8167.tar.gz \
  --out-dir work_dirs/task6/source_arcface_8167 \
  --summary-out reports/task6/summaries/source_model_summary.json
```

Run dynamic quantization and LFW comparison:

```bash
python code/task6/stage2_task6_2_quantize_arcface.py \
  --config configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py \
  --checkpoint work_dirs/task6/source_arcface_8167/best.pth \
  --lfw-dir data/task5_lfw \
  --summary-out reports/task6/summaries/quantization_summary.json \
  --comparison-plot-out reports/task6/assets/evaluation/task6_quantization_comparison.png \
  --device cpu
```

Export ONNX and test ONNX Runtime inference:

```bash
python code/task6/stage2_task6_3_export_onnx.py \
  --config configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py \
  --checkpoint work_dirs/task6/source_arcface_8167/best.pth \
  --lfw-dir data/task5_lfw \
  --onnx-out work_dirs/task6/onnx/arcface_iresnet50_8167.onnx \
  --summary-out reports/task6/summaries/onnx_summary.json \
  --comparison-plot-out reports/task6/assets/evaluation/task6_onnx_comparison.png \
  --device cuda:0
```

Generate reports and weekly PDF:

```bash
python code/task6/stage2_task6_write_reports.py
python code/task6/stage2_task6_export_weekly_pdf.py \
  --source reports/weekly/week2_report_2026-05-28.md \
  --output reports/weekly/week2_report_2026-05-28.pdf
```

## Stage2 Task 3 v2

Task 3 v2 is the second-round WIDER FACE detection improvement track. It keeps
the SSD300 baseline artifacts under `reports/task3/` and writes all new
diagnostics, figures, and summaries under `reports/task3_v2/`.

- Task 3 v2 report: `reports/task3_v2/stage2_task3_v2_detection_improvement_plan.md`
- Task 3 v2 configs: `configs/mmdet/scrfd_like_r50_fpn_widerface_640_*.py`
- Task 3 v2 code: `code/task3/stage2_task3_v2_check_widerface.py` and `code/task3/stage2_task3_4_threshold_sweep.py`
- Task 3 v2 work dirs: `work_dirs/task3_v2/`

Data check:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task3/stage2_task3_v2_check_widerface.py \
    --data-root data/WIDERFace \
    --summary-out reports/task3_v2/summaries/widerface_v2_data_check.json
```

SSD300 threshold diagnosis:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task3/stage2_task3_4_threshold_sweep.py \
    --config configs/mmdet/ssd300_widerface_full_gpu.py \
    --checkpoint work_dirs/ssd300_widerface_full_gpu/epoch_24.pth \
    --data-root data/WIDERFace \
    --ann-file val.txt \
    --split val \
    --device cuda:0 \
    --score-thrs 0.05,0.1,0.2,0.3,0.5 \
    --max-per-img-values 50,100,200 \
    --summary-out reports/task3_v2/summaries/ssd300_threshold_sweep_summary.json \
    --plot-out reports/task3_v2/assets/diagnostics/ssd300_threshold_sweep.png
```

SCRFD-like 640 FPN smoke:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task3/stage2_task3_2_run_mmdet.py train \
    --config configs/mmdet/scrfd_like_r50_fpn_widerface_640_smoke_gpu.py \
    --work-dir work_dirs/task3_v2/scrfd_like_r50_fpn_widerface_640_smoke_gpu \
    --summary-out reports/task3_v2/summaries/scrfd_like_640_smoke_train_summary.json \
    --loss-plot-out reports/task3_v2/assets/training/scrfd_like_640_smoke_loss_curve.png \
    --device cuda:0
```
