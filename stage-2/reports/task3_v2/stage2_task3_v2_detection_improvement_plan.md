# Stage2 Task3 v2 Detection Improvement

## Motivation

The first full SSD300 WIDER FACE run is kept as the Task3 baseline. Its custom
IoU=0.5 evaluation reported AP50 0.3689, precision 0.042, and recall 0.441.
The low precision is consistent with a detector evaluated at a low score
threshold while keeping up to 200 detections per image. SSD300 also resizes the
scene to 300x300 and filters small training boxes, which is a poor fit for the
tiny-face distribution in WIDER FACE.

Task3 v2 therefore keeps the baseline untouched and adds two independent tracks:

- Threshold diagnostics for the existing SSD300 checkpoint.
- A 640x640 FPN dense detector inspired by RetinaFace/SCRFD design choices.

## Output Isolation

- Baseline Task3 remains under `reports/task3/` and `work_dirs/ssd300_widerface_full_gpu/`.
- Task3 v2 writes reports and images under `reports/task3_v2/`.
- Task3 v2 training writes checkpoints under `work_dirs/task3_v2/`.

## Data Check

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task3/stage2_task3_v2_check_widerface.py \
    --data-root data/WIDERFace \
    --summary-out reports/task3_v2/summaries/widerface_v2_data_check.json
```

Expected counts are 12,337 train images and 3,079 validation images.

## Baseline Threshold Diagnosis

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

This sweep reuses one inference pass and varies `score_thr` and `max_per_img`.
The model config's NMS IoU is recorded in each row.

## SCRFD-like / RetinaFace-inspired Detector

The new config is `configs/mmdet/scrfd_like_r50_fpn_widerface_640_gpu.py`.
It is not the official SCRFD implementation. It is an MMDetection-native dense
detector that adopts the design direction needed for WIDER FACE:

- 640x640 input rather than 300x300.
- ResNet50 backbone with P2-P6 FPN outputs.
- Single-ratio dense anchors at strides 4, 8, 16, 32, and 64.
- Focal loss for foreground/background imbalance.
- Small-face-friendly dataset filtering.

Smoke run:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task3/stage2_task3_2_run_mmdet.py train \
    --config configs/mmdet/scrfd_like_r50_fpn_widerface_640_smoke_gpu.py \
    --work-dir work_dirs/task3_v2/scrfd_like_r50_fpn_widerface_640_smoke_gpu \
    --summary-out reports/task3_v2/summaries/scrfd_like_640_smoke_train_summary.json \
    --loss-plot-out reports/task3_v2/assets/training/scrfd_like_640_smoke_loss_curve.png \
    --device cuda:0
```

Full 24-epoch run:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task3/stage2_task3_2_run_mmdet.py train \
    --config configs/mmdet/scrfd_like_r50_fpn_widerface_640_gpu.py \
    --work-dir work_dirs/task3_v2/scrfd_like_r50_fpn_widerface_640_gpu \
    --summary-out reports/task3_v2/summaries/scrfd_like_640_train_summary.json \
    --loss-plot-out reports/task3_v2/assets/training/scrfd_like_640_loss_curve.png \
    --device cuda:0
```

If the 8GB 4060 runs out of memory, rerun with
`configs/mmdet/scrfd_like_r50_fpn_widerface_640_bs1_gpu.py`.

Local 4060 smoke status:

- Batch size 2 started correctly and trained to `1940/6169` iterations.
- Logged memory peaked at `8138 MiB`, essentially the full 8GB card.
- The `docker compose run --rm` container exited before writing `epoch_1.pth`,
  so the batch size 2 smoke is treated as failed/unstable on the local 4060.
- Because the model only appears viable with batch size 1 on this machine, the
  24-epoch run should be moved to cloud GPU according to the decision rule.

Batch-size-1 fallback smoke:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task3/stage2_task3_2_run_mmdet.py train \
    --config configs/mmdet/scrfd_like_r50_fpn_widerface_640_smoke_bs1_gpu.py \
    --work-dir work_dirs/task3_v2/scrfd_like_r50_fpn_widerface_640_smoke_bs1_gpu \
    --summary-out reports/task3_v2/summaries/scrfd_like_640_smoke_bs1_train_summary.json \
    --loss-plot-out reports/task3_v2/assets/training/scrfd_like_640_smoke_bs1_loss_curve.png \
    --device cuda:0
```

Evaluation:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task3/stage2_task3_3_evaluate_widerface.py \
    --config configs/mmdet/scrfd_like_r50_fpn_widerface_640_gpu.py \
    --checkpoint work_dirs/task3_v2/scrfd_like_r50_fpn_widerface_640_gpu/epoch_24.pth \
    --data-root data/WIDERFace \
    --ann-file val.txt \
    --split val \
    --input-dir reports/task3_v2/assets/inputs \
    --out-dir reports/task3_v2/assets/detection \
    --summary-out reports/task3_v2/summaries/scrfd_like_640_eval_summary.json \
    --device cuda:0 \
    --score-thr 0.1 \
    --iou-thr 0.5 \
    --visualize-count 8 \
    --vis-top-k 30 \
    --metrics-plot-out reports/task3_v2/assets/evaluation/scrfd_like_640_eval_metrics.png
```

## Local 4060 Decision Rule

Use the 1-epoch smoke run to estimate runtime. If 24 epochs exceed 18 hours or
the model only fits with batch size 1 and unstable throughput, move the same
config to the A800/A100 cloud machine.
