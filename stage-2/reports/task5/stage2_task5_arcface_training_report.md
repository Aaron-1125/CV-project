# Stage2 Task 5.x ArcFace Training Report

## Deliverable Layout

Task 5.x is separated from the task 3.x WIDER FACE detector and task 4.x
landmark/alignment outputs:

- code: `stage-2/code/task5/`
- config: `stage-2/configs/task5_arcface/`
- data: `stage-2/data/task5_ms1mv3_dense/` and `stage-2/data/task5_lfw/`
- reports: `stage-2/reports/task5/`
- work dirs: `stage-2/work_dirs/task5/`

The dataset and model checkpoints are local artifacts only. They are ignored by
Git through `stage-2/.gitignore` rules for `data/`, `work_dirs/`, `*.pth`, and
`*.pt`.

## Environment

The GPU Docker path is used for task 5.x:

```bash
docker compose build stage2-gpu
docker compose run --rm -w /workspace stage2-gpu python docker/verify_environment.py
```

The verified environment includes CUDA-enabled PyTorch, OpenCV, WebDataset,
PyArrow, TorchMetrics, InsightFace, MMDetection, and MMPose. The verification
run reported `cuda_available True`.

## Data

The original MS-Celeb-1M release is not a normal public download path now, so
this task uses the cleaned MS-Celeb-1M derivative MS1MV3/MS1M-RetinaFace subset
from Hugging Face dataset `gaunernst/ms1mv3-wds-gz`.

The final dense local subset was expanded after the first training run:

- images: `400000`
- identities: `10000`
- mean images per identity: `40.0`
- max images per identity: `80`
- image errors: `0`
- summary: `reports/task5/summaries/ms1mv3_dense_summary.json`

LFW preparation completed with the official 6000-pair protocol:

- positive pairs: `3000`
- negative pairs: `3000`
- summary: `reports/task5/summaries/lfw_dataset_summary.json`

## Commands

Prepare dense MS1MV3:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task5/stage2_task5_2_prepare_ms1mv3.py \
    --dataset gaunernst/ms1mv3-wds-gz \
    --data-dir data/task5_ms1mv3_dense \
    --report-dir reports/task5 \
    --target-hours 7 \
    --mode subset \
    --max-images 400000 \
    --max-identities 10000 \
    --images-per-identity-cap 80 \
    --min-subset-images 400000 \
    --output-tag ms1mv3_dense
```

Prepare LFW:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task5/stage2_task5_3_prepare_lfw.py \
    --data-dir data/task5_lfw \
    --report-dir reports/task5
```

Train ResNet50/IResNet50 + ArcFace:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task5/stage2_task5_run_arcface.py train \
    --config configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py \
    --work-dir work_dirs/task5/resnet50_arcface_ms1mv3_dense \
    --summary-out reports/task5/summaries/ms1mv3_dense_train_summary.json \
    --loss-plot-out reports/task5/assets/training/ms1mv3_dense_loss_acc_curve.png \
    --device cuda:0
```

Evaluate LFW:

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task5/stage2_task5_run_arcface.py eval-lfw \
    --config configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py \
    --checkpoint work_dirs/task5/resnet50_arcface_ms1mv3_dense/best.pth \
    --lfw-dir data/task5_lfw \
    --summary-out reports/task5/summaries/lfw_eval_summary.json \
    --roc-plot-out reports/task5/assets/evaluation/lfw_roc_curve.png \
    --device cuda:0
```

## Training Result

The completed training run used the 200k dense subset before the later 400k
expansion. It ran for the configured 7-hour local budget and stopped cleanly:

- completed epochs: `13`
- actual batch size: `128`
- gradient accumulation: `2`
- effective batch size: `256`
- final train loss: `9.0617`
- best checkpoint: `work_dirs/task5/resnet50_arcface_ms1mv3_dense/best.pth`
- train summary: `reports/task5/summaries/ms1mv3_dense_train_summary.json`
- loss/LFW curve: `reports/task5/assets/training/ms1mv3_dense_loss_acc_curve.png`

The best LFW accuracy from this checkpoint was:

- accuracy: `0.7790`
- target: `0.9850`
- target met: `false`
- ROC AUC: `0.8478`
- ROC plot: `reports/task5/assets/evaluation/lfw_roc_curve.png`
- similarity histogram: `reports/task5/assets/evaluation/lfw_similarity_histogram.png`

The LFW summary is `reports/task5/summaries/lfw_eval_summary.json`.

## 400k Resume Attempt

Because the first 200k run did not reach 98.5%, the dense subset was expanded to
400k images with the same 10k identities. A resume training attempt was started
from `last.pth` at epoch 14. On the local RTX 4060 8GB environment, the 400k
epoch progressed too slowly to complete a new epoch in a practical time window.
The run was stopped intentionally after confirming the bottleneck, preserving
the completed 200k checkpoint and the expanded 400k dataset.

This means the final delivered model is a real locally trained
ResNet50/IResNet50 + ArcFace checkpoint, but it does not satisfy the 98.5% LFW
target. The most likely reason is insufficient local training budget/throughput
for from-scratch ArcFace convergence on this hardware.

The 400k attempt record is:

- `reports/task5/summaries/ms1mv3_dense_400k_resume_attempt_summary.json`

## Conclusion

Task 5.x code, configs, data preparation, training wrapper, LFW evaluation,
plots, and reports are delivered under the task 5.x directories. The acceptance
target was not reached by the local training run, and the reports keep that
status explicit instead of substituting a public pretrained model.
