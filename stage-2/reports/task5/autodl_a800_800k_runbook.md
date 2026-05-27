# AutoDL A800 800k Runbook

This runbook is for finishing Stage2 Task 5.x on one AutoDL A800/A100 80GB GPU.
It keeps all artifacts under the Task5 paths:

- data: `data/task5_ms1mv3_dense/`, `data/task5_lfw/`
- work dir: `work_dirs/task5/resnet50_arcface_ms1mv3_dense/`
- reports: `reports/task5/`

## Instance

Recommended AutoDL instance:

- GPU: 1 x A800/A100 80GB
- data disk: at least 150 GB, preferably 200 GB
- image: PyTorch with CUDA and Python 3.10

## Setup

```bash
cd /root/autodl-tmp
git clone <your-repo-url> CV-project
cd /root/autodl-tmp/CV-project

pip install -U pip
pip install mmengine numpy pandas matplotlib scikit-learn pillow tqdm \
  opencv-python datasets huggingface_hub pyarrow torchmetrics webdataset

python - <<'PY'
import torch, torchvision, mmengine, webdataset, pyarrow, torchmetrics, cv2
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
PY
```

Do not reinstall `torch` or `torchvision` unless the AutoDL image is missing
them; use the PyTorch build that comes with the CUDA image.

## Prepare Data

```bash
cd /root/autodl-tmp/CV-project/stage-2

python code/task5/stage2_task5_3_prepare_lfw.py \
  --data-dir data/task5_lfw \
  --report-dir reports/task5

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

## Train

```bash
cd /root/autodl-tmp/CV-project/stage-2

python code/task5/stage2_task5_run_arcface.py train \
  --config configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py \
  --work-dir work_dirs/task5/resnet50_arcface_ms1mv3_dense \
  --summary-out reports/task5/summaries/ms1mv3_dense_train_summary.json \
  --loss-plot-out reports/task5/assets/training/ms1mv3_dense_loss_acc_curve.png \
  --device cuda:0
```

The cloud config uses batch size 512, effective batch size 512, 12 DataLoader
workers, AMP, CUDNN benchmark, 60 epochs, and a 36-hour max training budget.
It keeps `best.pth` and `last.pth`; per-epoch checkpoint saving is disabled to
avoid wasting cloud disk space.

## Evaluate LFW

```bash
cd /root/autodl-tmp/CV-project/stage-2

python code/task5/stage2_task5_run_arcface.py eval-lfw \
  --config configs/task5_arcface/resnet50_arcface_ms1mv3_dense_gpu.py \
  --checkpoint work_dirs/task5/resnet50_arcface_ms1mv3_dense/best.pth \
  --lfw-dir data/task5_lfw \
  --summary-out reports/task5/summaries/lfw_eval_summary.json \
  --roc-plot-out reports/task5/assets/evaluation/lfw_roc_curve.png \
  --device cuda:0
```

Success target: `reports/task5/summaries/lfw_eval_summary.json` has
`"accuracy" >= 0.985`.

If 800k still misses the target, expand to 1M+ images and resume the same work
dir rather than switching datasets.
