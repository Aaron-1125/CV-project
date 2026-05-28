# AutoDL A800 Official InsightFace Full MS1MV3 Runbook

This runbook replaces the earlier 800k JPEG subset path. It uses the official
InsightFace ArcFace Torch training pipeline and full MS1MV3 RecordIO data.

## Instance

- GPU: 1 x A800/A100 80GB
- data disk: at least 120 GB; 200 GB is safer for archives, logs, and outputs
- Python: 3.8 is recommended because `mxnet==1.9.1` is needed by InsightFace
  RecordIO and validation helpers

## Setup

```bash
cd /root/autodl-tmp
git clone <your-repo-url> CV-project
cd /root/autodl-tmp/CV-project

pip install -U pip
pip install numpy==1.23.5 easydict tensorboard mxnet==1.9.1 \
  mmengine pandas matplotlib scikit-learn scipy pillow tqdm opencv-python \
  datasets huggingface_hub hf_transfer hf_xet pyarrow torchmetrics webdataset

python - <<'PY'
import torch, mxnet, easydict
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("mxnet:", mxnet.__version__)
PY
```

Do not reinstall `torch` or `torchvision` unless the AutoDL image is missing
them. Use the PyTorch build that matches the CUDA image.

## Data

```bash
cd /root/autodl-tmp/CV-project/stage-2

python code/task5/stage2_task5_3_prepare_lfw.py \
  --data-dir data/task5_lfw \
  --report-dir reports/task5

python code/task5/stage2_task5_4_prepare_ms1mv3_recordio.py \
  --download \
  --dataset gaunernst/ms1mv3-recordio \
  --data-dir data/task5_ms1mv3_full_recordio \
  --lfw-dir data/task5_lfw \
  --report-dir reports/task5
```

If AutoDL download is slow, run the same RecordIO preparation locally, archive
`stage-2/data/task5_ms1mv3_full_recordio/`, upload it to AutoDL, and extract it
back to the same path.

If Hugging Face Xet connections are unstable, retry the same command; it is
resume-safe. As a fallback, set `HF_HUB_DISABLE_XET=1` before running the
prepare command and let the Hub client use the regular HTTP path.

## Train

```bash
cd /root/autodl-tmp/CV-project/stage-2

python code/task5/stage2_task5_5_run_insightface.py setup \
  --config configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py \
  --summary-out reports/task5/summaries/insightface_full_setup_summary.json

python code/task5/stage2_task5_5_run_insightface.py train \
  --config configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py \
  --summary-out reports/task5/summaries/insightface_full_train_summary.json
```

The generated official config is written into the runtime clone at
`external/insightface/recognition/arcface_torch/configs/stage2_ms1mv3_r50_full.py`.
Training outputs go to `work_dirs/task5/insightface_ms1mv3_r50_full/`.

## Evaluate

```bash
python code/task5/stage2_task5_5_run_insightface.py eval-summary \
  --config configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py \
  --checkpoint work_dirs/task5/insightface_ms1mv3_r50_full/model.pt \
  --summary-out reports/task5/summaries/insightface_full_lfw_eval_summary.json
```

Success target:

- `accuracy >= 0.985`
- `target_met: true`
- checkpoint exists at `work_dirs/task5/insightface_ms1mv3_r50_full/model.pt`

The previous `81.67%` result remains a failed baseline and is not the final
Task5 result.
