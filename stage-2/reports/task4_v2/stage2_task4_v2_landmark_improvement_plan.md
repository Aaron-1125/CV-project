# Stage2 Task4 v2 Landmark Improvement

## Baseline

The current Task4 HRNetv2-W18 300W run is valid but weaker on difficult images:

| Split | NME |
| --- | ---: |
| common | 0.0291 |
| challenge | 0.0552 |
| full/valid | 0.0342 |

The challenge split is almost 1.9x the common NME, which matches the mentor's
feedback about occlusion, large pose, and hard faces.

## V2 Strategy

Task4 v2 is isolated from the original Task4 delivery:

- Reports: `reports/task4_v2/`
- Work dirs: `work_dirs/task4_v2/`
- Main config: `configs/task4_mmpose/td-hm_hrnetv2-w32_300w_aug_cloud.py`
- Ablation config: `configs/task4_mmpose/td-hm_hrnetv2-w18_300w_aug_cloud.py`

The main cloud run changes four things:

- HRNetv2-W18 -> HRNetv2-W32 for more capacity.
- 256x256 input -> 384x384 input with 96x96 heatmaps.
- 60 epochs -> 120 epochs.
- Stronger bbox shift, scale, and rotation augmentation.

## A800 Commands

Train the main model:

```bash
cd /root/autodl-tmp/CV-project/stage-2
python code/task4/stage2_task4_run_mmpose.py train \
  --config configs/task4_mmpose/td-hm_hrnetv2-w32_300w_aug_cloud.py \
  --work-dir work_dirs/task4_v2/hrnetv2_w32_300w_aug_cloud \
  --summary-out reports/task4_v2/summaries/300w_w32_aug_train_summary.json \
  --loss-plot-out reports/task4_v2/assets/training/300w_w32_aug_loss_curve.png \
  --device cuda:0
```

Evaluate common/challenge/full:

```bash
python code/task4/stage2_task4_run_mmpose.py test \
  --config configs/task4_mmpose/td-hm_hrnetv2-w32_300w_aug_cloud.py \
  --checkpoint work_dirs/task4_v2/hrnetv2_w32_300w_aug_cloud/best.pth \
  --work-dir work_dirs/task4_v2/hrnetv2_w32_300w_aug_cloud_eval \
  --summary-out reports/task4_v2/summaries/300w_w32_aug_eval_summary.json \
  --metrics-plot-out reports/task4_v2/assets/evaluation/300w_w32_aug_nme_metrics.png \
  --device cuda:0
```

If W32 overfits or does not improve challenge NME, run the W18 augmented
ablation with the same command shape and compare it against the baseline.

## Acceptance

The v2 target is not just lower full NME. It should specifically reduce
`challenge` NME below `0.0552` while keeping `common` near or below `0.0291`.
If challenge improves but common regresses slightly, the report should frame
that tradeoff explicitly.

## Round 1 Result

The first cloud run, `td-hm_hrnetv2-w32_300w_aug_cloud.py`, did not improve the
mentor-facing metric:

| Model | common NME | challenge NME | full NME |
| --- | ---: | ---: | ---: |
| Baseline W18/256 | 0.0291 | 0.0552 | 0.0342 |
| W32/384 strong aug | 0.0294 | 0.0561 | 0.0346 |

This is treated as a failed ablation. The likely cause is that the first v2
config changed too many variables at once: larger backbone, larger input,
stronger augmentation, AdamW, and a higher learning rate. The next round should
reduce augmentation strength and isolate the source of improvement.

## Round 2 Order

Run these in order on A800:

1. Evaluate available W32 checkpoints, especially `epoch_120.pth`, because the
   saved `best.pth` is selected by full valid NME, not challenge NME.
2. Train `td-hm_hrnetv2-w18_300w_mildaug_384_cloud.py`.
3. If needed, train `td-hm_hrnetv2-w18_300w_mildaug_256_cloud.py` as a stable
   same-resolution ablation.
4. Only if W18 improves, try `td-hm_hrnetv2-w32_300w_mildaug_384_cloud.py`.

The first successful checkpoint is the one that reduces challenge NME below
`0.0552` without pushing common above the baseline by more than a small margin.

## Round 2 Result

The `td-hm_hrnetv2-w18_300w_mildaug_384_cloud.py` run also regressed:

| Model | common NME | challenge NME | full NME |
| --- | ---: | ---: | ---: |
| Baseline W18/256 | 0.0291 | 0.0552 | 0.0342 |
| W18/384 mild aug | 0.0295 | 0.0563 | 0.0348 |

This means the 384-input route is not currently helping the prepared Kaggle
300W split. Continue with checkpoint sweep first, then a low-LR baseline
fine-tune at the original 256x256 geometry.

## Checkpoint Sweep

Use this before launching another long training job:

```bash
python code/task4/stage2_task4_4_sweep_checkpoints.py \
  --config configs/task4_mmpose/td-hm_hrnetv2-w18_300w_mildaug_384_cloud.py \
  --checkpoint-dir work_dirs/task4_v2/hrnetv2_w18_300w_mildaug_384_cloud \
  --summary-out reports/task4_v2/summaries/300w_w18_mildaug_384_checkpoint_sweep.json \
  --plot-out reports/task4_v2/assets/evaluation/300w_w18_mildaug_384_checkpoint_sweep.png
```

Also sweep the W32 strong-augmentation run:

```bash
python code/task4/stage2_task4_4_sweep_checkpoints.py \
  --config configs/task4_mmpose/td-hm_hrnetv2-w32_300w_aug_cloud.py \
  --checkpoint-dir work_dirs/task4_v2/hrnetv2_w32_300w_aug_cloud \
  --summary-out reports/task4_v2/summaries/300w_w32_aug_checkpoint_sweep.json \
  --plot-out reports/task4_v2/assets/evaluation/300w_w32_aug_checkpoint_sweep.png
```

## Low-LR Baseline Fine-Tune

If the sweep finds no checkpoint below the baseline challenge NME, run:

```bash
python code/task4/stage2_task4_run_mmpose.py train \
  --config configs/task4_mmpose/td-hm_hrnetv2-w18_300w_finetune_baseline_256_cloud.py \
  --work-dir work_dirs/task4_v2/hrnetv2_w18_300w_finetune_baseline_256_cloud \
  --summary-out reports/task4_v2/summaries/300w_w18_finetune_256_train_summary.json \
  --loss-plot-out reports/task4_v2/assets/training/300w_w18_finetune_256_loss_curve.png \
  --device cuda:0
```

Then evaluate:

```bash
python code/task4/stage2_task4_run_mmpose.py test \
  --config configs/task4_mmpose/td-hm_hrnetv2-w18_300w_finetune_baseline_256_cloud.py \
  --checkpoint work_dirs/task4_v2/hrnetv2_w18_300w_finetune_baseline_256_cloud/best.pth \
  --work-dir work_dirs/task4_v2/hrnetv2_w18_finetune_256_eval \
  --summary-out reports/task4_v2/summaries/300w_w18_finetune_256_eval_summary.json \
  --metrics-plot-out reports/task4_v2/assets/evaluation/300w_w18_finetune_256_nme_metrics.png \
  --device cuda:0
```

## Final Decision

The Task4 v2 tuning attempts did not improve the baseline challenge split. The
project should keep the original Task4 HRNetv2-W18 checkpoint as the final
landmark model and report the v2 runs as controlled negative experiments.

Final model to keep:

- Config: `configs/task4_mmpose/td-hm_hrnetv2-w18_300w_full_gpu.py`
- Checkpoint: `work_dirs/task4/hrnetv2_w18_300w_full/best.pth`
- Reported NME: common `0.0291`, challenge `0.0552`, full `0.0342`

Failed improvement attempts:

- HRNetv2-W32, 384 input, strong augmentation: regressed to challenge `0.0561`.
- HRNetv2-W18, 384 input, mild augmentation: regressed to challenge `0.0563`.
- HRNetv2-W18, 256 input, low-LR baseline fine-tune: did not beat the baseline.

Interpretation for the coursework report:

- Stronger augmentation and higher input resolution did not automatically
  improve the 300W challenge subset.
- The challenge split is small, only 135 images, so NME can be sensitive to
  bbox normalization and a few difficult samples.
- With the current Kaggle-prepared 300W data, the original W18/256 model is the
  best validated checkpoint and should remain the submitted Task4 result.
- A more meaningful next improvement would require additional hard-pose/occlusion
  data, more robust face crops from a better detector, or an externally
  pre-trained face-landmark checkpoint, not just longer training.
