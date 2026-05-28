# Stage2 Task 5.x ArcFace Training Report

## Deliverable Layout

Task 5.x remains isolated from task 3.x detection and task 4.x landmark
deliverables:

- code: `stage-2/code/task5/`
- configs: `stage-2/configs/task5_arcface/`
- data: `stage-2/data/task5_ms1mv3_full_recordio/` and `stage-2/data/task5_lfw/`
- reports: `stage-2/reports/task5/`
- work dirs: `stage-2/work_dirs/task5/`
- runtime external source: `stage-2/external/insightface/`

Datasets, runtime-cloned external source, and model checkpoints are local
artifacts only. They are ignored by Git through `data/`, `work_dirs/`,
`stage-2/external/`, `*.rec`, `*.idx`, `*.pth`, and `*.pt` rules.

## Baseline Result From the Previous Wrapper

The previous project-local ResNet50/IResNet50 + ArcFace wrapper completed a real
AutoDL A800 run on the dense MS1MV3 JPEG subset:

- images: `800000`
- identities: `20000`
- epochs completed: `60`
- batch size: `512`
- best LFW accuracy: `0.8167`
- ROC AUC: `0.8791`
- target met: `false`

The training loss dropped to a very low value while LFW stayed far below the
98.5% target. That pattern indicates poor open-set generalization from this
custom subset/pipeline, not merely an unfinished epoch count. The checkpoint is
kept as a failed baseline and is not treated as the final task result.

## Official InsightFace Route

The new main route uses the official InsightFace ArcFace Torch implementation:

- upstream repo: `deepinsight/insightface`
- runtime source path: `stage-2/external/insightface/`
- official subproject: `recognition/arcface_torch`
- data source: `gaunernst/ms1mv3-recordio`
- dataset layout: `data/task5_ms1mv3_full_recordio/ms1m-retinaface-t1/`
- model: ResNet50 / `r50`
- loss: ArcFace margin `(1.0, 0.5, 0.0)`
- full MS1MV3 size: `93431` identities, `5179510` images
- default single A800 config: batch size `128`, lr `0.02`, `20` epochs, fp16
- validation target: `lfw`

The official source is cloned at runtime and the resolved commit SHA is written
to the training summary. The project does not vendor the InsightFace source.

## Commands

Prepare LFW if needed:

```bash
cd /root/autodl-tmp/CV-project/stage-2

python code/task5/stage2_task5_3_prepare_lfw.py \
  --data-dir data/task5_lfw \
  --report-dir reports/task5
```

Prepare full MS1MV3 RecordIO and create `lfw.bin`:

```bash
python code/task5/stage2_task5_4_prepare_ms1mv3_recordio.py \
  --download \
  --dataset gaunernst/ms1mv3-recordio \
  --data-dir data/task5_ms1mv3_full_recordio \
  --lfw-dir data/task5_lfw \
  --report-dir reports/task5
```

Run official InsightFace setup validation:

```bash
python code/task5/stage2_task5_5_run_insightface.py setup \
  --config configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py \
  --summary-out reports/task5/summaries/insightface_full_setup_summary.json
```

Train with the official pipeline:

```bash
python code/task5/stage2_task5_5_run_insightface.py train \
  --config configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py \
  --summary-out reports/task5/summaries/insightface_full_train_summary.json
```

Parse the final official LFW validation result:

```bash
python code/task5/stage2_task5_5_run_insightface.py eval-summary \
  --config configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py \
  --checkpoint work_dirs/task5/insightface_ms1mv3_r50_full/model.pt \
  --summary-out reports/task5/summaries/insightface_full_lfw_eval_summary.json
```

## Acceptance

The Task5 target is met only when
`reports/task5/summaries/insightface_full_lfw_eval_summary.json` reports:

```json
{
  "accuracy": 0.985,
  "target_met": true
}
```

If the official full MS1MV3 run still falls below 98.5%, the report must keep
`target_met: false` and record the exact metric instead of substituting a public
pretrained checkpoint.
