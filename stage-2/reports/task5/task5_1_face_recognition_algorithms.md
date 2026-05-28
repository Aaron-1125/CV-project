# Stage2 Task 5.1 Face Recognition Algorithm Notes

## Task Scope

Task 5.x focuses on closed-set training and open-set verification:

- Train a face embedding model with an InsightFace ResNet50/IResNet50 backbone.
- Optimize the embedding space with ArcFace additive angular margin loss.
- Validate the trained checkpoint with the official LFW 6000-pair, 10-fold protocol.

All task 5.x deliverables are isolated under `stage-2/code/task5/`,
`stage-2/configs/task5_arcface/`, `stage-2/reports/task5/`, and
`stage-2/work_dirs/task5/`.

## ResNet50 / IResNet50 Backbone

The project uses an InsightFace-style IResNet50 backbone for 112x112 aligned
face crops. Compared with a classification-only ImageNet ResNet, the face
recognition variant keeps residual convolution blocks but ends with a 512-D
feature embedding and feature batch normalization. The output embedding is
L2-normalized before ArcFace classification and before LFW cosine similarity.

## ArcFace Loss

ArcFace improves softmax classification by enforcing an angular margin between
identities. For normalized embedding `x` and normalized class weight `W`, the
target class logit is changed from `cos(theta_y)` to:

```text
s * cos(theta_y + m)
```

The non-target logits remain `s * cos(theta_j)`. In this task the default
parameters are the common ArcFace values `s=64` and `m=0.5`.

## Dataset Choice

The original Microsoft MS-Celeb-1M release is no longer a practical normal
download path. Task 5.x therefore uses the cleaned MS-Celeb-1M derivative
MS1MV3/MS1M-RetinaFace full RecordIO data from Hugging Face:

- source dataset: `gaunernst/ms1mv3-recordio`
- official-compatible layout: `ms1m-retinaface-t1/{train.rec,train.idx,property}`
- full size: `93431` identities and `5179510` images
- validation target: generated `lfw.bin`

The earlier `gaunernst/ms1mv3-wds-gz` JPEG subset route is retained only as a
failed baseline. The official route avoids the custom CSV/JPEG loader and uses
InsightFace's own RecordIO training pipeline.

## Official InsightFace Pipeline

The final route runtime-clones `deepinsight/insightface` and runs
`recognition/arcface_torch/train_v2.py` with a generated config based on the
official `ms1mv3_r50_onegpu.py` settings:

- network: `r50`
- batch size: `128`
- learning rate: `0.02`
- epochs: `20`
- fp16: enabled
- val targets: `["lfw"]`

## LFW Verification

LFW is evaluated as 1:1 face verification, not classification. The official
InsightFace validation callback loads `lfw.bin`, evaluates flipped embeddings,
and reports the 10-fold verification accuracy.

The project-local 6000-pair evaluator is kept only as a secondary diagnostic
tool for older checkpoints. The acceptance metric for the official route is the
LFW accuracy parsed from official InsightFace validation logs.

The acceptance target for this task is `accuracy >= 0.985` from the checkpoint
trained in this project run.
