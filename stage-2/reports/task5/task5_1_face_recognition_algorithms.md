# Stage2 Task 5.1 Face Recognition Algorithm Notes

## Task Scope

Task 5.x focuses on closed-set training and open-set verification:

- Train a face embedding model with a ResNet50/IResNet50 backbone.
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
MS1MV3/MS1M-RetinaFace subset from Hugging Face:

- source dataset: `gaunernst/ms1mv3-wds-gz`
- image format: aligned 112x112 face JPEG
- label field: `cls`

The preparation script exports a time-budgeted subset by default and can be
rerun with a larger `--max-images` or `--mode full` when the LFW target has not
yet been reached.

## LFW Verification

LFW is evaluated as 1:1 face verification, not classification. The wrapper
computes embeddings for every image referenced in `pairs.txt`, scores each pair
with cosine similarity, and follows the 10-fold protocol:

- train threshold on nine folds
- test on the held-out fold
- report mean accuracy and standard deviation over ten folds

The acceptance target for this task is `accuracy >= 0.985` from the checkpoint
trained in this project run.
