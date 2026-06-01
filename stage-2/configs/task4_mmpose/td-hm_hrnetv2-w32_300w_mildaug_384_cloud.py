# Stage2 Task4 v2 round 2: HRNetv2-W32 with milder augmentation.
#
# Use this after W18 mild-aug. It keeps the higher-capacity W32/384 route but
# removes the over-aggressive rotation/scale/shift used in the first W32 run.

_base_ = "./td-hm_hrnetv2-w32_300w_aug_cloud.py"

work_dir = "work_dirs/task4_v2/hrnetv2_w32_300w_mildaug_384_cloud"

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1.25e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", shift_prob=0.15, rotate_factor=45, scale_factor=(0.7, 1.35)),
    dict(type="TopdownAffine", input_size=(384, 384)),
    dict(type="GenerateTarget", encoder=dict(type="MSRAHeatmap", input_size=(384, 384), heatmap_size=(96, 96), sigma=2.0)),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(batch_size=64, num_workers=8, dataset=dict(pipeline=train_pipeline))
