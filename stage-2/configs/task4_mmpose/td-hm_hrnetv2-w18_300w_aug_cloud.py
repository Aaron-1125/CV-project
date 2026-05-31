# Stage2 Task4 v2 ablation: keep HRNetv2-W18 but use the stronger augmentation
# and 384x384 input from the cloud plan. Run this if HRNetv2-W32 overfits or if
# a same-backbone comparison is needed for the report.

_base_ = "./td-hm_hrnetv2-w18_300w_full_gpu.py"

work_dir = "work_dirs/task4_v2/hrnetv2_w18_300w_aug_cloud"

train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)
codec = dict(type="MSRAHeatmap", input_size=(384, 384), heatmap_size=(96, 96), sigma=2.0)

model = dict(
    backbone=dict(init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://msra/hrnetv2_w18")),
    head=dict(decoder=codec),
)

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", shift_prob=0.3, rotate_factor=80, scale_factor=(0.65, 1.45)),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(batch_size=64, num_workers=8, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=64, num_workers=4, dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=2.5e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)
param_scheduler = [
    dict(type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type="MultiStepLR", begin=0, end=120, milestones=[80, 110], gamma=0.1, by_epoch=True),
]
default_hooks = dict(checkpoint=dict(interval=5, save_best="NME", rule="less", max_keep_ckpts=3))
env_cfg = dict(cudnn_benchmark=True)
