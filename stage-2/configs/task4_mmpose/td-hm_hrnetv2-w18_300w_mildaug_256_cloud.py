# Stage2 Task4 v2 round 2: conservative HRNetv2-W18 tune.
#
# The W32/384 strong-augmentation experiment slightly regressed challenge NME.
# This config keeps the original 256x256 resolution and W18 capacity, but trains
# longer with milder bbox perturbations to improve difficult faces without
# destroying the common split.

_base_ = "./td-hm_hrnetv2-w18_300w_full_gpu.py"

work_dir = "work_dirs/task4_v2/hrnetv2_w18_300w_mildaug_256_cloud"

train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", shift_prob=0.15, rotate_factor=45, scale_factor=(0.7, 1.35)),
    dict(type="TopdownAffine", input_size=(256, 256)),
    dict(type="GenerateTarget", encoder=dict(type="MSRAHeatmap", input_size=(256, 256), heatmap_size=(64, 64), sigma=1.5)),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(batch_size=64, num_workers=8, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=64, num_workers=4)
test_dataloader = val_dataloader

optim_wrapper = dict(optimizer=dict(type="Adam", lr=1.25e-4))
param_scheduler = [
    dict(type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type="MultiStepLR", begin=0, end=120, milestones=[80, 110], gamma=0.1, by_epoch=True),
]
default_hooks = dict(checkpoint=dict(interval=5, save_best="NME", rule="less", max_keep_ckpts=3))
env_cfg = dict(cudnn_benchmark=True)
