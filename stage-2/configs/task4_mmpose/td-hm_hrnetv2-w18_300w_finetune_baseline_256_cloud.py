# Stage2 Task4 v2 round 3: low-LR fine-tune from the original baseline.
#
# Use after the 384-input experiments. This keeps the original model/input
# geometry and starts from the original Task4 best checkpoint, then fine-tunes
# gently with slightly milder augmentation and a lower LR.

_base_ = "./td-hm_hrnetv2-w18_300w_full_gpu.py"

load_from = "work_dirs/task4/hrnetv2_w18_300w_full/best.pth"
work_dir = "work_dirs/task4_v2/hrnetv2_w18_300w_finetune_baseline_256_cloud"

train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", shift_prob=0.1, rotate_factor=45, scale_factor=(0.8, 1.25)),
    dict(type="TopdownAffine", input_size=(256, 256)),
    dict(type="GenerateTarget", encoder=dict(type="MSRAHeatmap", input_size=(256, 256), heatmap_size=(64, 64), sigma=1.5)),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(batch_size=64, num_workers=8, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=64, num_workers=4)
test_dataloader = val_dataloader

optim_wrapper = dict(optimizer=dict(type="Adam", lr=2.5e-5))
param_scheduler = [
    dict(type="LinearLR", begin=0, end=100, start_factor=0.1, by_epoch=False),
    dict(type="MultiStepLR", begin=0, end=40, milestones=[25, 35], gamma=0.1, by_epoch=True),
]
default_hooks = dict(checkpoint=dict(interval=1, save_best="NME", rule="less", max_keep_ckpts=5))
env_cfg = dict(cudnn_benchmark=True)
