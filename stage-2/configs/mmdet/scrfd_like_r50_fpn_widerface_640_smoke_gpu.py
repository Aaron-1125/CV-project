_base_ = "./scrfd_like_r50_fpn_widerface_640_gpu.py"

work_dir = "work_dirs/task3_v2/scrfd_like_r50_fpn_widerface_640_smoke_gpu"
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1, val_interval=1)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=20),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1),
)
