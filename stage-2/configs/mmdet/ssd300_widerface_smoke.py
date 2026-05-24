_base_ = "./ssd300_widerface_full.py"

work_dir = "work_dirs/ssd300_widerface_smoke"

# Smoke training should finish on CPU/MPS-class local machines. It deliberately
# starts from random weights to avoid downloading the VGG Caffe checkpoint.
model = dict(backbone=dict(init_cfg=None))

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(ann_file="smoke_train.txt"),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(ann_file="smoke_val.txt"),
)
test_dataloader = val_dataloader

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1, val_interval=1)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=False, begin=0, end=10),
    dict(type="MultiStepLR", by_epoch=True, milestones=[1], gamma=0.1),
]
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=5e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=5),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=2),
)
