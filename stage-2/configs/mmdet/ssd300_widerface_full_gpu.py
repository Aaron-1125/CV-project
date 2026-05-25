_base_ = "./ssd300_widerface_full.py"

work_dir = "work_dirs/ssd300_widerface_full_gpu"

# GPU full training keeps the full train.txt/val.txt split and 24 epochs from
# the base config, but uses a smaller per-GPU batch for 8 GB NVIDIA cards.
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
)
test_dataloader = val_dataloader

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.003, momentum=0.9, weight_decay=5e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)
auto_scale_lr = dict(enable=False, base_batch_size=256)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3),
)
