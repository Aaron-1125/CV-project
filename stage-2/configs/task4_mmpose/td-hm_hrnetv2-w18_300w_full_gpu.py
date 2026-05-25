# Self-contained MMPose HRNetv2-W18 config for Stage2 task 4.x.

default_scope = "mmpose"

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="NME",
        rule="less",
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="PoseVisualizationHook", enable=False),
)
custom_hooks = [dict(type="SyncBuffersHook")]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="PoseLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True, num_digits=6)
log_level = "INFO"
load_from = None
resume = False
backend_args = dict(backend="local")

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(optimizer=dict(type="Adam", lr=1.25e-4))
param_scheduler = [
    dict(type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type="MultiStepLR", begin=0, end=60, milestones=[40, 55], gamma=0.1, by_epoch=True),
]
auto_scale_lr = dict(enable=False, base_batch_size=512)

codec = dict(type="MSRAHeatmap", input_size=(256, 256), heatmap_size=(64, 64), sigma=1.5)

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="HRNet",
        in_channels=3,
        extra=dict(
            stage1=dict(num_modules=1, num_branches=1, block="BOTTLENECK", num_blocks=(4,), num_channels=(64,)),
            stage2=dict(num_modules=1, num_branches=2, block="BASIC", num_blocks=(4, 4), num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                multiscale_output=True,
            ),
            upsample=dict(mode="bilinear", align_corners=False),
        ),
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://msra/hrnetv2_w18"),
    ),
    neck=dict(type="FeatureMapProcessor", concat=True),
    head=dict(
        type="HeatmapHead",
        in_channels=270,
        out_channels=68,
        deconv_out_channels=None,
        conv_out_channels=(270,),
        conv_kernel_sizes=(1,),
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True, flip_mode="heatmap", shift_heatmap=True),
)

dataset_type = "Face300WDataset"
data_mode = "topdown"
data_root = "data/task4_300w/mmpose/300w/"

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", shift_prob=0, rotate_factor=60, scale_factor=(0.75, 1.25)),
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

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/face_landmarks_300w_train.json",
        data_prefix=dict(img="images/"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/face_landmarks_300w_valid.json",
        data_prefix=dict(img="images/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="NME", norm_mode="keypoint_distance")
test_evaluator = val_evaluator

work_dir = "work_dirs/task4/hrnetv2_w18_300w_full"
