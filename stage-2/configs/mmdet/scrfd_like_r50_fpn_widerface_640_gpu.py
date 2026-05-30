# SCRFD/RetinaFace-inspired WIDER FACE detector for Task3 v2.
#
# This stays inside the current MMDetection 3.x stack instead of pulling an
# older RetinaFace/SCRFD training framework into the project. The main changes
# over the SSD300 baseline are 640x640 input, P2-P6 FPN features, single-ratio
# dense anchors, focal loss, and much smaller training filters for tiny faces.

default_scope = "mmdet"
input_size = 640
data_root = "data/WIDERFace/"
work_dir = "work_dirs/task3_v2/scrfd_like_r50_fpn_widerface_640_gpu"
backend_args = None

model = dict(
    type="RetinaNet",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="checkpoints/resnet50-0676ba61.pth"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs="on_input",
        num_outs=5,
    ),
    bbox_head=dict(
        type="RetinaHead",
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            octave_base_scale=2,
            scales_per_octave=3,
            ratios=[1.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
        ),
        sampler=dict(type="PseudoSampler"),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.02,
        nms=dict(type="nms", iou_threshold=0.4),
        max_per_img=200,
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type="Resize", scale=(640, 640), keep_ratio=False),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(640, 640), keep_ratio=False),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

dataset_type = "WIDERFaceDataset"
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train.txt",
        data_prefix=dict(img="WIDER_train"),
        filter_cfg=dict(filter_empty_gt=True, bbox_min_size=4, min_size=8),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="val.txt",
        data_prefix=dict(img="WIDER_val"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="VOCMetric", metric="mAP", eval_mode="11points")
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type="MultiStepLR", by_epoch=True, milestones=[16, 22], gamma=0.1),
]

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.00125, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)
auto_scale_lr = dict(enable=False, base_batch_size=16)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="gloo"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
