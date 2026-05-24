# Self-contained MMDetection SSD300 config for WIDER FACE.
# Based on OpenMMLab mmdetection configs/wider_face/ssd300_8xb32-24e_widerface.py.

default_scope = "mmdet"
input_size = 300
data_root = "data/WIDERFace/"
work_dir = "work_dirs/ssd300_widerface_full"
backend_args = None

model = dict(
    type="SingleStageDetector",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[1, 1, 1],
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),
    backbone=dict(
        type="SSDVGG",
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://vgg16_caffe"),
    ),
    neck=dict(
        type="SSDNeck",
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256),
        level_strides=(2, 2, 1, 1),
        level_paddings=(1, 1, 0, 0),
        l2_norm_scale=20,
    ),
    bbox_head=dict(
        type="SSDHead",
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=1,
        anchor_generator=dict(
            type="SSDAnchorGenerator",
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False,
        ),
        sampler=dict(type="PseudoSampler"),
        smoothl1_beta=1.0,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type="nms", iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
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
    dict(
        type="Expand",
        mean=model["data_preprocessor"]["mean"],
        to_rgb=model["data_preprocessor"]["bgr_to_rgb"],
        ratio_range=(1, 4),
    ),
    dict(type="MinIoURandomCrop", min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    dict(type="Resize", scale=(300, 300), keep_ratio=False),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(300, 300), keep_ratio=False),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

dataset_type = "WIDERFaceDataset"
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train.txt",
        data_prefix=dict(img="WIDER_train"),
        filter_cfg=dict(filter_empty_gt=True, bbox_min_size=17, min_size=32),
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
    dict(type="MultiStepLR", by_epoch=True, milestones=[16, 20], gamma=0.1),
]

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.012, momentum=0.9, weight_decay=5e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)
auto_scale_lr = dict(enable=False, base_batch_size=256)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="gloo"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
