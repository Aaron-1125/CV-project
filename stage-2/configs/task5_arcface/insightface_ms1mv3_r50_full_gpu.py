"""Official InsightFace full MS1MV3 ResNet50/ArcFace config for task 5.x.

This project config is consumed by ``stage2_task5_5_run_insightface.py``. The
wrapper writes an equivalent job config into the runtime-cloned official
InsightFace ``recognition/arcface_torch/configs`` package.
"""

task_name = "stage2_task5_insightface_ms1mv3_r50_full"
seed = 42

insightface = dict(
    repo_url="https://github.com/deepinsight/insightface.git",
    ref="master",
    external_dir="external/insightface",
    arcface_subdir="recognition/arcface_torch",
    generated_config_name="stage2_ms1mv3_r50_full",
)

data = dict(
    dataset="gaunernst/ms1mv3-recordio",
    recordio_root="data/task5_ms1mv3_full_recordio",
    rec="data/task5_ms1mv3_full_recordio/ms1m-retinaface-t1",
    lfw_dir="data/task5_lfw",
    expected_num_classes=93431,
    expected_num_images=5179510,
    expected_image_size=(112, 112),
)

official = dict(
    margin_list=(1.0, 0.5, 0.0),
    network="r50",
    resume=False,
    output="work_dirs/task5/insightface_ms1mv3_r50_full",
    embedding_size=512,
    sample_rate=1.0,
    fp16=True,
    momentum=0.9,
    weight_decay=5e-4,
    batch_size=128,
    lr=0.02,
    verbose=2000,
    dali=False,
    dali_aug=False,
    optimizer="sgd",
    num_workers=8,
    num_classes=93431,
    num_image=5179510,
    num_epoch=20,
    warmup_epoch=0,
    val_targets=["lfw"],
)

train = dict(
    device="cuda:0",
    target_lfw_accuracy=0.985,
    summary_out="reports/task5/summaries/insightface_full_train_summary.json",
    eval_summary_out="reports/task5/summaries/insightface_full_lfw_eval_summary.json",
)
