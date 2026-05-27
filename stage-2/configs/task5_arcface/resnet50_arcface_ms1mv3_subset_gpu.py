"""Stage2 task 5.x ResNet50 + ArcFace training config.

The dataset index is produced by code/task5/stage2_task5_2_prepare_ms1mv3.py.
Training starts from random initialization; public pretrained recognition
weights are intentionally not used for the LFW target.
"""

task_name = "stage2_task5_resnet50_arcface_ms1mv3_subset"
seed = 42

data = dict(
    ms1mv3_root="data/task5_ms1mv3",
    train_index="data/task5_ms1mv3/index/train_subset.csv",
    identity_map="data/task5_ms1mv3/index/identity_map.json",
    lfw_root="data/task5_lfw",
    image_size=112,
    num_workers=4,
)

model = dict(
    backbone="iresnet50",
    embedding_size=512,
    dropout=0.0,
)

loss = dict(
    name="arcface",
    scale=64.0,
    margin=0.5,
)

train = dict(
    epochs=24,
    batch_size=96,
    effective_batch_size=192,
    optimizer="sgd",
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    amp=True,
    log_interval=50,
    lfw_eval_interval=1,
    target_lfw_accuracy=0.985,
    stop_on_target=True,
    save_every_epoch=True,
    max_hours=7.0,
    resume=True,
)
