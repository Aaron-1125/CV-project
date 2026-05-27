"""Dense MS1MV3 subset config for Stage2 task 5.x.

This fallback is used when the first time-budgeted subset has too few samples
per identity. It keeps fewer identities and more images per identity, which is
more suitable for ArcFace convergence and LFW verification.
"""

task_name = "stage2_task5_resnet50_arcface_ms1mv3_dense"
seed = 42

data = dict(
    ms1mv3_root="data/task5_ms1mv3_dense",
    train_index="data/task5_ms1mv3_dense/index/train_subset.csv",
    identity_map="data/task5_ms1mv3_dense/index/identity_map.json",
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
    epochs=60,
    batch_size=128,
    effective_batch_size=256,
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
