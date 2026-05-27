"""Dense MS1MV3 subset config for Stage2 task 5.x on AutoDL A800.

This config is tuned for a single A800/A100 80GB cloud GPU and the 800k dense
MS1MV3 subset. It keeps enough identities and per-identity samples for a serious
LFW 98.5% target attempt while preserving the same report/work-dir layout.
"""

task_name = "stage2_task5_resnet50_arcface_ms1mv3_dense"
seed = 42

data = dict(
    ms1mv3_root="data/task5_ms1mv3_dense",
    train_index="data/task5_ms1mv3_dense/index/train_subset.csv",
    identity_map="data/task5_ms1mv3_dense/index/identity_map.json",
    lfw_root="data/task5_lfw",
    image_size=112,
    num_workers=12,
    persistent_workers=True,
    prefetch_factor=4,
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
    batch_size=512,
    effective_batch_size=512,
    optimizer="sgd",
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    amp=True,
    cudnn_benchmark=True,
    log_interval=25,
    lfw_eval_interval=1,
    target_lfw_accuracy=0.985,
    stop_on_target=True,
    save_every_epoch=False,
    max_hours=36.0,
    resume=True,
)

autodl = dict(
    recommended_gpu="A800/A100 80GB",
    recommended_images=800000,
    recommended_identities=20000,
    recommended_images_per_identity_cap=80,
    expected_runtime_hours="12-28",
)
