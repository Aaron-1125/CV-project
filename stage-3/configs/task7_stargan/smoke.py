"""Stage3 Task7 tiny smoke config for local wrapper validation."""

task_name = "stage3_task7_stargan_celeba_128_smoke"
seed = 20260603

stargan = dict(
    repo_url="https://github.com/yunjey/StarGAN.git",
    ref="94dd002e93a2863d9b987a937b85925b80f7a19f",
    repo_path="/Users/aaron/Documents/字节实习/task/StarGAN",
)

data = dict(
    celeba_root="tmp/tiny_celeba",
    prepared_root="data/celeba_smoke",
    image_dir="data/celeba_smoke/images",
    attr_path="data/celeba_smoke/list_attr_celeba.txt",
    fixed_sample_count=8,
    eval_sample_count=8,
)

model = dict(
    dataset="CelebA",
    image_size=128,
    celeba_crop_size=178,
    c_dim=5,
    selected_attrs=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
    g_conv_dim=8,
    d_conv_dim=8,
    g_repeat_num=1,
    d_repeat_num=3,
    lambda_cls=1,
    lambda_rec=10,
    lambda_gp=10,
)

train = dict(
    batch_size=2,
    num_iters=2,
    num_iters_decay=1,
    g_lr=0.0001,
    d_lr=0.0001,
    n_critic=1,
    beta1=0.5,
    beta2=0.999,
    num_workers=0,
    use_tensorboard=False,
    log_step=1,
    sample_step=1,
    model_save_step=1,
    lr_update_step=1,
    work_dir="work_dirs/task7/stargan_celeba_128_smoke",
    final_test_iters=2,
)

monitor = dict(
    sample_every_iters=1,
    include_pretrained=False,
)

evaluation = dict(
    device="cpu",
    batch_size=2,
    attribute_classifier_epochs=1,
    attribute_classifier_batch_size=4,
    attribute_classifier_lr=0.001,
    attribute_classifier_pretrained=False,
    attribute_threshold=0.5,
    identity_model="buffalo_l",
    identity_similarity_warning=0.35,
    fid_is_batch_size=2,
)

reports = dict(
    report_dir="reports/task7_smoke",
    summary_dir="reports/task7_smoke/summaries",
    asset_dir="reports/task7_smoke/assets",
)
