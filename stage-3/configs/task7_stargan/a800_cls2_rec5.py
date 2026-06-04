"""Stage3 Task7 A800 StarGAN CelebA config: lambda_cls=2, lambda_rec=5."""

task_name = "stage3_task7_stargan_celeba_128_cls2_rec5"
seed = 20260603

stargan = dict(
    repo_url="https://github.com/yunjey/StarGAN.git",
    ref="94dd002e93a2863d9b987a937b85925b80f7a19f",
    repo_path="/Users/aaron/Documents/字节实习/task/StarGAN",
)

data = dict(
    celeba_root="data/celeba_source",
    prepared_root="data/celeba",
    image_dir="data/celeba/images",
    attr_path="data/celeba/list_attr_celeba.txt",
    fixed_sample_count=16,
    eval_sample_count=512,
)

model = dict(
    dataset="CelebA",
    image_size=128,
    celeba_crop_size=178,
    c_dim=5,
    selected_attrs=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
    g_conv_dim=64,
    d_conv_dim=64,
    g_repeat_num=6,
    d_repeat_num=6,
    lambda_cls=2,
    lambda_rec=5,
    lambda_gp=10,
)

train = dict(
    batch_size=16,
    num_iters=200000,
    num_iters_decay=100000,
    g_lr=0.0001,
    d_lr=0.0001,
    n_critic=5,
    beta1=0.5,
    beta2=0.999,
    num_workers=8,
    use_tensorboard=False,
    log_step=100,
    sample_step=5000,
    model_save_step=10000,
    lr_update_step=1000,
    work_dir="work_dirs/task7/stargan_celeba_128_cls2_rec5",
    final_test_iters=200000,
)

monitor = dict(
    sample_every_iters=10000,
    include_pretrained=True,
)

evaluation = dict(
    device="cuda:0",
    batch_size=64,
    attribute_classifier_epochs=3,
    attribute_classifier_batch_size=256,
    attribute_classifier_lr=0.001,
    attribute_classifier_pretrained=True,
    attribute_threshold=0.5,
    identity_model="buffalo_l",
    identity_similarity_warning=0.35,
    fid_is_batch_size=64,
)

reports = dict(
    report_dir="reports/task7_cls2_rec5",
    summary_dir="reports/task7_cls2_rec5/summaries",
    asset_dir="reports/task7_cls2_rec5/assets",
)
