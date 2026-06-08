"""Stage3 Task9 config for MediaPipe Face Mesh dynamic face effects."""

task_name = "stage3_task9_mediapipe_dynamic_face_effects"
seed = 20260605

data = dict(
    celeba_root="/root/autodl-pub/CelebA",
    sample_count=8,
    use_celeba_for_static_images=True,
    recursive=False,
)

input = dict(
    video_path="reports/task9/assets/input/user_video.mp4",
    alternate_video_path="reports/task9/assets/input/videos/user_video.mp4",
    image_dir="reports/task9/assets/input/static_images",
    allow_synthetic_video_from_images=False,
)

stickers = dict(
    glasses_path="reports/task9/assets/stickers/glasses.png",
    hat_path="reports/task9/assets/stickers/hat.png",
    cache_enabled=True,
    angle_quantization=5,
    size_quantization=8,
)

effects = dict(
    enable_glasses=True,
    enable_hat=True,
    enable_smooth=True,
    enable_whiten=True,
    enable_lipstick=True,
    lipstick_color=(190, 35, 80),  # RGB
    lipstick_alpha=0.45,
    smooth_strength=0.55,
    whiten_strength=0.35,
    face_roi_margin=0.15,
    lips_roi_margin=0.25,
    smooth_backend="bilateral_fast",
    smooth_downscale=0.5,
    smooth_diameter=5,
    smooth_sigma_color=30,
    smooth_sigma_space=30,
    glasses_scale_factor=2.2,
    glasses_y_offset_factor=0.03,
    hat_scale_factor=1.35,
    hat_y_offset_factor=0.55,
)

video = dict(
    fps=20,
    width=0,   # 0 keeps the input width.
    height=0,  # 0 keeps the input height.
    process_width=1280,
    process_height=720,
    keep_aspect_ratio=True,
    output_width=0,
    output_height=0,
    fast_mode=False,
    quality_process_width=1280,
    quality_process_height=720,
    max_keyframes=8,
)

reports = dict(
    report_dir="reports/task9",
    asset_dir="reports/task9/assets",
    summary_dir="reports/task9/summaries",
)
