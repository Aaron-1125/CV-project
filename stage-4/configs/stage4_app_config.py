"""Stage4 local vision application integration config."""

task_name = "stage4_local_vision_app_integration"

task9 = dict(
    config_path="configs/task9_effects/a800_mediapipe_face_effects.py",
)

paths = dict(
    default_video="stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4",
    fallback_videos=[
        "stage-3/reports/task9/assets/input/user_video.mp4",
        "stage-3/reports/task9/assets/input/videos/user_video.mp4",
        "stage-3/reports/task9/assets/inputs/user_video.mp4",
        "stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4",
    ],
    default_output_video="stage-4/reports/assets/videos/stage4_task9_effects_export.mp4",
)

effects = dict(
    enable_glasses=True,
    enable_hat=True,
    enable_smooth=True,
    enable_whiten=True,
    enable_lipstick=True,
    smooth_strength=0.55,
    whiten_strength=0.35,
    lipstick_alpha=0.45,
)

preview = dict(
    process_width=640,
    process_height=360,
    fast_mode=True,
    max_fps=24,
)

export = dict(
    process_width=None,
    process_height=None,
    fast_mode=False,
    camera_max_frames=300,
)

upgrades = dict(
    arcface=dict(enabled=False, description="Reserved identity recognition backend."),
    stargan=dict(enabled=False, description="Reserved attribute editing backend."),
    threeddfa=dict(enabled=False, description="Reserved 3D reconstruction backend."),
)
