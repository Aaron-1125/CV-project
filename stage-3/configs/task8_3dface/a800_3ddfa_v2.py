"""Stage3 Task8 AutoDL A800 config for the official 3DDFA_V2 wrapper."""

task_name = "stage3_task8_official_3ddfa_v2_single_image_reconstruction"
seed = 20260604

third_party = dict(
    repo_url="https://github.com/cleardusk/3DDFA_V2.git",
    repo_path="/root/autodl-tmp/task/3DDFA_V2",
    official_config="configs/mb1_120x120.yml",
    preferred_checkpoint="weights/mb1_120x120.pth",
    preferred_onnx="weights/mb1_120x120.onnx",
    checkpoint_paths=[
        "weights/mb1_120x120.pth",
        "weights/mb1_120x120.onnx",
        "weights/mb05_120x120.pth",
        "weights/mb05_120x120.onnx",
    ],
    faceboxes_checkpoint_paths=[
        "FaceBoxes/weights/FaceBoxesProd.pth",
        "FaceBoxes/weights/FaceBoxesProd.onnx",
    ],
    required_paths=[
        "demo.py",
        "TDDFA.py",
        "TDDFA_ONNX.py",
        "configs/mb1_120x120.yml",
        "FaceBoxes",
        "utils/serialization.py",
        "utils/functions.py",
    ],
)

data = dict(
    celeba_root="/root/autodl-pub/CelebA",
    sample_count=8,
    link_mode="symlink",
    recursive=False,
)

reconstruction = dict(
    runner="official_subprocess",
    backend="auto",
    mode="gpu",
    official_outputs=["2d_sparse", "3d", "pose", "obj"],
    show_flag=False,
    allow_cpu_fallback=True,
    allow_backend_fallback=True,
)

render = dict(
    backend="auto",
    image_size=640,
    max_faces=8000,
    angles=[
        dict(name="frontal", yaw=0.0, pitch=0.0),
        dict(name="left_yaw_30", yaw=-30.0, pitch=0.0),
        dict(name="right_yaw_30", yaw=30.0, pitch=0.0),
        dict(name="left_yaw_60", yaw=-60.0, pitch=0.0),
        dict(name="right_yaw_60", yaw=60.0, pitch=0.0),
    ],
)

reports = dict(
    report_dir="reports/task8",
    asset_dir="reports/task8/assets",
    summary_dir="reports/task8/summaries",
)
