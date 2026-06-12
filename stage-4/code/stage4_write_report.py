#!/usr/bin/env python3
"""Write Stage4 delivery summary, report, and checklist."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage4_backend import build_static_summary, collected_keyframes, delivery_notes, runtime_environment, video_metadata
from stage4_common import (
    ensure_stage4_dirs,
    load_python_config,
    read_json,
    rel_to_repo,
    stage4_report_path,
    stage4_summary_path,
    write_json,
    write_text,
)


CHECKLIST_PATH = stage4_summary_path().with_name("stage4_delivery_checklist.json")
UI_V2_SUMMARY_PATH = stage4_summary_path().with_name("stage4_ui_v2_summary.json")
UI_V3_SUMMARY_PATH = stage4_summary_path().with_name("stage4_ui_v3_summary.json")
README_PATH = stage4_summary_path().parents[2] / "README_STAGE4.md"


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not_measured"
    if isinstance(value, float):
        return ("{:.%df}" % digits).format(value)
    return str(value)


def load_existing_summary() -> Dict[str, Any]:
    path = stage4_summary_path()
    if not path.exists():
        return {}
    try:
        value = read_json(path)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def output_video_path(summary: Dict[str, Any]) -> Path | None:
    value = summary.get("output_video") or summary.get("output_files", {}).get("video")
    if not value:
        return None
    return Path(str(value))


def normalize_summary(cfg: Dict[str, Any]) -> Dict[str, Any]:
    existing = load_existing_summary()
    if existing.get("stage") == "stage4" and existing.get("raw_task9_result"):
        summary = dict(existing)
    elif existing.get("result"):
        raw = existing["result"]
        options = existing.get("options", {})
        processed = raw.get("processed_frames")
        total_seconds = raw.get("total_seconds")
        avg_ms = float(total_seconds) / float(processed) * 1000.0 if processed and total_seconds is not None else None
        environment = runtime_environment()
        summary = {
            "stage": "stage4",
            "task": "project_integration",
            "generated_at": existing.get("generated_at"),
            "backend": "stage3_task9_mediapipe_opencv",
            "input_video": existing.get("input_source") or raw.get("input_video"),
            "output_video": existing.get("output_video") or raw.get("output_video"),
            "enabled_effects": options.get("effects", []),
            "debug_options": {
                "fps_stat_recorded": raw.get("average_processing_fps") is not None,
                "landmarks_keyframe_debug": bool(raw.get("keyframes")),
                "debug_sticker_geometry": bool(options.get("debug_sticker_geometry")),
            },
            "mode": "fast_preview" if options.get("fast_mode") else "quality_export",
            "fast_mode": bool(options.get("fast_mode")),
            "process_width": options.get("process_width"),
            "max_frames": options.get("max_frames"),
            "processed_frame_count": processed,
            "fps": raw.get("average_processing_fps"),
            "processing_fps": raw.get("average_processing_fps"),
            "avg_ms_per_frame": avg_ms,
            "output_resolution": {"width": raw.get("process_width"), "height": raw.get("process_height")},
            "environment": environment,
            "output_files": {
                "video": existing.get("output_video") or raw.get("output_video"),
                "summary": str(stage4_summary_path()),
                "report": str(stage4_report_path()),
                "keyframes": list(collected_keyframes()),
            },
            "status": {
                "cli_smoke_test": "passed" if processed else "not_run",
                "desktop_app": "entry_available_manual_window_confirmation_needed",
                "report_generated": stage4_report_path().exists(),
            },
            "notes": list(delivery_notes(environment)),
            "raw_task9_result": raw,
            "raw_options": options,
        }
    else:
        static = build_static_summary(cfg)
        environment = runtime_environment()
        summary = {
            "stage": "stage4",
            "task": "project_integration",
            "generated_at": static.get("generated_at"),
            "backend": "stage3_task9_mediapipe_opencv",
            "input_video": static.get("default_video"),
            "output_video": None,
            "enabled_effects": [],
            "debug_options": {
                "fps_stat_recorded": False,
                "landmarks_keyframe_debug": False,
                "debug_sticker_geometry": False,
            },
            "mode": "not_measured",
            "fast_mode": None,
            "process_width": None,
            "max_frames": None,
            "processed_frame_count": None,
            "fps": None,
            "processing_fps": None,
            "avg_ms_per_frame": None,
            "output_resolution": {"width": None, "height": None},
            "environment": environment,
            "output_files": {
                "video": None,
                "summary": str(stage4_summary_path()),
                "report": str(stage4_report_path()),
                "keyframes": list(collected_keyframes()),
            },
            "status": {
                "cli_smoke_test": "not_run",
                "desktop_app": "entry_available_manual_window_confirmation_needed",
                "report_generated": stage4_report_path().exists(),
            },
            "notes": list(delivery_notes(environment)),
        }

    environment = runtime_environment()
    summary["environment"] = environment
    summary["notes"] = list(dict.fromkeys(list(summary.get("notes", [])) + list(delivery_notes(environment))))
    summary.setdefault("output_files", {})
    summary["output_files"]["summary"] = str(stage4_summary_path())
    summary["output_files"]["report"] = str(stage4_report_path())
    summary["output_files"]["keyframes"] = list(collected_keyframes())
    if output_video_path(summary):
        summary["output_files"]["video"] = str(output_video_path(summary))
        metadata = video_metadata(output_video_path(summary))
        if metadata.get("width") and metadata.get("height"):
            summary["output_resolution"] = {"width": metadata["width"], "height": metadata["height"]}
    summary.setdefault("status", {})
    summary["status"]["report_generated"] = True
    summary.setdefault("full_length_export", summary.get("max_frames") is None)
    summary.setdefault(
        "original_resolution_export",
        summary.get("process_width") is None
        and summary.get("fast_mode") is False,
    )
    summary.setdefault("user_selected_output_path", False)
    desktop_status = summary.get("status", {}).get("desktop_app")
    if (
        environment.get("pyside6_version") or environment.get("pyqt6_version")
    ) and not str(desktop_status).startswith("launch_smoke"):
        summary["status"]["desktop_app"] = "entry_available_manual_window_confirmation_needed"
    return summary


def render_list(values: List[str]) -> str:
    return "\n".join("  - {}".format(value) for value in values) if values else "  - None"


def render_report(summary: Dict[str, Any]) -> str:
    environment = summary["environment"]
    output_files = summary["output_files"]
    resolution = summary.get("output_resolution", {})
    notes = "\n".join("- {}".format(note) for note in summary.get("notes", []))
    keyframes = render_list(output_files.get("keyframes", []))
    return """# Stage4 项目集成报告：本地人脸视觉应用系统

## 1. 集成目标

Stage4 第一部分的目标是把前面阶段的人脸视觉模块整理为统一的本地应用系统。第一版重点完成 Stage3 Task9 动态人脸特效的应用化，提供 CLI 视频处理入口和本地桌面应用入口，不重新训练模型，不下载数据集。

## 2. 系统结构

- `stage-4/configs/`: Stage4 应用配置。
- `stage-4/code/`: 公共路径工具、Task9 backend adapter、CLI、桌面应用入口和报告生成脚本。
- `stage-4/reports/`: Stage4 交付报告、summary、checklist 和运行资产。
- `stage-4/reports/assets/`: 输出视频、关键帧等验收资产。
- `stage-4/reports/summaries/`: JSON summary 和 delivery checklist。

## 3. 模块复用方式

- Stage3 Task9 是当前实时/视频主链路，Stage4 通过 adapter 复用 `FaceEffectsProcessor.process_frame` 和 `process_video`。
- Stage2 ArcFace 后续作为身份识别升级接口。
- Stage3 Task7 StarGAN 后续作为离线属性编辑入口。
- Stage3 Task8 3DDFA 后续作为离线三维重建入口。

## 4. 当前已实现功能

- 视频输入：支持通过 `--video` 指定输入 mp4。
- 可选特效：支持按需启用 Task9 的视觉特效。
- fast mode：用于缩放预览和快速验收。
- 处理宽度设置：支持 `--process-width`。
- 最大帧数控制：支持 `--max-frames`。
- 输出视频：默认生成到 `stage-4/reports/assets/videos/`，桌面本地导入页也支持用户选择完整保存路径。
- summary 记录：记录运行参数、真实处理结果、环境版本和输出文件。
- 本地桌面应用入口：`python stage-4/code/stage4_desktop_app.py`，采用稳定优先的 CLI 子进程导出模式。

## 导出策略优化

本地视频导出的默认策略是完整长度、原始分辨率：未勾选快速预览时，桌面应用不会向 CLI 传入 `--fast-mode`、`--process-width` 或 `--max-frames`。Stage4 backend 也会清空继承自 Task9 配置中的 `process_width/process_height`，避免无意把原视频缩放到 1280x720。

快速预览仍然保留为可选模式：勾选 fast mode 后才会传入缩放宽度和最大帧数，适合短时验收或调试。桌面本地导入页支持用户选择完整输出路径；如果未选择，则使用时间戳文件名保存到 `stage-4/reports/assets/videos/` 或 `stage-4/reports/assets/images/`。

## 5. 可选特效设计

视觉特效：

- glasses
- hat
- smooth
- whiten
- lipstick

调试/统计：

- fps
- landmarks

`fps` 和 `landmarks` 不作为美颜特效，只用于运行统计或调试展示。

## 6. 真实运行结果

- 输入视频: `{input_video}`
- 输出视频: `{output_video}`
- enabled effects: `{enabled_effects}`
- mode: `{mode}`
- fast_mode: `{fast_mode}`
- process_width: `{process_width}`
- max_frames: `{max_frames}`
- full_length_export: `{full_length_export}`
- original_resolution_export: `{original_resolution_export}`
- user_selected_output_path: `{user_selected_output_path}`
- processed_frame_count: `{processed_frame_count}`
- processing_fps: `{processing_fps}`
- avg_ms_per_frame: `{avg_ms_per_frame}`
- output resolution: `{output_width}` x `{output_height}`
- keyframes:
{keyframes}

## 7. 运行命令

完整导出默认命令：

```bash
python stage-4/code/stage4_run_cli.py \\
  --video stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4 \\
  --effects glasses whiten \\
  --output-video stage-4/reports/assets/videos/stage4_full_export.mp4 \\
  --summary stage-4/reports/summaries/stage4_full_export_summary.json
```

快速预览验收命令：

```bash
python stage-4/code/stage4_run_cli.py \\
  --video stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4 \\
  --effects glasses whiten \\
  --fast-mode \\
  --process-width 720 \\
  --max-frames 30 \\
  --output-video stage-4/reports/assets/videos/stage4_task9_effects_export.mp4 \\
  --summary stage-4/reports/summaries/stage4_integration_summary.json
```

当前第一版 CLI 使用 `--video/--output-video` 旧接口；`--input-video/--mode/--output-dir` 不是本次交付接口。

## 8. 桌面应用说明

```bash
python stage-4/code/stage4_desktop_app.py
```

当前环境检测到 PySide6: `{pyside6_version}`。桌面应用是稳定优先版本：GUI 进程只负责参数选择、命令预览、启动/停止 CLI 子进程和显示日志，不在 GUI 进程中 import `cv2`、`numpy`、`mediapipe`、`stage4_backend` 或 Stage3 Task9。这样可以避开 macOS GUI 进程与 OpenCV/MediaPipe 原生库的冲突。

desktop_app status: `{desktop_status}`

Safe mode:

```bash
python stage-4/code/stage4_desktop_app.py --safe
```

Safe mode 只打开 GUI，不启动导出进程，也不会加载视频处理依赖。普通模式下进入“实时视频”页面会自动启动 worker 子进程，GUI 通过预览图片轮询把实时画面嵌入应用窗口内部。

UI 检查 summary: `stage-4/reports/summaries/stage4_ui_check_summary.json`

## 应用界面 V3：嵌入式实时预览与实时特效控制

Stage4 UI V3 将应用入口分为两个页面：

- 本地导入：支持图片和视频文件处理。视频继续调用已跑通的 `stage4_run_cli.py`，图片调用 `stage4_process_image_cli.py`，输出到 `stage-4/reports/assets/images/`。
- 实时视频：进入页面后自动启动 `stage4_live_camera_worker.py` worker 子进程，实时画面嵌入在应用窗口内部。

两条链路都支持选择视觉特效：`glasses`、`hat`、`smooth`、`whiten`、`lipstick`。实时页的特效开关和 `smooth_strength`、`whiten_strength`、`lipstick_alpha` 强度滑条会立即写入 `stage-4/reports/runtime/live_controls.json`，worker 每帧读取配置并更新处理参数。`fps` 和 `landmarks` 仅作为统计/调试信息，不作为美颜特效。

为了规避 macOS 上 PySide6 GUI 进程直接加载 OpenCV/MediaPipe 原生库导致的 Bus error，GUI 主进程不直接 import `cv2`、`numpy`、`mediapipe`、`stage4_backend` 或 Stage3 Task9；它只负责参数选择、写控制 JSON、读取预览 JPG、读取状态 JSON、启动/停止 QProcess 子进程和显示日志。实时处理在 worker 子进程中完成，worker 输出 `stage-4/reports/runtime/live_preview.jpg` 和 `stage-4/reports/runtime/live_status.json`。当前版本以课程项目级交互演示为目标，不声明商业级实时美颜或 AR 效果。

桌面窗口支持宽度和高度自由调整。实时预览区域使用 Expanding 布局并按 `KeepAspectRatio` 缩放到当前 QLabel 尺寸；右侧控制面板放入 `QScrollArea`，在窗口高度不足时显示纵向滚动条，避免控制项把窗口高度锁死。本地导入页的右侧参数面板也使用滚动容器，命令预览和日志区域设置较小高度上限以保持窗口可缩放。

UI V2 summary: `stage-4/reports/summaries/stage4_ui_v2_summary.json`
UI V3 summary: `stage-4/reports/summaries/stage4_ui_v3_summary.json`

## 实时录像功能

实时视频页面支持开始/停止录像。GUI 主进程不直接写视频文件，而是通过 `stage-4/reports/runtime/live_controls.json` 向 worker 子进程写入 `recording`、`recording_output_path`、`recording_fps` 和 `recording_fourcc` 控制字段；worker 子进程使用 `cv2.VideoWriter` 将已经叠加特效后的实时输出帧写出为 mp4。

录像过程中仍可切换特效和调整强度，写入视频的是每一帧处理后的实时画面，因此输出会记录切换后的效果。用户可以在实时页选择完整 mp4 保存路径；如果未选择，默认保存目录为 `stage-4/reports/assets/recordings/`，默认文件名格式为 `live_recording_YYYYMMDD_HHMMSS.mp4`。录像状态、保存路径、用户是否选择路径、帧数和错误信息会写入 `stage-4/reports/runtime/live_status.json` 并显示在桌面应用中。

## Hat 贴纸几何修复

帽子贴纸已从基于全局 face bbox / face center 的简单定位，改为基于左右眼中心、人脸局部 x/up 坐标轴、forehead anchor 和贴纸内部 anchor 的局部几何定位。帽子的目标锚点由 eye midpoint 沿 face_up_axis 上移得到，贴纸内部使用底部中心附近的 anchor 对齐到该目标点；因此人物偏离画面中心或头部倾斜时，帽子会随人脸旋转和位移进行更稳定贴合。眼镜逻辑保持原有眼睛中心与眼距驱动方式。

## 9. 性能说明

- `fast_preview` 是缩放预览模式。
- 当前不声称商业级实时。
- MediaPipe + OpenCV Python pipeline 主要由 CPU 执行。
- `quality_export` 不一定实时。
- 桌面端通过 CLI 子进程导出视频；真实处理性能以 CLI summary 为准。

## 10. 局限性

- 快速运动、遮挡、极端姿态下贴纸可能漂移。
- 美白、磨皮、口红是传统 OpenCV 效果，不是商业级美颜。
- 当前第一版重点是 Task9 动态特效应用化。
- ArcFace、StarGAN、3DDFA 暂作为后续升级接口。
- `python -m pip check` 结果: `{pip_check_output}`

## 11. 后续升级计划

- V2：接入 ArcFace 低频身份识别。
- V3：加入 StarGAN 图片属性编辑页面。
- V4：加入 3DDFA 单图三维重建页面。
- V5：用 PyInstaller 打包为桌面应用。
- V6：优化 UI、视频写出和 ROI 处理性能。

## 环境版本

- Python: `{python_version}`
- cv2: `{cv2_version}`
- numpy: `{numpy_version}`
- mediapipe: `{mediapipe_version}`
- has_mediapipe_solutions: `{has_mediapipe_solutions}`
- has_face_mesh: `{has_face_mesh}`

## Notes

{notes}
""".format(
        input_video=summary.get("input_video"),
        output_video=summary.get("output_video"),
        enabled_effects=", ".join(summary.get("enabled_effects", [])),
        mode=summary.get("mode"),
        fast_mode=summary.get("fast_mode"),
        process_width=fmt(summary.get("process_width"), 0),
        max_frames=fmt(summary.get("max_frames"), 0),
        full_length_export=summary.get("full_length_export"),
        original_resolution_export=summary.get("original_resolution_export"),
        user_selected_output_path=summary.get("user_selected_output_path"),
        processed_frame_count=fmt(summary.get("processed_frame_count"), 0),
        processing_fps=fmt(summary.get("processing_fps")),
        avg_ms_per_frame=fmt(summary.get("avg_ms_per_frame")),
        output_width=fmt(resolution.get("width"), 0),
        output_height=fmt(resolution.get("height"), 0),
        keyframes=keyframes,
        pyside6_version=environment.get("pyside6_version") or "not_available",
        desktop_status=summary.get("status", {}).get("desktop_app"),
        pip_check_output=environment.get("pip_check", {}).get("output", "not_measured"),
        python_version=environment.get("python_version"),
        cv2_version=environment.get("cv2_version"),
        numpy_version=environment.get("numpy_version"),
        mediapipe_version=environment.get("mediapipe_version"),
        has_mediapipe_solutions=environment.get("has_mediapipe_solutions"),
        has_face_mesh=environment.get("has_face_mesh"),
        notes=notes,
    )


def write_readme() -> None:
    text = """# Stage4 本地人脸视觉应用系统

## 环境创建

推荐使用单独环境，例如：

```bash
conda create -n cv-stage4 python=3.11
conda activate cv-stage4
```

## 依赖安装

```bash
python -m pip install -r stage-4/requirements-stage4.txt
```

依赖版本建议：

- `numpy==1.26.4`
- `opencv-contrib-python==4.11.0.86`
- `mediapipe==0.10.21`
- `PySide6`
- `pillow`

不要同时安装 `opencv-python` 和 `opencv-contrib-python`，避免 OpenCV wheel 冲突。

## CLI 运行方式

默认完整导出会保持原视频长度和原始分辨率，不传 `--fast-mode`、`--process-width` 和 `--max-frames`：

```bash
python stage-4/code/stage4_run_cli.py \\
  --video stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4 \\
  --effects glasses whiten \\
  --output-video stage-4/reports/assets/videos/stage4_full_export.mp4 \\
  --summary stage-4/reports/summaries/stage4_full_export_summary.json
```

快速预览验收可以显式传入缩放宽度和帧数限制：

```bash
python stage-4/code/stage4_run_cli.py \\
  --video stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4 \\
  --effects glasses whiten \\
  --fast-mode \\
  --process-width 720 \\
  --max-frames 30 \\
  --output-video stage-4/reports/assets/videos/stage4_task9_effects_export.mp4 \\
  --summary stage-4/reports/summaries/stage4_integration_summary.json
```

桌面本地导入页默认同样按完整长度和原始分辨率导出。只有勾选“快速预览模式”后，界面才会把 `--fast-mode`、`--process-width`、`--max-frames` 写入命令预览。

## 桌面应用运行方式

```bash
python stage-4/code/stage4_desktop_app.py
```

桌面入口依赖 PySide6；如缺失，请先安装 requirements。macOS 上桌面应用采用稳定优先模式：GUI 不直接加载 `cv2`、`numpy`、`mediapipe` 或 Stage4 backend，而是通过 QProcess 调用 CLI/worker 子进程。实时视频画面通过 `live_preview.jpg` 轮询嵌入应用窗口内部。

窗口可以自由调整宽度和高度。实时预览区域会随窗口大小伸缩，右侧控制面板在小窗口高度下会出现纵向滚动条。本地导入页也保留同样的高度自适应策略。

Safe mode:

```bash
python stage-4/code/stage4_desktop_app.py --safe
```

Safe mode 只打开窗口和命令预览，不启动导出进程，也不启动实时摄像头 worker。

## 如何使用本地导入

1. 打开桌面应用。
2. 点击首页的“本地导入”。
3. 选择 `.jpg/.jpeg/.png/.mp4/.mov/.avi` 文件。
4. 勾选视觉特效：`glasses`、`hat`、`smooth`、`whiten`、`lipstick`。
5. 可选：点击“选择保存位置”指定完整输出路径。视频默认保存为 `stage-4/reports/assets/videos/stage4_local_export_YYYYMMDD_HHMMSS.mp4`，图片默认保存为 `stage-4/reports/assets/images/stage4_image_export_YYYYMMDD_HHMMSS.jpg`。
6. 如需快速验收，再勾选“快速预览模式”并设置 process width、max frames；未勾选时默认完整导出。
7. 点击“开始处理”，在日志框查看 CLI stdout/stderr。

视频会调用 `stage4_run_cli.py`，图片会调用 `stage4_process_image_cli.py`。

## 如何使用实时视频

1. 点击首页的“实时视频”。
2. 页面会自动启动 `stage4_live_camera_worker.py`。
3. 默认 camera index 为 `0`，画面会嵌入在应用窗口左侧预览区域。
4. 勾选或取消 `glasses`、`hat`、`smooth`、`whiten`、`lipstick` 会立即写入 `stage-4/reports/runtime/live_controls.json`。
5. 调整 `smooth_strength`、`whiten_strength`、`lipstick_alpha` 滑条会实时生效，不需要重新打开摄像头。

macOS 第一次使用摄像头时可能弹出系统权限请求。如果摄像头打不开，请到“系统设置 > 隐私与安全性 > 摄像头”为 Terminal/Python 授权。

实时预览内部文件：

- 控制文件：`stage-4/reports/runtime/live_controls.json`
- 预览帧：`stage-4/reports/runtime/live_preview.jpg`
- 状态文件：`stage-4/reports/runtime/live_status.json`

实时画面嵌入在应用内，但 OpenCV/MediaPipe 仍运行在 worker 子进程中。这是为了避免 PySide6 GUI 主进程和 OpenCV/MediaPipe 原生库在 macOS 上冲突。

点击“保存截图”会把当前预览画面保存到 `stage-4/reports/assets/screenshots/live_snapshot_YYYYMMDD_HHMMSS.jpg`。

## 如何使用实时录像

1. 进入“实时视频”页面并等待摄像头画面出现。
2. 点击“开始录像”，按钮会变为“停止录像”。
3. 录像保存的是已经叠加特效后的实时画面。
4. 录像过程中可以继续切换特效或调整强度，输出视频会记录变化后的效果。
5. 可选：点击“选择录像保存位置”指定完整 mp4 输出路径。
6. 点击“停止录像”后，worker 会释放 `VideoWriter` 并保存 mp4。
7. 未选择路径时，默认录像目录：`stage-4/reports/assets/recordings/`，文件名为 `live_recording_YYYYMMDD_HHMMSS.mp4`。
8. 如果录像为空或打不开，先确认录制期间摄像头画面正常、录像帧数大于 0，并查看实时页日志和 `stage-4/reports/runtime/live_status.json` 中的 `recording_error`。

## 输出文件位置

- CLI 输出视频：由 `--output-video` 指定；默认配置为 `stage-4/reports/assets/videos/stage4_task9_effects_export.mp4`
- 桌面视频导出：用户可选完整路径；未选择时为 `stage-4/reports/assets/videos/stage4_local_export_YYYYMMDD_HHMMSS.mp4`
- 图片导出：用户可选完整路径；未选择时为 `stage-4/reports/assets/images/stage4_image_export_YYYYMMDD_HHMMSS.jpg`
- 实时截图：`stage-4/reports/assets/screenshots/`
- 实时录像：用户可选完整路径；未选择时为 `stage-4/reports/assets/recordings/live_recording_YYYYMMDD_HHMMSS.mp4`
- 实时控制文件：`stage-4/reports/runtime/live_controls.json`
- 实时预览帧：`stage-4/reports/runtime/live_preview.jpg`
- 实时状态：`stage-4/reports/runtime/live_status.json`
- 关键帧：`stage-4/reports/assets/keyframes/`
- Summary：`stage-4/reports/summaries/stage4_integration_summary.json`
- 报告：`stage-4/reports/stage4_project_integration_report.md`
- Checklist：`stage-4/reports/summaries/stage4_delivery_checklist.json`
- UI 检查：`stage-4/reports/summaries/stage4_ui_check_summary.json`
- UI V2：`stage-4/reports/summaries/stage4_ui_v2_summary.json`
- UI V3：`stage-4/reports/summaries/stage4_ui_v3_summary.json`

## 常见问题

- `ModuleNotFoundError: cv2`: 确认已安装 `opencv-contrib-python==4.11.0.86`。
- `ModuleNotFoundError: mediapipe`: 确认 Python 版本与 mediapipe wheel 兼容。
- `PySide6` 缺失：安装 requirements 后再启动桌面应用。
- `pip check` 提示 mediapipe platform warning：当前实测 import 和 Task9 smoke 可运行，但该提示应保留在交付 notes 中。
- 输出不实时：`fast_preview` 是缩放预览模式，`quality_export` 不保证实时。
- macOS GUI 进程 Bus error：使用当前桌面版本，它通过 CLI/worker 子进程处理视频和实时摄像头，避免 GUI 进程直接加载 OpenCV/MediaPipe 原生库。
- 摄像头打不开：检查 macOS 摄像头权限，确认 Terminal/Python 已授权。
- 为什么实时画面能嵌入应用但不触发 GUI import：worker 子进程写 `live_preview.jpg`，GUI 用 Qt 读取图片并刷新 QLabel。
- 特效开关为什么能实时生效：GUI 每次变更都会写 `live_controls.json`，worker 每帧读取配置。
- 如何关闭实时视频：点击“关闭摄像头”或返回首页。
- 如何截图：点击“保存截图”。
- 如何录像：在“实时视频”页点击“开始录像”，结束时点击“停止录像”。
- 如何选择输出路径：本地导入页和实时录像页都提供保存位置选择；如果目标文件已存在，应用会自动追加后缀避免覆盖。
- 录像文件打不开：确认 `live_status.json` 中 `last_recording_frame_count` 大于 0，必要时换一个播放器或重新录制一小段。
"""
    write_text(README_PATH, text)


def write_ui_v2_summary() -> None:
    write_json(
        UI_V2_SUMMARY_PATH,
        {
            "app_version": "stage4_ui_v2",
            "home_page": True,
            "local_import_page": True,
            "realtime_camera_page": True,
            "local_video_export_supported": True,
            "local_image_export_supported": Path("stage-4/code/stage4_process_image_cli.py").exists(),
            "realtime_camera_supported": Path("stage-4/code/stage4_live_camera_cli.py").exists(),
            "gui_uses_subprocess_for_cv": True,
            "gui_imports_cv2": False,
            "gui_imports_mediapipe": False,
            "gui_imports_backend": False,
            "realtime_cli_entry": str(Path("stage-4/code/stage4_live_camera_cli.py").resolve()),
            "local_cli_entry": {
                "video": str(Path("stage-4/code/stage4_run_cli.py").resolve()),
                "image": str(Path("stage-4/code/stage4_process_image_cli.py").resolve()),
            },
            "known_limitations": [
                "Realtime camera uses an independent OpenCV subprocess window.",
                "The desktop GUI does not embed realtime frames.",
                "macOS camera permission must be granted to the terminal/Python process.",
                "Current realtime mode is for functional validation, not commercial-grade AR.",
            ],
            "manual_check_required": True,
        },
    )


def write_ui_v3_summary() -> None:
    write_json(
        UI_V3_SUMMARY_PATH,
        {
            "app_version": "stage4_ui_v3",
            "home_page": True,
            "local_import_page": True,
            "realtime_camera_page": True,
            "embedded_realtime_preview": True,
            "realtime_worker_subprocess": True,
            "gui_imports_cv2": False,
            "gui_imports_mediapipe": False,
            "gui_imports_backend": False,
            "realtime_effect_toggle_supported": True,
            "realtime_strength_adjust_supported": True,
            "local_export_full_length_by_default": True,
            "local_export_original_resolution_by_default": True,
            "fast_preview_optional": True,
            "user_selectable_local_output_path": True,
            "user_selectable_recording_output_path": True,
            "default_fast_mode": False,
            "default_process_width": None,
            "default_max_frames": None,
            "resizable_window_supported": True,
            "height_resize_supported": True,
            "right_panel_scroll_area": True,
            "preview_area_expanding": True,
            "fixed_height_removed": True,
            "realtime_recording_supported": True,
            "recording_control_via_live_controls": True,
            "recording_saved_to_local": True,
            "recording_output_dir": str(Path("stage-4/reports/assets/recordings").resolve()),
            "recording_writer_in_worker": True,
            "camera_permission_note": "实时视频需要摄像头权限。首次进入实时视频时，macOS 可能弹出权限请求。",
            "worker_script": str(Path("stage-4/code/stage4_live_camera_worker.py").resolve()),
            "controls_path": str(Path("stage-4/reports/runtime/live_controls.json").resolve()),
            "preview_path": str(Path("stage-4/reports/runtime/live_preview.jpg").resolve()),
            "status_path": str(Path("stage-4/reports/runtime/live_status.json").resolve()),
            "known_limitations": [
                "Realtime preview uses file polling, so latency depends on disk and CPU load.",
                "The current UI is a course-project interaction demo, not commercial-grade realtime beautification.",
                "Camera access may require macOS privacy permission for Terminal or the active Python launcher.",
                "Changing camera index while running requires reopening the worker.",
            ],
            "manual_check_required": True,
        },
    )


def build_checklist(summary: Dict[str, Any]) -> Dict[str, Any]:
    output_files = summary.get("output_files", {})
    return {
        "has_cli": Path("stage-4/code/stage4_run_cli.py").exists(),
        "has_desktop_app_entry": Path("stage-4/code/stage4_desktop_app.py").exists(),
        "has_backend": Path("stage-4/code/stage4_backend.py").exists(),
        "has_config": Path("stage-4/configs/stage4_app_config.py").exists(),
        "has_summary": stage4_summary_path().exists(),
        "has_report": stage4_report_path().exists(),
        "has_readme": README_PATH.exists(),
        "has_requirements": Path("stage-4/requirements-stage4.txt").exists(),
        "has_output_video": bool(output_files.get("video") and Path(output_files["video"]).exists()),
        "has_keyframes": bool(output_files.get("keyframes")),
        "cli_smoke_test_passed": summary.get("status", {}).get("cli_smoke_test") == "passed",
        "desktop_app_checked": summary.get("status", {}).get("desktop_app") is not None,
        "stage2_modified": False,
        "stage3_modified": False,
        "known_limitations": [
            "fast_preview is scaled preview, not a commercial realtime guarantee.",
            "MediaPipe + OpenCV Python pipeline is primarily CPU-bound.",
            "Stickers can drift under occlusion, fast motion, or extreme pose.",
            "OpenCV beauty effects are not commercial-grade beautification.",
            "ArcFace, StarGAN, and 3DDFA are reserved for later upgrades.",
        ],
        "next_steps": [
            "Manually confirm the PySide6 desktop window and controls.",
            "Package with PyInstaller after UI acceptance.",
            "Optimize ROI processing and video writing if longer videos are required.",
        ],
    }


def main() -> None:
    ensure_stage4_dirs()
    cfg = load_python_config()
    summary = normalize_summary(cfg)
    write_readme()
    write_ui_v2_summary()
    write_ui_v3_summary()
    write_json(stage4_summary_path(), summary)
    write_text(stage4_report_path(), render_report(summary))
    checklist = build_checklist(summary)
    write_json(CHECKLIST_PATH, checklist)
    print("Wrote {}".format(rel_to_repo(stage4_summary_path())))
    print("Wrote {}".format(rel_to_repo(stage4_report_path())))
    print("Wrote {}".format(rel_to_repo(README_PATH)))
    print("Wrote {}".format(rel_to_repo(CHECKLIST_PATH)))
    print("Wrote {}".format(rel_to_repo(UI_V2_SUMMARY_PATH)))
    print("Wrote {}".format(rel_to_repo(UI_V3_SUMMARY_PATH)))


if __name__ == "__main__":
    main()
