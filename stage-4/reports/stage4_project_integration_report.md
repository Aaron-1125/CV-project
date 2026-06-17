# Stage4 项目集成报告：本地人脸视觉应用系统

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

- 输入视频: `/Users/aaron/Documents/字节实习/task/CV project/stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4`
- 输出视频: `/Users/aaron/Documents/字节实习/task/CV project/stage-4/reports/assets/videos/stage4_task9_effects_export.mp4`
- enabled effects: `glasses, whiten`
- mode: `fast_preview`
- fast_mode: `True`
- process_width: `720`
- max_frames: `30`
- full_length_export: `False`
- original_resolution_export: `False`
- user_selected_output_path: `False`
- processed_frame_count: `30`
- processing_fps: `36.594`
- avg_ms_per_frame: `27.327`
- output resolution: `380` x `720`
- keyframes:
  - /Users/aaron/Documents/字节实习/task/CV project/stage-4/reports/assets/keyframes/frame_start.jpg
  - /Users/aaron/Documents/字节实习/task/CV project/stage-4/reports/assets/keyframes/frame_middle.jpg
  - /Users/aaron/Documents/字节实习/task/CV project/stage-4/reports/assets/keyframes/frame_end.jpg

## 7. 运行命令

完整导出默认命令：

```bash
python stage-4/code/stage4_run_cli.py \
  --video stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4 \
  --effects glasses whiten \
  --output-video stage-4/reports/assets/videos/stage4_full_export.mp4 \
  --summary stage-4/reports/summaries/stage4_full_export_summary.json
```

快速预览验收命令：

```bash
python stage-4/code/stage4_run_cli.py \
  --video stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4 \
  --effects glasses whiten \
  --fast-mode \
  --process-width 720 \
  --max-frames 30 \
  --output-video stage-4/reports/assets/videos/stage4_task9_effects_export.mp4 \
  --summary stage-4/reports/summaries/stage4_integration_summary.json
```

当前第一版 CLI 使用 `--video/--output-video` 旧接口；`--input-video/--mode/--output-dir` 不是本次交付接口。

## 8. 桌面应用说明

```bash
python stage-4/code/stage4_desktop_app.py
```

当前环境检测到 PySide6: `6.11.1`。桌面应用是稳定优先版本：GUI 进程只负责参数选择、命令预览、启动/停止 CLI 子进程和显示日志，不在 GUI 进程中 import `cv2`、`numpy`、`mediapipe`、`stage4_backend` 或 Stage3 Task9。这样可以避开 macOS GUI 进程与 OpenCV/MediaPipe 原生库的冲突。

desktop_app status: `launch_smoke_checked_no_immediate_crash_terminated_after_4s_manual_visual_confirmation_needed`

Safe mode:

```bash
python stage-4/code/stage4_desktop_app.py --safe
```

Safe mode 只打开 GUI，不启动导出进程，也不会加载视频处理依赖。普通模式下进入“实时视频”页面会自动启动 worker 子进程，GUI 通过预览图片轮询把实时画面嵌入应用窗口内部。

UI 检查 summary: `stage-4/reports/summaries/stage4_ui_check_summary.json`

## macOS 应用封装

Stage4 已增加 macOS `.app` 封装准备，应用名为 `Stage4FaceEffects.app`。打包入口统一为 `stage-4/code/stage4_app_main.py`，源码模式和 PyInstaller frozen 模式都通过该入口分发不同运行模式：

- GUI: `--gui`
- 本地导出 CLI: `--run-cli`
- 实时摄像头 worker: `--live-worker`
- 报告生成: `--write-report`
- 环境检查: `--check-env`

PyInstaller spec 位于 `stage-4/packaging/stage4_face_effects.spec`，构建脚本位于 `stage-4/packaging/build_macos_app.sh`。打包后 GUI 主进程仍不直接 import OpenCV/MediaPipe，实时 worker 和本地导出 CLI 会通过当前 app executable 加 `--live-worker` / `--run-cli` 作为子进程启动，从而继续保持 GUI 与 CV 处理进程分离。

用户生成内容不会默认写入 `.app` bundle 内部。源码模式继续使用 `stage-4/reports/`；打包模式默认写入 `~/Documents/Stage4FaceEffects/`，用户在界面中选择的输出路径优先。当前版本只做本地课程项目演示，不做 Apple notarization；如果 macOS 阻止打开，可右键 app 选择“打开”，或在系统设置中允许。

Packaging summary: `stage-4/reports/summaries/stage4_packaging_summary.json`
Packaging checklist: `stage-4/reports/summaries/stage4_packaging_checklist.json`

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
- `python -m pip check` 结果: `mediapipe 0.10.21 is not supported on this platform`

## 11. 后续升级计划

- V2：接入 ArcFace 低频身份识别。
- V3：加入 StarGAN 图片属性编辑页面。
- V4：加入 3DDFA 单图三维重建页面。
- V5：用 PyInstaller 打包为桌面应用。
- V6：优化 UI、视频写出和 ROI 处理性能。

## 环境版本

- Python: `3.11.15`
- cv2: `4.11.0`
- numpy: `1.26.4`
- mediapipe: `0.10.21`
- has_mediapipe_solutions: `True`
- has_face_mesh: `True`

## Notes

- Stage4 first delivery focuses on Task9 dynamic face effects integration.
- fast_preview is a scaled preview mode; quality_export is not guaranteed realtime.
- Task9 uses a MediaPipe + OpenCV Python pipeline, primarily CPU-bound in this setup.
- fps and landmarks are debug/statistical outputs, not beauty effects.
- ArcFace, StarGAN, and 3DDFA are reserved as later upgrade interfaces only.
- Current CLI uses the v1 interface: --video, --effects, --fast-mode, --process-width, --max-frames, --output-video, --summary.
- pip check warning: mediapipe 0.10.21 is not supported on this platform
- Desktop GUI uses QProcess to call the CLI and intentionally avoids importing cv2, numpy, mediapipe, stage4_backend, or Stage3 Task9 in the GUI process.
- Stage4 UI V3 embeds realtime preview by polling worker-generated preview JPG/status JSON while controls are written through live_controls.json.
- Hat sticker geometry uses eye centers, local face axes, forehead anchor, and sticker-anchor alignment for more stable off-center and tilted-head placement.
- Local video export defaults to full-length, original-resolution processing unless fast preview options are explicitly enabled.
- Desktop local import and realtime recording support user-selected output paths, with timestamped defaults under stage-4/reports/assets/.
