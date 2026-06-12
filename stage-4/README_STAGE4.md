# Stage4 本地人脸视觉应用系统

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
python stage-4/code/stage4_run_cli.py \
  --video stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4 \
  --effects glasses whiten \
  --output-video stage-4/reports/assets/videos/stage4_full_export.mp4 \
  --summary stage-4/reports/summaries/stage4_full_export_summary.json
```

快速预览验收可以显式传入缩放宽度和帧数限制：

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
