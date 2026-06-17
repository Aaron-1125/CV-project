# 项目总结

## 项目目标

本项目围绕人脸视觉任务，完成从基础算法模块到本地桌面应用的完整工程闭环。项目目标不是单一模型训练，而是覆盖人脸检测、关键点定位、人脸识别、属性编辑、三维重建、动态特效、性能优化和应用封装，最终形成可交互、可导出、可双击运行的 macOS 桌面应用。

## 完成内容

Stage1 完成 CelebA、LFW 等数据探索，以及基础检测、关键点和识别可视化，为后续任务提供数据和实验背景。

Stage2 完成人脸检测、关键点、人脸识别和推理加速。检测任务从 SSD300 baseline 迭代到 SCRFD-like R50-FPN 640，AP50 从 0.3689 提升到 0.5997，并显著降低 false positive。关键点任务完成 300W/HRNet 评估，并如实记录增强实验没有超过 baseline。ArcFace 识别使用官方 InsightFace 链路，在 aligned 112x112 LFW bin 上达到 99.80% accuracy。推理加速部分对 PyTorch FP32/FP16、ONNX FP32/FP16 和 Dynamic INT8 进行比较，最终确认 Dynamic INT8 不适合 Conv2d-heavy backbone，PyTorch FP16 CUDA 是本次记录中的最佳加速路线。

Stage3 完成 StarGAN 属性编辑、3DDFA_V2 单图三维重建和 MediaPipe 动态人脸特效。StarGAN 支持 Black_Hair、Blond_Hair、Brown_Hair、Male、Young 五个属性编辑，并完成属性成功率和身份保持评估。3DDFA_V2 在 500 张 CelebA 样本上完成 2D sparse landmarks、3D overlay、pose、OBJ mesh 和多视角渲染。Task9 基于 MediaPipe Face Mesh 和 OpenCV 实现 glasses、hat、smooth、whiten、lipstick 动态特效，并完成质量路径和 profile benchmark。

Stage4 完成项目集成和桌面应用。应用支持本地图片/视频处理、特效选择、强度调整、用户自定义导出路径、默认完整视频原始分辨率导出、实时摄像头预览、实时特效开关、截图保存、实时录像和 macOS `.app` 封装。GUI 主进程不直接加载 OpenCV/MediaPipe，实时处理由 worker 子进程完成，提高了 macOS 稳定性。

## 核心成果

- 完成跨阶段人脸视觉系统：检测、关键点、识别、属性编辑、三维重建、动态特效和应用集成。
- Stage2 Task3 检测 AP50 达到 0.5997，Precision 达到 0.3006，Recall 达到 0.6596。
- Stage2 Task5 ArcFace 在官方 InsightFace 验证链路上达到 LFW accuracy 0.9980。
- Stage2 Task6 记录最佳加速路线为 PyTorch FP16 CUDA，speedup vs PyTorch FP32 为 4.7186。
- Stage3 Task8 3DDFA_V2 在 500 张样本上成功率为 1.0。
- Stage3 Task9 动态特效完成真实视频处理，912 帧全部检测到人脸。
- Stage4 CLI smoke test 处理 30 帧，记录 FPS 约 36.59。
- macOS app `Stage4FaceEffects.app` 已生成，bundle、executable、Info.plist 均存在，codesign 为 ad_hoc。

## 技术亮点

第一，系统从模型实验扩展到本地应用。Stage4 不重新训练模型，而是复用 Stage3 Task9 动态特效作为主链路，将已有算法模块封装为用户可操作的桌面系统。

第二，采用 GUI 与 CV runtime 分离的子进程架构。PySide6 GUI 主进程不直接 import `cv2`、`mediapipe` 或 Stage3 Task9，避免 macOS 原生库冲突。实时预览通过 `live_controls.json`、`live_status.json` 和 `live_preview.jpg` 完成跨进程通信。

第三，修复了动态帽子贴纸几何问题。帽子定位从简单 bbox/face center 偏移，改为左右眼中心、人脸局部坐标系、forehead anchor 和 sticker anchor 对齐，提升人物偏移和头部倾斜时的贴合稳定性。

第四，完成 PyInstaller macOS 封装。新增统一入口 `stage4_app_main.py`，支持 GUI、run-cli、live-worker、write-report 和 check-env 模式，打包后子进程通过当前 app executable 分发，不再依赖源码脚本路径。

## 性能与应用化

Stage4 区分高质量导出和实时预览。高质量导出默认完整长度和原始分辨率，不保证实时；fast preview 是快速验收模式。Stage4 已记录 30 帧 CLI smoke test，`stage4_integration_summary.json` 中 processing_fps 为 36.5941。实时预览在人工验收中观察稳定在约 30 FPS

