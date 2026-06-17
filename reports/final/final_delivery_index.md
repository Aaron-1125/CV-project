# 最终交付物索引

| 类别 | 内容 | 路径 | 说明 |
| -- | -- | -- | -- |
| Stage1 | 数据集与基础能力报告 | `stage-1/reports/stage1_face_basics_dataset_report.md` | CelebA/LFW 数据探索、基础检测与识别材料 |
| Stage1 | 数据集 summary | `stage-1/reports/stage1_task2_2_dataset_summary.json` | CelebA、LFW 数据统计 |
| Stage1 | 基础识别 summary | `stage-1/reports/assets/evaluation/lfw_insightface_verification_summary.json` | InsightFace buffalo_l baseline，LFW accuracy 0.9665 |
| Stage2 | Task3 检测报告 | `stage-2/reports/task3/stage2_task3_face_detection_training_report.md` | 初版 WIDERFace 检测训练报告 |
| Stage2 | Task3 改进计划 | `stage-2/reports/task3_v2/stage2_task3_v2_detection_improvement_plan.md` | SSD300 误检诊断与 SCRFD-like 改进 |
| Stage2 | Task3 改进 summary | `stage-2/reports/task3_v2/summaries/task3_v2_baseline_comparison.json` | baseline 与 SCRFD-like 指标对比 |
| Stage2 | Task3 评估图 | `stage-2/reports/task3_v2/assets/evaluation/scrfd_like_640_eval_metrics.png` | 检测指标可视化 |
| Stage2 | Task3 阈值诊断图 | `stage-2/reports/task3_v2/assets/diagnostics/ssd300_threshold_sweep.png` | SSD300 threshold sweep |
| Stage2 | Task4 关键点报告 | `stage-2/reports/task4/stage2_task4_landmark_alignment_report.md` | 300W/HRNet 关键点任务报告 |
| Stage2 | Task4 NME summary | `stage-2/reports/task4/summaries/300w_full_eval_summary.json` | common/challenge/full NME |
| Stage2 | Task4 v2 comparison | `stage-2/reports/task4_v2/summaries/task4_v2_results_comparison.json` | 增强实验负向消融 |
| Stage2 | Task4 可视化 | `stage-2/reports/task4/assets/alignment/` | 对齐、关键点和 before/after 图 |
| Stage2 | Task5 ArcFace 报告 | `stage-2/reports/task5/stage2_task5_arcface_training_report.md` | ArcFace 训练与官方 InsightFace 链路 |
| Stage2 | Task5 验收 summary | `stage-2/reports/task5/summaries/insightface_full_lfw_eval_summary.json` | LFW accuracy 0.9980 |
| Stage2 | Task5 ROC | `stage-2/reports/task5/assets/evaluation/lfw_roc_curve.png` | LFW ROC 曲线 |
| Stage2 | Task6 最终报告 | `stage-2/reports/task6/final/stage2_task6_final_latency_report.md` | 模型推理加速最终报告 |
| Stage2 | Task6 latency summary | `stage-2/reports/task6/final/summaries/final_latency_summary.json` | PyTorch/ONNX/FP16 延迟与 speedup |
| Stage2 | Task6 latency 图 | `stage-2/reports/task6/final/assets/evaluation/final_latency_comparison.png` | 加速结果可视化 |
| Stage3 | Task7 StarGAN 报告 | `stage-3/reports/task7/stage3_task7_stargan_attribute_editing_report.md` | 属性编辑报告 |
| Stage3 | Task7 evaluation summary | `stage-3/reports/task7/summaries/task7_evaluation_summary.json` | 成功率、身份保持、FID/IS 状态 |
| Stage3 | Task7 属性可视化 | `stage-3/reports/task7/assets/evaluation/iter_200000/source_vs_generated/` | Source vs generated 图 |
| Stage3 | Task8 3DDFA 报告 | `stage-3/reports/task8/stage3_task8_3d_face_reconstruction_report.md` | 单图三维人脸重建报告 |
| Stage3 | Task8 reconstruction summary | `stage-3/reports/task8/summaries/task8_reconstruction_summary.json` | 500 样本重建成功率 |
| Stage3 | Task8 渲染 summary | `stage-3/reports/task8/summaries/task8_render_summary.json` | 12 个 showcase 多视角渲染 |
| Stage3 | Task8 重建资源 | `stage-3/reports/task8/assets/reconstruction/sample_000/` | 2D sparse、3D overlay、pose、OBJ mesh |
| Stage3 | Task8 多视角图 | `stage-3/reports/task8/assets/rendered_views/sample_000/multiview_grid.jpg` | 多视角 mesh 渲染 |
| Stage3 | Task9 动态特效报告 | `stage-3/reports/task9/stage3_task9_dynamic_face_effects_report.md` | MediaPipe + OpenCV 动态特效报告 |
| Stage3 | Task9 effects summary | `stage-3/reports/task9/summaries/task9_effects_summary.json` | 912 帧真实视频质量路径处理结果 |
| Stage3 | Task9 performance summary | `stage-3/reports/task9/summaries/task9_performance_summary.json` | profile benchmark |
| Stage3 | Task9 demo video | `stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4` | 动态特效 demo 视频 |
| Stage4 | 集成报告 | `stage-4/reports/stage4_project_integration_report.md` | Stage4 本地应用系统报告 |
| Stage4 | README | `stage-4/README_STAGE4.md` | 环境、运行、实时录像、macOS 打包说明 |
| Stage4 | requirements | `stage-4/requirements-stage4.txt` | Stage4 运行依赖 |
| Stage4 | packaging requirements | `stage-4/requirements-packaging.txt` | PyInstaller 打包依赖 |
| Stage4 | CLI smoke summary | `stage-4/reports/summaries/stage4_integration_summary.json` | 30 帧 smoke test |
| Stage4 | UI summary | `stage-4/reports/summaries/stage4_ui_v3_summary.json` | 实时预览、录像、路径选择能力 |
| Stage4 | packaging summary | `stage-4/reports/summaries/stage4_packaging_summary.json` | macOS app 打包状态 |
| Stage4 | keyframes | `stage-4/reports/assets/keyframes/` | start/middle/end 关键帧 |
| Stage4 | output videos | `stage-4/reports/assets/videos/` | Stage4 smoke、fast preview、hat geometry test 视频 |
| App packaging | Unified entry | `stage-4/code/stage4_app_main.py` | GUI/run-cli/live-worker/write-report/check-env 统一入口 |
| App packaging | Packaging utils | `stage-4/code/stage4_packaging_utils.py` | source/frozen 路径兼容 |
| App packaging | PyInstaller spec | `stage-4/packaging/stage4_face_effects.spec` | `.app` 构建配置 |
| App packaging | Build script | `stage-4/packaging/build_macos_app.sh` | macOS app 构建脚本 |
| App packaging | macOS app | `stage-4/packaging/dist/Stage4FaceEffects.app` | 可双击运行的 app bundle |
| Final documentation | 主技术文档 | `reports/final/final_technical_documentation.md` | 完整技术文档 |
| Final documentation | 项目总结 | `reports/final/final_project_summary.md` | 简短项目总结 |
| Final documentation | 交付索引 | `reports/final/final_delivery_index.md` | 本文件 |
| Final documentation | 资源索引 | `reports/final/final_technical_documentation_assets.md` | 图片/视频/summary 资源用途说明 |
| Final documentation | Checklist | `reports/final/final_documentation_checklist.json` | 最终文档覆盖检查 |

