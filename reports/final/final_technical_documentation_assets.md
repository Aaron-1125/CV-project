# 最终技术文档资源索引

本文档列出最终技术文档引用的关键图片、视频和 summary 资源。资源保留在原阶段目录中，`reports/final/` 不复制图片或视频。

## Stage1 资源

| 资源 | 路径 | 用途 |
| -- | -- | -- |
| CelebA 属性统计图 | `stage-1/reports/assets/dataset/celeba_attribute_top15.png` | 展示 CelebA 属性分布 |
| CelebA 样本图 | `stage-1/reports/assets/dataset/celeba_samples.png` | 展示 Stage1 数据探索样本 |
| LFW 样本图 | `stage-1/reports/assets/dataset/lfw_samples.png` | 展示 LFW 数据样本 |
| 基础检测可视化 | `stage-1/reports/assets/detection/mmdet_lfw_public_00_faces.jpg` | 展示 Stage1 基础人脸检测 |
| LFW ROC | `stage-1/reports/assets/evaluation/lfw_roc_curve.png` | 展示 Stage1 InsightFace baseline 验证 |
| Stage1 summary | `stage-1/reports/stage1_task2_2_dataset_summary.json` | 数据规模和属性统计来源 |

## Stage2 资源

| 资源 | 路径 | 用途 |
| -- | -- | -- |
| Task3 检测指标图 | `stage-2/reports/task3_v2/assets/evaluation/scrfd_like_640_eval_metrics.png` | 展示 SCRFD-like R50-FPN 640 检测结果 |
| Task3 threshold sweep | `stage-2/reports/task3_v2/assets/diagnostics/ssd300_threshold_sweep.png` | 展示 SSD300 误检诊断 |
| Task3 检测样例 | `stage-2/reports/task3_v2/assets/detection/detection_00_0_Parade_Parade_0_102.jpg` | 展示检测可视化 |
| Task3 baseline comparison | `stage-2/reports/task3_v2/summaries/task3_v2_baseline_comparison.json` | AP50、Precision、Recall、TP/FP/FN 指标来源 |
| Task4 NME 图 | `stage-2/reports/task4/assets/evaluation/300w_nme_metrics.png` | 展示关键点 NME |
| Task4 对齐样例 | `stage-2/reports/task4/assets/alignment/00_lfpw_testset_image_0208_before_after.jpg` | 展示关键点与对齐效果 |
| Task4 summary | `stage-2/reports/task4/summaries/300w_full_eval_summary.json` | common/challenge/full NME 来源 |
| Task4 v2 comparison | `stage-2/reports/task4_v2/summaries/task4_v2_results_comparison.json` | 增强实验负向消融来源 |
| Task5 LFW ROC | `stage-2/reports/task5/assets/evaluation/lfw_roc_curve.png` | 展示 ArcFace 验证曲线 |
| Task5 similarity histogram | `stage-2/reports/task5/assets/evaluation/lfw_similarity_histogram.png` | 展示同/异人相似度分布 |
| Task5 final summary | `stage-2/reports/task5/summaries/insightface_full_lfw_eval_summary.json` | LFW 99.80% 验收精度来源 |
| Task6 latency comparison | `stage-2/reports/task6/final/assets/evaluation/final_latency_comparison.png` | 展示推理加速对比 |
| Task6 ONNX comparison | `stage-2/reports/task6/assets/evaluation/task6_onnx_comparison.png` | 展示 ONNX 评估结果 |
| Task6 final summary | `stage-2/reports/task6/final/summaries/final_latency_summary.json` | latency、throughput、speedup 来源 |

## Stage3 资源

| 资源 | 路径 | 用途 |
| -- | -- | -- |
| Task7 Black_Hair 可视化 | `stage-3/reports/task7/assets/evaluation/iter_200000/source_vs_generated/Black_Hair_source_vs_generated.jpg` | 展示 StarGAN 属性编辑效果 |
| Task7 Blond_Hair 可视化 | `stage-3/reports/task7/assets/evaluation/iter_200000/source_vs_generated/Blond_Hair_source_vs_generated.jpg` | 展示 StarGAN 属性编辑效果 |
| Task7 fixed grid | `stage-3/reports/task7/assets/pretrained/pretrained_200000_fixed_grid.jpg` | 展示固定样本生成效果 |
| Task7 evaluation summary | `stage-3/reports/task7/summaries/task7_evaluation_summary.json` | 属性成功率、身份保持、FID/IS 状态来源 |
| Task8 2D sparse | `stage-3/reports/task8/assets/reconstruction/sample_000/official_2d_sparse.jpg` | 展示 3DDFA_V2 2D sparse landmarks |
| Task8 3D overlay | `stage-3/reports/task8/assets/reconstruction/sample_000/official_3d_overlay.jpg` | 展示单图三维重建 overlay |
| Task8 pose | `stage-3/reports/task8/assets/reconstruction/sample_000/official_pose.jpg` | 展示姿态估计 |
| Task8 OBJ mesh | `stage-3/reports/task8/assets/reconstruction/sample_000/official_mesh.obj` | 3D mesh 交付物 |
| Task8 multiview | `stage-3/reports/task8/assets/rendered_views/sample_000/multiview_grid.jpg` | 展示多视角渲染 |
| Task8 reconstruction summary | `stage-3/reports/task8/summaries/task8_reconstruction_summary.json` | 500 样本成功率来源 |
| Task9 demo video | `stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4` | Stage3 动态特效最终 demo |
| Task9 effects summary | `stage-3/reports/task9/summaries/task9_effects_summary.json` | 质量路径处理帧数/FPS/耗时来源 |
| Task9 performance summary | `stage-3/reports/task9/summaries/task9_performance_summary.json` | profile benchmark 来源 |

## Stage4 资源

| 资源 | 路径 | 用途 |
| -- | -- | -- |
| Stage4 集成报告 | `stage-4/reports/stage4_project_integration_report.md` | Stage4 系统集成说明 |
| Stage4 README | `stage-4/README_STAGE4.md` | 运行和打包说明 |
| Stage4 keyframe start | `stage-4/reports/assets/keyframes/frame_start.jpg` | 展示输出关键帧 |
| Stage4 keyframe middle | `stage-4/reports/assets/keyframes/frame_middle.jpg` | 展示输出关键帧 |
| Stage4 keyframe end | `stage-4/reports/assets/keyframes/frame_end.jpg` | 展示输出关键帧 |
| Stage4 CLI smoke video | `stage-4/reports/assets/videos/stage4_task9_effects_export.mp4` | Stage4 真实 CLI smoke 输出 |
| Stage4 fast preview regression video | `stage-4/reports/assets/videos/stage4_fast_preview_regression_test.mp4` | 快速预览回归输出 |
| Stage4 hat geometry test video | `stage-4/reports/assets/videos/stage4_hat_geometry_fix_test.mp4` | 帽子几何修复输出 |
| Stage4 integration summary | `stage-4/reports/summaries/stage4_integration_summary.json` | Stage4 CLI smoke test 指标来源 |
| Stage4 UI summary | `stage-4/reports/summaries/stage4_ui_v3_summary.json` | UI 功能覆盖来源 |
| Stage4 packaging summary | `stage-4/reports/summaries/stage4_packaging_summary.json` | macOS app 封装状态来源 |
| Stage4 packaging checklist | `stage-4/reports/summaries/stage4_packaging_checklist.json` | app bundle 人工验收清单 |

## 打包资源

| 资源 | 路径 | 用途 |
| -- | -- | -- |
| Unified app entry | `stage-4/code/stage4_app_main.py` | 打包后 GUI/CLI/worker 统一入口 |
| Packaging utils | `stage-4/code/stage4_packaging_utils.py` | 源码/frozen 路径兼容 |
| PyInstaller spec | `stage-4/packaging/stage4_face_effects.spec` | macOS app 构建配置 |
| Build script | `stage-4/packaging/build_macos_app.sh` | 构建 `Stage4FaceEffects.app` |
| macOS app | `stage-4/packaging/dist/Stage4FaceEffects.app` | 可双击应用包 |

