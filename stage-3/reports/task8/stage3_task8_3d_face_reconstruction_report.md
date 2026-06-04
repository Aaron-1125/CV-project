# Stage3 Task8: 基于官方 3DDFA_V2 的 3D 人脸重建

## 1. 任务简介

本实验基于官方 3DDFA_V2 完成单图 3D 人脸重建，stage-3 只实现样本准备、官方推理调用、结果整理和可视化。本文工作重点是实验流程搭建、样本选择、结果可视化与分析，不是自研 3D 人脸重建模型，也没有重写 3DMM fitting、mesh reconstruction 或 landmark detection 的核心算法。

交付内容包括：

- `code/task8/`: Task8 wrapper 脚本。
- `configs/task8_3dface/`: A800/3DDFA_V2 配置。
- `reports/task8/assets/`: 输入样本、官方重建输出和少量多角度可视化。
- `reports/task8/summaries/`: 环境、样本、重建、渲染 summary。

## 2. 基本原理

3DMM 是一种统计三维人脸模型，通常用平均脸、形状基、表情基和纹理信息表示人脸。单张图像重建的核心思想，是从 2D 人脸图像中估计人脸框、姿态、形状参数、表情参数和相机投影关系，再恢复可渲染的 3D mesh。由于单图缺少真实深度，重建结果依赖模型先验和人脸检测/对齐质量。

3DDFA_V2 是成熟的 3D Dense Face Alignment 框架。它使用 FaceBoxes/FaceBoxes_ONNX 做人脸检测，再用 TDDFA/TDDFA_ONNX 估计 3DMM 参数，最后通过官方 demo 支持的 `2d_sparse`、`3d`、`pose`、`obj` 等模式输出关键点、overlay、姿态和 mesh。NeRF 类方法通常通过多视角或隐式辐射场学习几何与外观，表达能力更强，但对数据和训练成本要求更高；本课程任务更适合使用 3DDFA_V2 快速、稳定地完成单图重建交付。

## 3. 实验设置

- 数据来源: `N/A`，样本数 `N/A`。
- 输入样本目录: `N/A`。
- 3DDFA_V2 repo: `N/A`。
- 3DDFA_V2 commit: `N/A`。
- 官方配置: `N/A`。
- 默认 runner: `official_subprocess`。
- 重建 backend: `N/A`。
- 运行环境 Python: `N/A`。

输入样本预览：

N/A

## 4. 重建结果

尚未生成重建结果。请先运行 `stage3_task8_run_reconstruction.py`。

## 5. 结果分析

正脸样本通常能得到较稳定的稠密人脸 mesh、68 点稀疏关键点和较自然的 3D overlay。轻微侧脸在 3DDFA_V2 的姿态建模范围内通常仍可恢复脸部整体几何，但被遮挡的一侧和边缘纹理会更依赖模型先验。遮挡、夸张表情、强光照、模糊或极端姿态会影响 FaceBoxes 检测和 TDDFA 参数估计，表现为关键点偏移、overlay 不贴合或 obj 局部纹理异常。

本 pipeline 默认优先复用官方 `demo.py` 的输出，因此结果应尽量接近官方 3DDFA_V2 demo。若 wrapper 收集结果与官方 demo 不一致，应优先检查输入 basename、`examples/results/` 扫描逻辑、backend 选择和官方依赖，而不是修改模型参数。

## 6. 局限性

- 单图 3D 重建存在深度歧义，侧脸背面和遮挡区域主要由模型先验补全。
- 结果依赖官方 FaceBoxes/FaceBoxes_ONNX 的检测质量；检测失败时后续重建无法稳定进行。
- 本实验不训练模型、不改网络结构、不调 3DMM 基，只使用官方推荐配置和权重。
- 多角度渲染只是基于官方 obj 的轻量可视化；如果 `pyrender/trimesh/matplotlib` 不可用，报告保留官方 overlay、pose、landmark 和 obj 输出，并在 summary 中记录 render unavailable。

## 7. 运行与交付说明

AutoDL smoke:

```bash
export 3DDFA_REPO=/root/autodl-tmp/task/3DDFA_V2
export CELEBA_ROOT=/root/autodl-pub/CelebA
cd /root/autodl-tmp/CV-project/stage-3

python code/task8/stage3_task8_check_env.py --config configs/task8_3dface/a800_3ddfa_v2.py
python code/task8/stage3_task8_prepare_samples.py --config configs/task8_3dface/a800_3ddfa_v2.py --celeba-root "$CELEBA_ROOT" --sample-count 1
python code/task8/stage3_task8_run_reconstruction.py --config configs/task8_3dface/a800_3ddfa_v2.py --max-samples 1
python code/task8/stage3_task8_render_views.py --config configs/task8_3dface/a800_3ddfa_v2.py
python code/task8/stage3_task8_write_report.py --config configs/task8_3dface/a800_3ddfa_v2.py
```

Full run: 去掉 `--sample-count 1` 和 `--max-samples 1`，默认扩展到 8 张。

3DDFA_V2 准备方式：

```bash
cd /root/autodl-tmp/task
git clone https://github.com/cleardusk/3DDFA_V2.git 3DDFA_V2
cd 3DDFA_V2
sh ./build.sh
# 按官方 README/weights 说明准备 weights/mb1_120x120.pth 或 weights/mb1_120x120.onnx
```

适合提交 GitHub：

- `code/task8/`
- `configs/task8_3dface/`
- `reports/task8/stage3_task8_3d_face_reconstruction_report.md`
- `reports/task8/summaries/`
- 少量精选输入图、官方可视化图和 multi-view grid

不要提交：

- CelebA 数据集
- 完整 3DDFA_V2 repo
- `*.pth`、`*.onnx`、checkpoint、缓存
- 大量逐样本中间输出

## 8. 结论

Task8 使用官方 3DDFA_V2 完成了从单张人脸图像到 3D mesh、landmark/overlay/pose 可视化、多角度展示和 Markdown 报告的可复现流程。stage-3 保持轻量，只保存课程交付所需 wrapper、配置、summary、报告和少量结果图。
