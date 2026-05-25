# Stage2 Task 4.x 人脸关键点检测与对齐报告

## 1. 交付物隔离

任务 4.x 的代码、配置和报告与 3.x WIDER FACE 检测任务分开：

- 代码：`code/task4/`
- 配置：`configs/task4_mmpose/`
- 报告：`reports/task4/`
- 图表：`reports/task4/assets/`
- JSON 摘要：`reports/task4/summaries/`
- 训练输出：`work_dirs/task4/`
- 数据：`data/task4_300w/`

不复用或覆盖 `reports/assets/detection/`、`reports/assets/evaluation/`、`reports/assets/training/` 等 3.x 目录。

## 2. 环境

GPU Docker 环境在原 `stage2-gpu` 服务上追加：

- `mmpose==1.3.2`
- `xtcocotools>=1.12`
- `p7zip-full`

验证命令：

```bash
docker compose build stage2-gpu
docker compose run --rm stage2-gpu python -c "import torch, mmcv, mmpose; print(torch.cuda.is_available(), mmcv.__version__, mmpose.__version__)"
```

预期输出中 `torch.cuda.is_available()` 为 `True`，`mmpose.__version__` 为 `1.3.2`。

## 3. 数据准备

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task4/stage2_task4_2_prepare_300w.py \
    --download \
    --data-dir data/task4_300w \
    --report-dir reports/task4
```

脚本会自动下载 OpenMMLab 300W annotation。若 iBUG 官方图片分卷返回下载表单，请手动下载以下文件并放入 `stage-2/data/task4_300w/raw/`：

- `300w.zip.001`
- `300w.zip.002`
- `300w.zip.003`
- `300w.zip.004`

重新运行准备脚本后，目标结构为 `data/task4_300w/mmpose/300w/{annotations,images}`。

验收数量：

| split | images |
| --- | ---: |
| train | 3,148 |
| valid | 689 |
| valid_common | 554 |
| valid_challenge | 135 |
| test | 600 |

## 4. 训练

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task4/stage2_task4_run_mmpose.py train \
    --config configs/task4_mmpose/td-hm_hrnetv2-w18_300w_full_gpu.py \
    --work-dir work_dirs/task4/hrnetv2_w18_300w_full \
    --summary-out reports/task4/summaries/300w_full_train_summary.json \
    --loss-plot-out reports/task4/assets/training/300w_full_loss_curve.png \
    --device cuda:0
```

配置说明：

- 模型：MMPose `TopdownPoseEstimator` + HRNetv2-W18
- 输入：256x256
- 输出：68 点 heatmap
- epoch：60
- batch size：32
- optimizer：Adam
- learning rate：`1.25e-4`
- best checkpoint：按 `NME` 越低越好保存，并复制为 `work_dirs/task4/hrnetv2_w18_300w_full/best.pth`

如果 8GB 显存 OOM，将 batch size 固定降为 16，learning rate 固定降为 `6.25e-5`，仍使用完整 300W 训练集和 60 epoch。对应配置已放在 `configs/task4_mmpose/td-hm_hrnetv2-w18_300w_full_gpu_bs16.py`。

## 5. 评估

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task4/stage2_task4_run_mmpose.py test \
    --config configs/task4_mmpose/td-hm_hrnetv2-w18_300w_full_gpu.py \
    --checkpoint work_dirs/task4/hrnetv2_w18_300w_full/best.pth \
    --summary-out reports/task4/summaries/300w_full_eval_summary.json \
    --metrics-plot-out reports/task4/assets/evaluation/300w_nme_metrics.png \
    --device cuda:0
```

评估 wrapper 会分别跑：

- `face_landmarks_300w_valid.json`
- `face_landmarks_300w_valid_common.json`
- `face_landmarks_300w_valid_challenge.json`
- `face_landmarks_300w_test.json`

结果写入 `reports/task4/summaries/300w_full_eval_summary.json`，NME 图写入 `reports/task4/assets/evaluation/300w_nme_metrics.png`。

## 6. 人脸对齐

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task4/stage2_task4_3_align_faces.py \
    --config configs/task4_mmpose/td-hm_hrnetv2-w18_300w_full_gpu.py \
    --checkpoint work_dirs/task4/hrnetv2_w18_300w_full/best.pth \
    --data-dir data/task4_300w/mmpose/300w \
    --out-dir reports/task4/assets/alignment \
    --summary-out reports/task4/summaries/300w_alignment_summary.json \
    --visualize-count 8 \
    --device cuda:0
```

对齐流程：

1. 从 300W annotation 读取人脸 bbox。
2. 使用训练好的 HRNet 预测 68 点。
3. 从 68 点提取双眼中心、鼻尖、左右嘴角。
4. 使用 `cv2.estimateAffinePartial2D` 对齐到 112x112 ArcFace 模板。
5. 输出 landmark overlay、aligned face、before/after grid。

## 7. 当前状态

当前仓库已补齐 4.x 代码、配置和报告入口，并已完成本次 GPU 全量训练、评估和对齐可视化。

本次数据准备使用 Kaggle `ibug_300W_large_face_landmark_dataset` 已解压目录。train、valid、valid_common、valid_challenge 均通过完整图片校验；该 Kaggle 包缺少官方 `Test/` 图片，因此 official test split 在 summary 中标记为 skipped。

本次训练使用 RTX 4060 Laptop GPU、batch size 32，完成 300W train 全量 60 epoch。最优验证 NME 为 `0.0342045356`，固定权重路径为 `work_dirs/task4/hrnetv2_w18_300w_full/best.pth`。

本次评估结果：valid/full NME `0.0342045356`，common NME `0.0290786345`，challenge NME `0.0552407292`。已生成 loss 曲线、NME 图、8 组 landmark overlay / aligned face / before-after grid。
