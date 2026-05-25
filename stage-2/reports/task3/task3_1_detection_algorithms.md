# 任务 3.1：MTCNN、RetinaFace 与人脸检测算法原理

## 1. 学习目标

任务 3.x 的核心是从“调用预训练检测器”推进到“理解并训练检测模型”。任务 3.1 先梳理人脸检测模型的基本路线，为任务 3.2 使用 MMDetection 在 WIDER FACE 上训练 baseline 做准备。

本阶段重点关注：

- 多尺度人脸检测：真实图片中人脸尺度变化大，小脸、侧脸、遮挡脸会显著增加检测难度。
- 级联检测到单阶段密集检测的演进：MTCNN 代表级联粗到细流程，RetinaFace/SSD/FPN 代表更现代的密集预测思路。
- 检测与关键点的关系：关键点不是任务 3.x 的交付重点，但 RetinaFace 说明 landmark 监督能反过来改善定位质量。
- 评估指标：检测模型用 precision、recall、AP/mAP 衡量，不用分类 accuracy 作为主指标。

## 2. MTCNN

MTCNN 是经典级联式人脸检测与关键点定位方法，由三个阶段组成：

| 阶段 | 作用 | 输出 |
| --- | --- | --- |
| P-Net | 在图像金字塔上快速生成候选框 | 候选 bbox、粗分类分数 |
| R-Net | 对候选框二次筛选和回归 | 更准确 bbox、分类分数 |
| O-Net | 精修 bbox，并回归 5 点关键点 | 最终 bbox、landmark |

MTCNN 的优点是流程直观，适合理解“候选框生成 -> NMS -> bbox 回归 -> 关键点定位”的检测 pipeline。局限是级联流程推理链路较长，对密集小脸和复杂遮挡场景不如现代单阶段检测器稳定。

## 3. RetinaFace

RetinaFace 是 single-stage dense face localization 方法。它在密集 anchor/feature locations 上同时预测：

- face classification：当前位置是否有人脸。
- bbox regression：人脸框位置。
- landmark regression：常见 5 点关键点。
- dense face localization supervision：论文中用于增强复杂场景定位质量的额外监督。

RetinaFace 的关键启发是：人脸检测不只是找矩形框，眼睛、鼻子、嘴角等结构信息会帮助模型在遮挡、侧脸和小脸场景下做出更稳的定位。它也说明了任务 3.x 与后续任务 4.x 的连接：检测框质量会影响关键点、对齐和后续识别。

## 4. SSD/FPN 与 MMDetection Baseline

本阶段训练交付选择 MMDetection 官方 WIDER FACE SSD300 baseline，原因如下：

- PDF 任务要求使用 MMDetection 训练人脸检测模型。
- MMDetection 官方已有 WIDER FACE 配置入口，数据集类型为 `WIDERFaceDataset`。
- SSD300 是单阶段检测器，训练链路比 RetinaFace 完整复现更轻，适合本机 smoke training。
- 通过 full config 保留完整 WIDER FACE 训练命令，后续可以替换为 RetinaNet/FPN 或 RetinaFace/SCRFD 类更强模型。

SSD 的核心思想是在多个 feature map 位置直接预测类别和 bbox。它不像 MTCNN 那样多阶段筛选，而是通过 dense anchors 和 NMS 得到最终检测框。对于 WIDER FACE 这类多尺度数据，后续更强版本通常会引入 FPN 或专门的人脸检测 head 来提升小脸表现。

## 5. WIDER FACE 难点

WIDER FACE 是自然场景人脸检测 benchmark，难点包括：

- 人脸尺度跨度大：大头照和远处小脸同时存在。
- 姿态变化大：正脸、侧脸、低头、仰头都有。
- 遮挡多：手、口罩、头发、其他人脸遮挡。
- 密集人群：一张图中可能有大量小脸，NMS 和召回都更难。
- 光照与模糊：运动模糊、低光、强反差会降低定位质量。

因此任务 3.3 需要使用 mAP、precision、recall 和可视化案例一起判断模型效果。smoke training 的指标只用于验证训练/评估链路跑通，不能代表全量训练模型的最终能力。

## 6. 本任务结论

阶段二任务 3.x 的合理路线是：

```text
MTCNN / RetinaFace 原理学习
-> WIDER FACE 数据准备
-> MMDetection SSD300 smoke training
-> smoke validation mAP / precision / recall
-> WIDER FACE 公开验证图检测可视化
-> 保留 full config 供后续完整训练
```

本阶段先把训练工程链路跑通；模型能力提升放到后续完整训练、更多 epoch、更强 backbone/FPN 或更专业的人脸检测器中继续优化。
