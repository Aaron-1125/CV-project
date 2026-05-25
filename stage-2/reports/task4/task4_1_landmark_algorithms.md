# Stage2 Task 4.1 人脸关键点检测算法调研

## 任务目标

任务 4.x 从人脸检测推进到人脸关键点检测与对齐。本阶段重点不再是预测人脸框，而是在人脸框内定位稳定的语义点，例如眼角、鼻尖、嘴角和脸部轮廓点，并基于这些点完成标准化对齐。

本项目采用 300W 的 68 点标注体系，并使用 MMPose HRNetv2-W18 作为训练 baseline。

## 常用算法路线

| 路线 | 代表方法 | 核心思想 | 优点 | 局限 |
| --- | --- | --- | --- | --- |
| 级联回归 | ESR、SDM、LBF | 从初始形状开始逐步回归关键点偏移 | 速度快，传统工程实现简单 | 对大姿态、遮挡和复杂光照更敏感 |
| 热图回归 | Hourglass、CPM、HRNet | 为每个关键点预测一张 heatmap，再取峰值坐标 | 精度高，空间表达稳定 | 训练和推理成本高于传统回归 |
| 坐标回归 | DeepPose、MobileNet 回归头 | 直接输出关键点坐标 | 模型轻、端侧友好 | 精细定位通常弱于热图方法 |
| 遮挡鲁棒方法 | SAN、LAB、Wing Loss 系列 | 引入边界、注意力或鲁棒损失处理遮挡和模糊 | 适合 in-the-wild 场景 | 实现和调参复杂度更高 |

## HRNet 选择理由

HRNet 的关键优势是始终保留高分辨率特征，并在多个分辨率分支之间反复融合。相比先下采样再上采样的结构，它对眼角、嘴角这类小范围几何细节更友好。

本项目选择 HRNetv2-W18：

- 与项目现有 OpenMMLab 生态一致，MMPose 官方支持 300W。
- 计算量低于 W32/W48，更适合 RTX 4060 8GB 的本地训练。
- 官方配置使用 256x256 输入、68 点 heatmap、NME 指标，和任务 4.2/4.3 完整对齐。

## 数据集与指标

300W 由 AFW、HELEN、LFPW、IBUG 和 300W test 组成，采用 68 点人脸关键点标注。MMPose 官方要求目录结构为：

```text
data/task4_300w/mmpose/300w/
  annotations/
    face_landmarks_300w_train.json
    face_landmarks_300w_valid.json
    face_landmarks_300w_valid_common.json
    face_landmarks_300w_valid_challenge.json
    face_landmarks_300w_test.json
  images/
    afw/
    helen/
    ibug/
    lfpw/
    Test/
```

评估指标采用 NME，即关键点平均欧氏误差除以眼间距。NME 越低，关键点定位越准。300W 官方也使用基于眼间距归一化的平均点到点误差进行评估。

## 与任务 4.3 的关系

人脸对齐使用关键点预测结果生成仿射变换。实现中从 68 点中取 5 个稳定点：

- 左眼中心：36-41 点均值
- 右眼中心：42-47 点均值
- 鼻尖：30 点
- 左嘴角：48 点
- 右嘴角：54 点

这 5 点会对齐到 112x112 ArcFace 模板，用于生成标准化人脸图。

## 参考资料

- MMPose 1.x installation: https://mmpose.readthedocs.io/en/dev-1.x/installation.html
- MMPose 2D face keypoint datasets: https://mmpose.readthedocs.io/en/dev-1.x/dataset_zoo/2d_face_keypoint.html
- MMPose training/testing: https://mmpose.readthedocs.io/en/dev-1.x/user_guides/train_and_test.html
- iBUG 300W dataset: https://ibug.doc.ic.ac.uk/resources/300-W/
