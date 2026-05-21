# 阶段一：人脸识别基础与数据集探索报告

## 1. 人脸识别基础概念

人脸识别系统通常不是一个模型直接从原图输出身份，而是一条分阶段 pipeline：

```text
image / video
  -> face detection
  -> landmark detection
  -> alignment
  -> embedding extraction
  -> verification / identification
```

- 人脸检测：找出图像中每张脸的位置，输出 `bbox=[x1, y1, x2, y2]`。
- 关键点定位：定位眼睛、鼻尖、嘴角等稳定几何点，常见输出是 5 点、68 点或 106 点。
- 人脸对齐：根据关键点裁剪、旋转和缩放人脸，让后续识别模型看到更标准的输入。
- 人脸验证：`1:1` 判断两张脸是否属于同一个人。
- 人脸识别/检索：`1:N` 在人脸库中寻找最相似身份。

本阶段的工程目标是先跑通检测、关键点、embedding 和 LFW 验证闭环，不训练自定义识别模型。

## 2. 数据集探索

运行脚本：

```bash
python code/stage1_task2_2_dataset_exploration.py --download --data-dir data --report-dir reports
```

输出文件：

- `reports/stage1_task2_2_dataset_summary.json`
- `reports/stage1_task2_2_dataset_summary.md`
- `reports/assets/celeba_samples.png`
- `reports/assets/celeba_attribute_top15.png`
- `reports/assets/celeba_identity_top15.png`
- `reports/assets/lfw_samples.png`
- `reports/assets/lfw_identity_top15.png`

预期统计内容：

- CelebA：图像数量、身份数量、40 个属性分布、bbox/landmark 标注形状、样本图。
- LFW：人物数量、身份分布、10-fold verification pairs 的正负样本分布、样本图。

本次运行结果：

- CelebA：`202599` 张图像，`10177` 个身份，`40` 个属性；数据源为 Hugging Face `eurecom-ds/celeba` 的 `train+validation+test`。
- CelebA 属性正样本比例最高的属性包括 `No_Beard=0.8349`、`Young=0.7736`、`Attractive=0.5125`、`Mouth_Slightly_Open=0.4834`、`Smiling=0.4821`。
- LFW：`13233` 张图像，`5749` 个身份，`6000` 对 10-fold verification pairs，其中 same/different 各 `3000` 对。
- LFW 频次最高身份包括 `George W Bush=530`、`Colin Powell=236`、`Tony Blair=144`、`Donald Rumsfeld=121`、`Gerhard Schroeder=109`。

## 3. MMDetection 人脸检测

运行脚本：

```bash
python code/stage1_task2_3_mmdet_face_detection.py \
  --input-dir ../../../sample_inputs \
  --out-dir reports/assets \
  --checkpoint-dir checkpoints/mmdet \
  --texts "face . human face ."
```

实现选择：

- 框架：MMDetection。
- 模型：OpenMMLab GroundingDINO。
- 提示词：`face . human face .`
- 原因：官方 MMDetection 有 WIDER FACE 配置，但没有稳定随包下发的人脸检测 checkpoint；开放词检测能保持 MMDetection 框架约束，同时可复现地对人脸提示词输出 bbox。

输出文件：

- `reports/assets/mmdet_face_detection_summary.json`
- `reports/assets/mmdet_<image>_faces.jpg`

本次运行结果：

- `01036a162ec6e859bb81218ad79dc1aa.jpg`：检测到 `2` 张脸。
- `10ad277043b7ee0e9e185bebf7402495.jpg`：检测到 `1` 张脸。
- 脚本对 `face` / `human face` 双提示词产生的重复框做 IoU NMS 去重，保留最高分 bbox。

## 4. 关键点定位与 LFW 验证

运行脚本：

```bash
python code/stage1_task2_4_landmarks_and_lfw_eval.py \
  --download \
  --data-dir data \
  --out-dir reports/assets \
  --landmark-input-dir ../../../sample_inputs
```

实现选择：

- 模型：InsightFace `buffalo_l` 预训练模型。
- 关键点：优先绘制可用的 106 点关键点，同时标出 5 点关键点。
- 验证：对 LFW 10-fold pairs 提取归一化 embedding，使用 cosine similarity，逐折在训练折选择阈值，在验证折计算准确率。

输出文件：

- `reports/assets/lfw_insightface_verification_summary.json`
- `reports/assets/lfw_similarity_histogram.png`
- `reports/assets/lfw_roc_curve.png`
- `reports/assets/*_landmarks.jpg`

本次运行结果：

- LFW 10-fold pairs：`6000/6000` 对有效，失败对数 `0`。
- embedding 模式：直接使用 `buffalo_l` 内的 `w600k_r50.onnx` recognition 模型处理 LFW crop；关键点可视化仍使用 `FaceAnalysis`。
- 10-fold mean accuracy：`0.9665`，std：`0.009042`。
- AUC：`0.989726`。
- 关键点可视化：样例图分别检测到 `2` 张脸和 `1` 张脸，并输出 5 点/106 点关键点叠加图。

## 5. 当前验收状态

代码已完成阶段一任务 2.1-2.4 的交付接口与实际运行：

- 任务 2.1：报告第 1 节完成人脸检测、对齐、识别、验证/检索概念梳理。
- 任务 2.2：`stage1_task2_2_dataset_exploration.py` 已完成 CelebA/LFW 下载、统计和可视化。
- 任务 2.3：`stage1_task2_3_mmdet_face_detection.py` 已完成 MMDetection 开放词人脸检测。
- 任务 2.4：`stage1_task2_4_landmarks_and_lfw_eval.py` 已完成关键点定位和 LFW 验证准确率。

大文件策略：

- `data/`：原始数据集和 embedding cache，不提交 Git。
- `checkpoints/`：MMDetection/InsightFace 等模型权重，不提交 Git。
- `reports/`：提交可读报告、JSON 摘要和小体积可视化结果。
