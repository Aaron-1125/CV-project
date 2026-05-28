# Task 6.1 模型优化技术调研

## 动态量化

动态量化在推理时把部分权重转换为 int8，并在运行过程中动态处理激活值。它适合 `Linear`、RNN 等矩阵乘法占比较高的模块，部署简单，不需要重新训练或额外校准数据。Task6 使用 PyTorch `quantize_dynamic` 对 ArcFace backbone 中的 `Linear` 层进行动态量化。由于 IResNet50 的主要计算来自卷积层，动态量化的体积和速度收益有限，但它可以作为最小侵入的压缩 baseline。

## 剪枝

剪枝通过删除冗余通道、卷积核或权重连接来减少计算量。非结构化剪枝能制造稀疏权重，但通用硬件未必能直接获得速度收益；结构化通道剪枝更适合端侧部署，但通常需要剪枝后微调来恢复精度。人脸识别模型剪枝时要重点监控 embedding 角度分布和 LFW/业务验证集准确率，避免压缩后类间间隔变小。

## 蒸馏

知识蒸馏使用高精度 teacher 模型指导较小 student 模型学习。人脸识别中常见做法是让 student 同时学习分类损失、ArcFace margin 约束，以及 teacher embedding 的余弦相似度或特征距离。蒸馏通常比单纯剪枝更稳，但需要额外 teacher checkpoint 和重新训练成本。

## ONNX 部署

ONNX 将 PyTorch 模型导出为跨框架计算图，便于使用 ONNX Runtime 做 CPU/GPU 推理，也方便后续接入 TensorRT、OpenVINO 等推理引擎。Task6 导出的是 ArcFace embedding backbone，输入为 `N x 3 x 112 x 112`，输出为 `N x 512` 归一化 embedding。
