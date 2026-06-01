# Task6 模型压缩时延诊断

## 问题

Task6 的动态量化模型体积下降了，但端到端时延没有下降。这个现象不是偶然波动，而是当前压缩方法与 ArcFace/IResNet50 结构不匹配导致的。

## 现有结果证据

| 后端 | LFW accuracy | latency ms/image | throughput img/s | model size MB |
| --- | ---: | ---: | ---: | ---: |
| FP32 PyTorch | 81.68% | 62.408 | 16.02 | 166.58 |
| Dynamic INT8 PyTorch | 81.68% | 62.935 | 15.89 | 129.86 |
| ONNX Runtime CPU | 81.68% | 44.860 | 22.29 | 166.32 |

动态量化把模型体积降到 FP32 的约 `78.0%`，但 latency speedup 只有 `0.99x`，也就是没有真实加速。ONNX Runtime CPU 的 speedup 约为 `1.42x`，说明更有效的是推理图和 runtime kernel 优化，而不是当前的 Linear-only 动态量化。

## 根因

Task6 当前动态量化使用的是：

```python
torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

这只会量化 `Linear` 层。ArcFace 的 IResNet50/ResNet50 backbone 主要计算量来自大量 `Conv2d + BatchNorm/PReLU` 特征提取块，最耗时的卷积主干仍然以 FP32 执行。末端全连接层虽然被压缩成 INT8，但它不是整体推理时延的瓶颈，所以无法显著降低总耗时。

还有几个叠加因素：

- LFW benchmark 包含 PIL 读图、resize、normalize、DataLoader、embedding 归一化和 pair protocol 统计，这些非模型开销会稀释模型压缩收益。
- 动态 INT8 `Linear` 在 CPU 上可能引入量化/反量化和调度开销。
- PyTorch dynamic quantization 没有启用 `Conv2d` 的 INT8 kernel path。
- 112x112 小图批处理时，CPU 线程调度和内存带宽也会影响延迟。
- ONNX Runtime 变快是因为图优化和 CPU kernel 优化，不是因为模型权重被压缩。

## 结论

当前 Task6 结果仍然有交付价值：LFW accuracy 保持不变，模型体积下降，ONNX 导出数值一致且速度提升。但它不能证明“动态量化能加速这个人脸识别模型”。更准确的结论是：对卷积占主导的 ArcFace backbone，只量化 `Linear` 层不是有效的时延优化方法。

## 后续改进路线

如果目标是实际降低时延，优先建议：

- 使用 ONNX Runtime / TensorRT FP16 做 GPU 推理部署。
- 使用带 calibration 的静态 INT8 或 QAT，把 `Conv2d` 也纳入量化。
- 做结构化 channel pruning，并在 LFW/MS1MV3 上微调恢复精度。
- 将最终 Task5 ArcFace 模型蒸馏到 MobileFaceNet 等轻量 backbone。

如果 Task5 已经换成 LFW 达标的新 checkpoint，Task6 最终版需要重新基于该 checkpoint 跑量化与 ONNX；上述时延诊断逻辑仍然成立，但最终报告不应继续引用旧的 `81.67%` baseline 作为主结果。
