# Stage2 Task6 人脸识别模型压缩与低时延部署报告

## 1. 任务目标

Task6 面向 Task5 人脸识别模型做模型压缩与部署优化，重点回应导师提出的“模型压缩后时延没有下降”的问题。

本轮重新基于官方 InsightFace R50/ArcFace checkpoint 运行低时延复测：

- Checkpoint: `work_dirs/task5/insightface_ms1mv3_r50_full/model.pt`
- Accepted cloud 112x112 LFW accuracy: `0.998`
- Final Task6 outputs: `reports/task6/final/`
- Final ONNX artifacts: `work_dirs/task6/final_insightface_r50/`

说明：Task6 本机复测还使用了一个本地 LFW bin 来做 PyTorch/ONNX 同输入 latency 对比。这个本地 bin 的 accuracy 只用于比较不同部署后端是否保持一致，不作为 Task5 的验收精度；Task5 验收仍以云端 112x112 LFW 的 `0.998` 为准。

## 2. 历史 Baseline

早期自研 IResNet50 + ArcFace checkpoint 完成过动态量化和 ONNX Runtime CPU 对比：

| Backend | LFW accuracy | ROC AUC | latency ms/image | throughput img/s | model size MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 PyTorch | 81.68% | 0.8791 | 62.408 | 16.02 | 166.58 |
| Dynamic INT8 PyTorch | 81.68% | 0.8790 | 62.935 | 15.89 | 129.86 |
| ONNX Runtime CPU | 81.68% | 0.8791 | 44.860 | 22.29 | 166.32 |

这个结果说明动态量化可以减小模型体积，但没有降低端到端时延；ONNX Runtime CPU 通过推理图和 kernel 优化获得了约 `1.42x` 加速。

## 3. 时延未下降原因

历史动态量化使用：

```python
torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

它只量化 `Linear` 层，而 ArcFace R50/IResNet50 的主要计算来自 `Conv2d + BatchNorm/PReLU` 主干。卷积主干仍然以 FP32 执行，所以最耗时的部分没有变化。

此外，LFW benchmark 还包含图片解码、resize、normalize、batch 组装、embedding 归一化和 pair protocol 统计，这些非模型开销进一步稀释了 Linear-only INT8 的收益。

因此导师指出“模型压缩时延好像没有下降”的判断成立；根因不是量化实现失败，而是压缩策略与卷积主导模型结构不匹配。动态量化在本任务中应作为体积压缩 baseline，而不是主要加速方案。

## 4. Final 低时延复测

本轮改用更合适的部署路线：

- PyTorch FP32 CUDA baseline
- PyTorch FP16 CUDA
- ONNX FP32 export + ONNX Runtime CUDA
- ONNX FP16 export + ONNX Runtime CUDA
- Dynamic INT8 Linear-only CPU control

完整结果见：

- `reports/task6/final/summaries/final_latency_summary.json`
- `reports/task6/final/stage2_task6_final_latency_report.md`
- `reports/task6/final/assets/evaluation/final_latency_comparison.png`

复测摘要：

| Backend | Local-bin accuracy | latency ms/image | speedup vs PyTorch FP32 |
| --- | ---: | ---: | ---: |
| PyTorch FP32 CUDA | 0.8605 | 35.895 | 1.00x |
| PyTorch FP16 CUDA | 0.8615 | 7.607 | 4.72x |
| ONNX FP32 CUDA | 0.8605 | 13.107 | 2.74x |
| ONNX FP16 CUDA | 0.8582 | 9.078 | 3.95x |

ONNX 一致性检查正常：

- ONNX FP32 mean cosine vs PyTorch: `0.9999996`
- ONNX FP16 mean cosine vs PyTorch: `0.9999937`

## 5. 结论

时延问题已经闭环：动态量化没有降时延的原因是 Linear-only INT8 不覆盖卷积主干；改用 GPU FP16/ONNX Runtime 后，实测端到端时延明显下降，其中 PyTorch FP16 CUDA 是本轮最快路径，约 `4.72x`。

最终交付口径为：Task5 源模型采用云端 112x112 LFW `0.998` 作为验收精度；Task6 本机 local-bin accuracy 只用于同输入部署后端对比。压缩后精度在各后端之间保持接近，ONNX FP16 文件体积约减半，GPU FP16/ONNX 路线解决了“压缩后时延未下降”的问题。
