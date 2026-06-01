# Stage2 Task6 人脸识别模型压缩与低时延部署报告

## 1. 任务目标

Task6 面向 Task5 人脸识别模型做模型压缩与部署优化。当前最终源模型已经切换为官方 InsightFace full-MS1MV3 ResNet50 + ArcFace checkpoint：

- Final checkpoint: `work_dirs/task5/insightface_ms1mv3_r50_full/model.pt`
- Final LFW accuracy: `0.998`
- Final rerun outputs: `reports/task6/final/`
- Final ONNX artifacts: `work_dirs/task6/final_insightface_r50/`

早期自研 800k dense 模型的 `81.67%` LFW 结果只保留为历史 baseline，不再作为 Task6 最终源模型。

## 2. 已完成的历史 baseline

历史 baseline 使用自研 IResNet50 + ArcFace checkpoint，完成了动态量化和 ONNX Runtime CPU 对比：

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

它只量化 `Linear` 层，而 ArcFace R50/IResNet50 的主要计算来自 `Conv2d + BatchNorm/PReLU` 主干。卷积主干仍然以 FP32 执行，所以最耗时部分没有变化。LFW benchmark 还包含图片解码、resize、normalize、batch 组装、embedding 归一化和 pair protocol 统计，这些非模型开销进一步稀释了 Linear-only INT8 的收益。

因此，导师指出“模型压缩时延好像没有下降”的结论成立；根因不是实现失败，而是压缩策略与卷积主导模型结构不匹配。动态量化在本任务中应作为体积压缩 baseline，而不是主要加速方案。

## 4. Final 低时延路线

Final Task6 改为基于 LFW 达标的 official InsightFace R50 checkpoint 重跑：

- PyTorch FP32 CUDA baseline
- PyTorch FP16 CUDA baseline
- ONNX FP32 export + ONNX Runtime CPU/CUDA
- ONNX FP16 export + ONNX Runtime CUDA
- Dynamic INT8 Linear-only CPU control

运行入口：

```bash
docker compose run --rm -w /workspace/stage-2 stage2-gpu \
  python code/task6/stage2_task6_5_final_insightface_latency.py \
    --config configs/task5_arcface/insightface_ms1mv3_r50_full_gpu.py \
    --checkpoint work_dirs/task5/insightface_ms1mv3_r50_full/model.pt \
    --device cuda:0 \
    --providers cuda,cpu \
    --summary-out reports/task6/final/summaries/final_latency_summary.json \
    --report-out reports/task6/final/stage2_task6_final_latency_report.md \
    --plot-out reports/task6/final/assets/evaluation/final_latency_comparison.png
```

该脚本会导出 FP32/FP16 ONNX，验证 ONNX 与 PyTorch embedding 的 cosine consistency，并记录 batch `1/16/64/256` 的 model-only latency 和完整 LFW end-to-end latency。

## 5. 预期结论

如果 ONNX Runtime CUDA/FP16 快于 PyTorch CUDA，最终交付结论为：动态量化不适合 ArcFace R50 的时延优化，ONNX GPU/FP16 是更合理的部署加速路径。

如果 ONNX Runtime CUDA/FP16 在 RTX 4060 上提升不明显，最终交付结论为：模型压缩链路正确，动态量化保持精度和体积收益，但本机小 batch、图片预处理和 runtime provider 限制了端到端时延收益；后续若要继续压低延迟，应使用 TensorRT FP16/INT8、Conv2d 静态量化/QAT，或蒸馏到 MobileFaceNet。
