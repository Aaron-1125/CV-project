# Task6 Final InsightFace R50 Latency Report

## Source

- Checkpoint: `work_dirs/task5/insightface_ms1mv3_r50_full/model.pt`
- Source LFW accuracy: `0.998`
- Device: `cuda:0`
- ONNX Runtime providers available: `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']`

## Full LFW End-to-End Benchmark

| Backend | Accuracy | Latency ms/image | Throughput img/s | Size MB |
| --- | ---: | ---: | ---: | ---: |

## Model-Only Latency

| Backend | Batch | Latency ms/image | Throughput img/s |
| --- | ---: | ---: | ---: |
| pytorch_fp32_cuda | 1 | 6.2720 | 159.44 |
| pytorch_fp16_cuda | 1 | 7.8207 | 127.87 |
| onnx_fp32_cuda | 1 | 4.5532 | 219.63 |
| onnx_fp32_cpu | 1 | 43.6511 | 22.91 |
| onnx_fp16_cuda | 1 | 4.8353 | 206.81 |

## Conclusion

ONNX Runtime GPU/FP16 is the preferred deployment path if it beats the PyTorch CUDA baseline. Dynamic quantization is retained only as a CPU Linear-layer control because it does not target Conv2d.

Dynamic quantization remains a CPU Linear-layer control and is not the main acceleration route for this convolution-heavy ArcFace R50 model.
