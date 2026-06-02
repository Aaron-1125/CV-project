# Task6 Final InsightFace R50 Latency Report

## Source

- Checkpoint: `work_dirs/task5/insightface_ms1mv3_r50_full/model.pt`
- Accepted cloud 112x112 LFW accuracy: `0.9980`
- Target LFW accuracy: `0.9850`
- Target met: `True`
- Local latency-bin LFW accuracy: `0.8605`

The local LFW-bin score is kept only as a same-input comparison across PyTorch/ONNX backends during Task6 latency testing. The Task5 acceptance metric remains the cloud 112x112 LFW result.

## Full LFW End-to-End Benchmark

| Backend | Local-bin accuracy | Latency ms/image | Throughput img/s | Speedup vs PyTorch FP32 | Size MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| pytorch_fp32_cuda | 0.8605 | 35.895 | 27.86 | 1.000 | 0.00 |
| pytorch_fp16_cuda | 0.8615 | 7.607 | 131.45 | 4.719 | 0.00 |
| onnx_fp32_cuda | 0.8605 | 13.107 | 76.29 | 2.739 | 166.32 |
| onnx_fp16_cuda | 0.8582 | 9.078 | 110.15 | 3.954 | 83.18 |

## Model-Only Latency

| Backend | Batch | Latency ms/image | Throughput img/s | Note |
| --- | ---: | ---: | ---: | --- |
| pytorch_fp32_cuda | 1 | 9.5128 | 105.12 |  |
| pytorch_fp16_cuda | 1 | 10.6624 | 93.79 |  |
| onnx_fp32_cuda | 1 | 6.0597 | 165.02 |  |
| onnx_fp16_cuda | 1 | 7.3675 | 135.73 |  |
| pytorch_fp32_cuda | 16 | 2.4548 | 407.37 |  |
| pytorch_fp16_cuda | 16 | 1.4530 | 688.24 |  |
| onnx_fp32_cuda | 16 | 2.5916 | 385.86 |  |
| onnx_fp16_cuda | 16 | 1.4761 | 677.44 |  |
| pytorch_fp32_cuda | 64 | 2.3363 | 428.03 |  |
| pytorch_fp16_cuda | 64 | 1.3313 | 751.12 |  |
| onnx_fp32_cuda | 64 | 2.4959 | 400.65 |  |
| onnx_fp16_cuda | 64 | 1.4036 | 712.45 |  |
| pytorch_fp32_cuda | 256 | 9.8744 | 101.27 |  |
| pytorch_fp16_cuda | 256 | 7.5717 | 132.07 |  |
| onnx_fp32_cuda | 256 | skipped | - | Skipped by default on 8GB laptop GPUs to avoid cuDNN algorithm-search failures; set --onnx-cuda-max-batch >= 256 to force it. |
| onnx_fp16_cuda | 256 | skipped | - | Skipped by default on 8GB laptop GPUs to avoid cuDNN algorithm-search failures; set --onnx-cuda-max-batch >= 256 to force it. |

## Consistency

| Backend | Mean Cosine vs PyTorch | Max Abs Diff |
| --- | ---: | ---: |
| onnx_fp32_cuda | 1.000000 | 0.000174 |
| onnx_fp16_cuda | 0.999994 | 0.000703 |

## Conclusion

- Latency improved: PyTorch FP16 CUDA is the fastest full local-bin path in this run, about `4.72x` faster than PyTorch FP32 CUDA.
- ONNX FP32 CUDA is also deployable and about `2.74x` faster than PyTorch FP32 CUDA; ONNX FP16 CUDA is about `3.95x` faster and halves the ONNX model size from about `166.32 MB` to `83.18 MB`.
- Dynamic quantization remains only a CPU Linear-layer control; it is not suitable as the main acceleration path for Conv2d-heavy ArcFace R50.
- The accepted source model remains the cloud 112x112 LFW `0.998` checkpoint; Task6 focuses on deployment latency and backend consistency.
