# Task6 Final Latency Execution Notes

## Purpose

This final Task6 rerun uses the official InsightFace R50 checkpoint:

- checkpoint: `work_dirs/task5/insightface_ms1mv3_r50_full/model.pt`
- accepted cloud 112x112 LFW accuracy: `0.998`
- local latency-bin LFW accuracy: about `0.8605`
- final outputs: `reports/task6/final/`
- ONNX artifacts: `work_dirs/task6/final_insightface_r50/`

The old `81.67%` Task5 custom-trainer checkpoint remains a historical baseline
only. The local Task6 LFW bin is used for same-input backend comparison during
latency testing; the accepted Task5 validation metric is the cloud 112x112 LFW
result.

## Local Docker Command

Run from the project root:

```bash
docker compose build stage2-gpu
docker compose run --rm -w /workspace stage2-gpu python docker/verify_environment.py

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

## Expected Result

The script benchmarks:

- PyTorch FP32 CUDA
- PyTorch FP16 CUDA
- ONNX Runtime FP32 CPU
- ONNX Runtime FP32 CUDA
- ONNX Runtime FP16 CUDA
- PyTorch dynamic INT8 Linear-only CPU control

Success for Task6 latency means at least one deployable GPU/ONNX path reduces
latency against the PyTorch CUDA baseline. Accuracy preservation in this script
is measured on the local latency bin so all backends see exactly the same input
pairs.

For RTX 4060 Laptop 8GB, ONNX CUDA benchmarking defaults to max batch `64`
while PyTorch still reports batch `256`; this avoids unstable cuDNN algorithm
search at very large ONNX CUDA batches. To force larger ONNX CUDA batches, pass
`--onnx-cuda-max-batch 256 --onnx-cuda-lfw-batch-size 256` after rebooting the
GPU.

If `CUDAExecutionProvider` is missing, rebuild `stage2-gpu`; the GPU Dockerfile
replaces CPU `onnxruntime` with CUDA-11 `onnxruntime-gpu`.
