# Task6 Final Latency Execution Notes

## Purpose

This final Task6 rerun uses the LFW-qualified official InsightFace R50 checkpoint:

- checkpoint: `work_dirs/task5/insightface_ms1mv3_r50_full/model.pt`
- source LFW accuracy: `0.998`
- final outputs: `reports/task6/final/`
- ONNX artifacts: `work_dirs/task6/final_insightface_r50/`

The old `81.67%` Task5 custom-trainer checkpoint remains a historical baseline only.

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

Success means at least one deployable ONNX GPU path reduces latency against the
PyTorch CUDA baseline while keeping LFW accuracy close to the source model.

For RTX 4060 Laptop 8GB, ONNX CUDA benchmarking defaults to max batch `64`
while PyTorch still reports batch `256`; this avoids unstable cuDNN algorithm
search at very large ONNX CUDA batches. To force larger ONNX CUDA batches, pass
`--onnx-cuda-max-batch 256 --onnx-cuda-lfw-batch-size 256` after rebooting the
GPU.

If `CUDAExecutionProvider` is missing, rebuild `stage2-gpu`; the GPU Dockerfile
replaces CPU `onnxruntime` with CUDA-11 `onnxruntime-gpu`.
