#!/usr/bin/env python3
"""Write the Stage3 Task8 3D face reconstruction Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from stage3_task8_common import (
    cfg_get,
    env_summary_path,
    load_config,
    prepare_summary_path,
    read_json,
    reconstruction_summary_path,
    relpath_for_markdown,
    render_summary_path,
    report_path,
    summary_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task8_3dface/a800_3ddfa_v2.py")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def maybe_json(path: Path) -> Dict[str, Any]:
    return read_json(path) if path.exists() else {}


def md_path(value: Any, output: Path) -> str:
    return relpath_for_markdown(value, output) if value else ""


def image_line(label: str, value: Any, output: Path) -> str:
    rel = md_path(value, output)
    if not rel:
        return "- {}: N/A".format(label)
    return "- {}: ![{}]({})".format(label, label, rel)


def sample_section(record: Dict[str, Any], render_records: Dict[str, Dict[str, Any]], output: Path) -> str:
    sample_id = record.get("sample_id", "sample")
    render = render_records.get(str(sample_id), {})
    grid = render.get("multiview_grid") if render.get("available") else None
    render_note = "available" if render.get("available") else "unavailable: {}".format(render.get("failure_reason", "not generated"))
    lines = [
        "### {}".format(sample_id),
        "",
        image_line("Input", record.get("input_image"), output)
        if md_path(record.get("input_image"), output)
        else "- Input: original CelebA symlink omitted from local deliverable; see input grid and prepare summary.",
        image_line("2D sparse landmarks", record.get("landmark_path"), output),
        image_line("3D overlay", record.get("overlay_path"), output),
        image_line("Pose", record.get("pose_path"), output),
        "- OBJ mesh: `{}`".format(md_path(record.get("obj_path"), output) or "N/A"),
        "- Multi-view render: {}".format(render_note),
    ]
    if grid:
        lines.append("")
        lines.append("![{} multiview]({})".format(sample_id, md_path(grid, output)))
    return "\n".join(lines)


def pct(value: Any) -> str:
    try:
        return "{:.2f}%".format(float(value) * 100.0)
    except Exception:
        return "N/A"


def count_success(records: Any) -> int:
    return sum(1 for row in records or [] if row.get("success"))


def count_failure(records: Any) -> int:
    return sum(1 for row in records or [] if row.get("status") == "failed" or row.get("success") is False)


def select_showcase_records(recon: Dict[str, Any], render: Dict[str, Any], max_samples: int) -> list:
    records_by_id = {str(row.get("sample_id")): row for row in recon.get("records", [])}
    selected = []
    used = set()
    for render_row in render.get("records", []):
        if not render_row.get("available"):
            continue
        sample_id = str(render_row.get("sample_id"))
        record = records_by_id.get(sample_id)
        if record and record.get("success"):
            selected.append(record)
            used.add(sample_id)
        if len(selected) >= max_samples:
            return selected
    for record in recon.get("records", []):
        sample_id = str(record.get("sample_id"))
        if record.get("success") and sample_id not in used:
            selected.append(record)
        if len(selected) >= max_samples:
            break
    return selected


def failure_examples(recon: Dict[str, Any], max_examples: int = 8) -> str:
    failed = recon.get("failed_records") or [
        row for row in recon.get("records", [])
        if row.get("status") == "failed" or row.get("success") is False
    ]
    if not failed:
        return "本次 summary 中暂无失败样本记录。"
    lines = []
    for row in failed[:max_examples]:
        reason = row.get("failure_reason") or "unknown"
        lines.append("- `{}`: {}".format(row.get("sample_id", "sample"), reason))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output = args.output or report_path(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = maybe_json(env_summary_path(cfg))
    prepare = maybe_json(prepare_summary_path(cfg))
    recon = maybe_json(reconstruction_summary_path(cfg))
    render = maybe_json(render_summary_path(cfg))
    render_records = {str(row.get("sample_id")): row for row in render.get("records", [])}
    max_showcase = int(cfg_get(cfg, "report", "max_showcase_samples", 12))
    show_failures = bool(cfg_get(cfg, "report", "show_failure_examples", True))
    showcase_records = select_showcase_records(recon, render, max_showcase)
    sample_sections = [
        sample_section(row, render_records, output)
        for row in showcase_records
    ]
    if not sample_sections:
        sample_sections = ["尚未生成重建结果。请先运行 `stage3_task8_run_reconstruction.py`。"]

    input_grid = md_path(prepare.get("input_grid"), output)
    input_grid_block = "![input samples]({})".format(input_grid) if input_grid else "N/A"
    repo = recon.get("repo") or env.get("repo") or {}
    requested_samples = prepare.get("requested_sample_count", cfg_get(cfg, "data", "sample_count", 500))
    actual_input_samples = prepare.get("actual_sample_count", prepare.get("sample_count", "N/A"))
    success_count = recon.get("success_count", count_success(recon.get("records", [])))
    failure_count = recon.get("failure_count", count_failure(recon.get("records", [])))
    success_rate = recon.get("success_rate", (float(success_count) / float(actual_input_samples) if isinstance(actual_input_samples, int) and actual_input_samples else None))
    rendered_count = render.get("rendered_count", sum(1 for row in render.get("records", []) if row.get("available")))
    outputs = recon.get("official_outputs", ["2d_sparse", "3d", "pose", "obj"])
    metrics_table = """| 指标 | 结果 |
|---|---:|
| Requested samples | {requested_samples} |
| Actual input samples | {actual_input_samples} |
| Successful reconstructions | {success_count} |
| Failed reconstructions | {failure_count} |
| Success rate | {success_rate} |
| Rendered showcase samples | {rendered_count} |
| Backend | {backend} |
| Mode | {mode} |
| Outputs | {outputs} |""".format(
        requested_samples=requested_samples,
        actual_input_samples=actual_input_samples,
        success_count=success_count,
        failure_count=failure_count,
        success_rate=pct(success_rate),
        rendered_count=rendered_count,
        backend=recon.get("selected_backend", "N/A"),
        mode=recon.get("selected_mode", "N/A"),
        outputs=" / ".join(outputs),
    )
    failure_block = failure_examples(recon) if show_failures else "报告配置未展示失败样本。"
    markdown = """# Stage3 Task8: 基于官方 3DDFA_V2 的 3D 人脸重建

## 1. 任务简介

本实验基于官方 3DDFA_V2 完成单图 3D 人脸重建，stage-3 只实现样本准备、官方推理调用、结果整理和可视化。本文工作重点是实验流程搭建、样本选择、结果可视化与分析，不是自研 3D 人脸重建模型，也没有重写 3DMM fitting、mesh reconstruction 或 landmark detection 的核心算法。

本次从 CelebA 中抽取 500 张测试图进行 3DDFA_V2 单图三维重建。500 张测试增强了流程稳定性验证，但该结果仍然是官方 3DDFA_V2 的 3DMM-based single-image reconstruction，不是高精度 3D 扫描。

交付内容包括：

- `code/task8/`: Task8 wrapper 脚本。
- `configs/task8_3dface/`: A800/3DDFA_V2 配置。
- `reports/task8/assets/`: 输入样本、官方重建输出和少量多角度可视化。
- `reports/task8/summaries/`: 环境、样本、重建、渲染 summary。

## 2. 基本原理

3DMM 是一种统计三维人脸模型，通常用平均脸、形状基、表情基和纹理信息表示人脸。单张图像重建的核心思想，是从 2D 人脸图像中估计人脸框、姿态、形状参数、表情参数和相机投影关系，再恢复可渲染的 3D mesh。由于单图缺少真实深度，重建结果依赖模型先验和人脸检测/对齐质量。

3DDFA_V2 是成熟的 3D Dense Face Alignment 框架。它使用 FaceBoxes/FaceBoxes_ONNX 做人脸检测，再用 TDDFA/TDDFA_ONNX 估计 3DMM 参数，最后通过官方 demo 支持的 `2d_sparse`、`3d`、`pose`、`obj` 等模式输出关键点、overlay、姿态和 mesh。NeRF 类方法通常通过多视角或隐式辐射场学习几何与外观，表达能力更强，但对数据和训练成本要求更高；本课程任务更适合使用 3DDFA_V2 快速、稳定地完成单图重建交付。

## 3. 实验设置

- 数据来源: `{source_dataset}`，请求样本数 `{requested_samples}`，实际输入 `{actual_input_samples}`。
- 输入样本目录: `{input_dir}`。
- 3DDFA_V2 repo: `{repo_path}`。
- 3DDFA_V2 commit: `{commit}`。
- 官方配置: `{official_config}`。
- 默认 runner: `{runner}`。
- 重建 backend: `{backend}`。
- 运行环境 Python: `{python_version}`。
- 抽样 seed: `{sample_seed}`。
- 抽样策略: `{sample_strategy}`。

输入样本预览只展示前若干张，不展示 500 张全集：

{input_grid_block}

## 4. 全量统计

{metrics_table}

## 5. 精选结果展示

{sample_sections}

## 6. 失败样本摘要

{failure_block}

## 7. 结果分析

正脸样本通常能得到较稳定的稠密人脸 mesh、68 点稀疏关键点和较自然的 3D overlay。轻微侧脸在 3DDFA_V2 的姿态建模范围内通常仍可恢复脸部整体几何，但被遮挡的一侧和边缘纹理会更依赖模型先验。遮挡、夸张表情、强光照、模糊或极端姿态会影响 FaceBoxes 检测和 TDDFA 参数估计，表现为关键点偏移、overlay 不贴合或 obj 局部纹理异常。

本 pipeline 默认优先复用官方 `demo.py` 的输出，因此结果应尽量接近官方 3DDFA_V2 demo。若 wrapper 收集结果与官方 demo 不一致，应优先检查输入 basename、`examples/results/` 扫描逻辑、backend 选择和官方依赖，而不是修改模型参数。

## 8. 局限性

- 单图 3D 重建存在深度歧义，侧脸背面和遮挡区域主要由模型先验补全。
- 结果依赖官方 FaceBoxes/FaceBoxes_ONNX 的检测质量；检测失败时后续重建无法稳定进行。
- 本实验不训练模型、不改网络结构、不调 3DMM 基，只使用官方推荐配置和权重。
- 多角度渲染只是基于官方 obj 的轻量可视化；如果 `pyrender/trimesh/matplotlib` 不可用，报告保留官方 overlay、pose、landmark 和 obj 输出，并在 summary 中记录 render unavailable。

## 9. 运行与交付说明

AutoDL 500-sample run:

```bash
export 3DDFA_REPO=/root/autodl-tmp/task/3DDFA_V2
export CELEBA_ROOT=/root/autodl-tmp/celeba
cd /root/autodl-tmp/CV-project/stage-3

python code/task8/stage3_task8_check_env.py --config configs/task8_3dface/a800_3ddfa_v2.py
python code/task8/stage3_task8_prepare_samples.py --config configs/task8_3dface/a800_3ddfa_v2.py --celeba-root "$CELEBA_ROOT" --sample-count 500 --clear-existing
python code/task8/stage3_task8_run_reconstruction.py --config configs/task8_3dface/a800_3ddfa_v2.py --resume --skip-existing --continue-on-error
python code/task8/stage3_task8_render_views.py --config configs/task8_3dface/a800_3ddfa_v2.py
python code/task8/stage3_task8_write_report.py --config configs/task8_3dface/a800_3ddfa_v2.py
```

分段续跑示例：

```bash
python code/task8/stage3_task8_run_reconstruction.py --config configs/task8_3dface/a800_3ddfa_v2.py --resume --skip-existing --continue-on-error --start-index 0 --end-index 100
python code/task8/stage3_task8_run_reconstruction.py --config configs/task8_3dface/a800_3ddfa_v2.py --resume --skip-existing --continue-on-error --start-index 100 --end-index 200
```

默认只渲染成功样本中的前 12 张。只有显式传入 `--render-all` 才会渲染全部成功样本。

3DDFA_V2 准备方式：

```bash
cd /root/autodl-tmp/task
git clone https://github.com/cleardusk/3DDFA_V2.git 3DDFA_V2
cd 3DDFA_V2
sh ./build.sh
# 按官方 README/weights 说明准备 weights/mb1_120x120.pth 或 weights/mb1_120x120.onnx
```

适合提交 GitHub：

- `code/task8/`
- `configs/task8_3dface/`
- `reports/task8/stage3_task8_3d_face_reconstruction_report.md`
- `reports/task8/summaries/`
- 少量精选输入图、官方可视化图和 multi-view grid

不要提交：

- CelebA 数据集
- 完整 3DDFA_V2 repo
- `*.pth`、`*.onnx`、checkpoint、缓存
- 大量逐样本中间输出

## 10. 结论

Task8 使用官方 3DDFA_V2 完成了从 CelebA 单张人脸图像到 3D mesh、landmark/overlay/pose 可视化、多角度展示和 Markdown 报告的可复现流程。500 张测试用于验证 pipeline 稳定性；stage-3 保持轻量，只保存课程交付所需 wrapper、配置、summary、报告和少量精选结果图。
""".format(
        source_dataset=prepare.get("source_dataset", "N/A"),
        requested_samples=requested_samples,
        actual_input_samples=actual_input_samples,
        input_dir=prepare.get("output_directory", "N/A"),
        repo_path=repo.get("repo_path", "N/A"),
        commit=repo.get("commit", "N/A"),
        official_config=repo.get("official_config", "N/A"),
        runner=recon.get("runner", "official_subprocess"),
        backend=recon.get("selected_backend", "N/A"),
        python_version=(env.get("python", "N/A").splitlines()[0] if env.get("python") else "N/A"),
        sample_seed=prepare.get("sample_seed", "N/A"),
        sample_strategy=prepare.get("sample_strategy", "N/A"),
        input_grid_block=input_grid_block,
        metrics_table=metrics_table,
        sample_sections="\n\n".join(sample_sections),
        failure_block=failure_block,
    )
    output.write_text(markdown, encoding="utf-8")
    print("Wrote {}".format(output))
    write_json(
        summary_dir(cfg) / "task8_report_summary.json",
        {
            "ready": True,
            "report": str(output),
            "samples_in_report": len(showcase_records),
            "requested_samples": requested_samples,
            "actual_input_samples": actual_input_samples,
            "successful_reconstructions": success_count,
            "failed_reconstructions": failure_count,
            "success_rate": success_rate,
            "rendered_showcase_samples": rendered_count,
            "render_ready": bool(render.get("ready", False)),
        },
    )


if __name__ == "__main__":
    main()
