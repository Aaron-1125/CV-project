#!/usr/bin/env python3
"""Write the Stage3 Task9 dynamic face effects Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from stage3_task9_common import (
    NO_VIDEO_HINT,
    demo_summary_path,
    effects_summary_path,
    env_summary_path,
    load_config,
    maybe_read_json,
    performance_summary_path,
    prepare_summary_path,
    relpath_for_markdown,
    report_path,
    report_summary_path,
    summary_dir,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task9_effects/a800_mediapipe_face_effects.py")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def md_path(value: Any, output: Path) -> str:
    return relpath_for_markdown(value, output) if value else ""


def image_line(label: str, value: Any, output: Path) -> str:
    rel = md_path(value, output)
    if not rel:
        return "- {}: N/A".format(label)
    return "- {}: ![{}]({})".format(label, label, rel)


def collect_static_result_lines(effects: Dict[str, Any], demo: Dict[str, Any], prepare: Dict[str, Any], output: Path) -> str:
    records = effects.get("records", [])
    lines: List[str] = []
    for row in records[:6]:
        if not row.get("success"):
            continue
        lines.append("### {}".format(Path(str(row.get("input_path", "image"))).name))
        lines.append("")
        lines.append(image_line("Before/after", row.get("before_after_path"), output))
        lines.append(image_line("Landmarks", row.get("landmark_path"), output))
        lines.append(image_line("Effects", row.get("output_path"), output))
        lines.append("")
    if lines:
        return "\n".join(lines).strip()
    contact = effects.get("static_contact_sheet") or demo.get("static_contact_sheet") or prepare.get("static_input_grid")
    if contact:
        return "![static contact sheet]({})".format(md_path(contact, output))
    return "尚未生成静态图片结果。可先运行 `stage3_task9_run_effects.py --input-dir reports/task9/assets/input/static_images`。"


def video_section(effects: Dict[str, Any], demo: Dict[str, Any], perf: Dict[str, Any], output: Path) -> str:
    output_video = demo.get("final_demo_video") or effects.get("output_video")
    if output_video:
        keyframes = effects.get("keyframes", [])
        keyframe_lines = []
        for row in keyframes[:4]:
            keyframe_lines.append(image_line("Frame {} after".format(row.get("frame_index")), row.get("after_path"), output))
            keyframe_lines.append(image_line("Frame {} landmarks".format(row.get("frame_index")), row.get("landmark_path"), output))
        avg_fps = perf.get("average_fps") or effects.get("average_processing_fps") or "N/A"
        return """视频 demo 已生成：

- Demo video: `{}`
- Benchmark FPS: `{}`
- Processed frames: `{}`
- Faces detected frames: `{}`

关键帧：

{}""".format(
            md_path(output_video, output),
            avg_fps,
            perf.get("processed_frames", effects.get("processed_frames", "N/A")),
            perf.get("faces_detected_frames", effects.get("faces_detected_frames", "N/A")),
            "\n".join(keyframe_lines) if keyframe_lines else "N/A",
        )
    reason = demo.get("video_demo_status") or effects.get("video_demo_status") or "skipped_no_user_video"
    return """动态视频 demo 未生成。

- Status: `{}`
- Reason: no user mp4 was provided.
- Hint: `{}`""".format(reason, NO_VIDEO_HINT)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output = args.output or report_path(cfg)
    env = maybe_read_json(env_summary_path(cfg))
    prepare = maybe_read_json(prepare_summary_path(cfg))
    effects = maybe_read_json(effects_summary_path(cfg))
    demo = maybe_read_json(demo_summary_path(cfg))
    perf = maybe_read_json(performance_summary_path(cfg))
    python_version = env.get("python", {}).get("version") if isinstance(env.get("python"), dict) else "N/A"
    deps = env.get("dependencies", {})
    cv2_version = deps.get("cv2", {}).get("version", "N/A") if isinstance(deps, dict) else "N/A"
    mp_version = deps.get("mediapipe", {}).get("version", "N/A") if isinstance(deps, dict) else "N/A"
    static_section = collect_static_result_lines(effects, demo, prepare, output)
    video_demo = video_section(effects, demo, perf, output)
    markdown = """# Stage3 Task9: 基于人脸关键点的动态贴纸与美颜美妆

## 1. 任务简介

本任务实现基于 MediaPipe Face Mesh 的实时人脸关键点检测和动态特效。动态视频演示只使用用户提供的真实 mp4；CelebA 仅用于静态图片 smoke test 和效果图展示，不会被拼接成假连续视频。

交付内容包括：

- `code/task9/`: 环境检查、素材准备、特效运行、demo 整理、benchmark 和报告脚本。
- `configs/task9_effects/`: A800/MediaPipe 配置。
- `reports/task9/assets/`: 用户视频位置、静态图片、贴纸、结果图、关键帧和 demo video。
- `reports/task9/summaries/`: 环境、准备、效果、demo、性能 summary。

## 2. 基本原理

MediaPipe Face Mesh 可在单张图像或视频帧中预测 468 个三维人脸关键点。贴纸类 AR 特效通常选取稳定的语义关键点作为锚点：眼镜使用左右眼外眼角估计中心、尺度和旋转角；帽子使用脸颊宽度与脸部上沿估计头部宽度和额头位置。贴纸是透明 PNG，经过缩放、旋转后用 alpha blend 叠加到原帧。

美颜美妆使用传统图像处理完成。磨皮使用双边滤波并限制在人脸 mask 内，尽量保留边缘；美白在人脸区域提升亮度并轻微调整饱和度，避免整图过曝；口红根据嘴唇关键点生成 polygon mask，在嘴唇区域叠加指定颜色。

## 3. 实验设置

- 静态图片来源: `{static_source}`。
- 静态样本数量: `{static_count}`。
- 用户视频状态: `{video_status}`。
- 默认用户视频路径: `reports/task9/assets/input/user_video.mp4`。
- OpenCV: `{cv2_version}`。
- MediaPipe: `{mp_version}`。
- Python: `{python_version}`。

## 4. 动态视频 Demo

{video_demo}

## 5. 静态效果展示

{static_section}

## 6. 性能分析

- Benchmark type: `{benchmark_type}`。
- 平均 FPS: `{average_fps}`。
- 图片吞吐: `{images_per_second}` images/s。
- 平均检测耗时: `{avg_det_ms}` ms。
- 平均渲染耗时: `{avg_render_ms}` ms。
- 平均写入耗时: `{avg_write_ms}` ms。
- CPU/GPU 说明: 本实验主要使用 MediaPipe + OpenCV，标准 Python pipeline 主要由 CPU 执行。A800 可被记录为环境信息，但本任务不强制使用 GPU，也不假设 MediaPipe 使用 A800。

## 7. 局限性

- 极端姿态、遮挡、运动模糊和强光照会影响 Face Mesh 关键点稳定性。
- 本任务使用简单自生成贴纸，视觉精细度不如商业 AR 素材。
- 美颜美妆是传统图像处理实现，不是深度学习美颜模型。
- 没有用户 mp4 时，只能展示静态图片效果；报告不会把 CelebA 多人图片拼接成连续动态视频。

## 8. 结论

Task9 构建了一个轻量、稳定、可复现的人脸关键点动态特效 pipeline。真实视频输入用于展示贴纸和美颜美妆的逐帧跟踪效果；CelebA 仅保留为静态 smoke test，避免不同人物图片拼接造成错误的动态演示结论。
""".format(
        static_source=prepare.get("source_dataset", "N/A"),
        static_count=prepare.get("static_sample_count", effects.get("processed_images", "N/A")),
        video_status=prepare.get("user_video_status", effects.get("video_demo_status", "N/A")),
        cv2_version=cv2_version,
        mp_version=mp_version,
        python_version=str(python_version).splitlines()[0] if python_version else "N/A",
        video_demo=video_demo,
        static_section=static_section,
        benchmark_type=perf.get("benchmark_type", "N/A"),
        average_fps=perf.get("average_fps", "N/A"),
        images_per_second=perf.get("images_per_second", "N/A"),
        avg_det_ms=perf.get("average_detection_ms", "N/A"),
        avg_render_ms=perf.get("average_render_ms", "N/A"),
        avg_write_ms=perf.get("average_write_ms", "N/A"),
    )
    write_text(output, markdown)
    write_json(
        report_summary_path(cfg),
        {
            "ready": True,
            "report": str(output),
            "has_video_demo": bool(demo.get("final_demo_video") or effects.get("output_video")),
            "has_static_results": bool(effects.get("records")),
        },
    )


if __name__ == "__main__":
    main()
