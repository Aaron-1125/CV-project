#!/usr/bin/env python3
"""Render lightweight multi-view visualizations from official 3DDFA_V2 OBJ files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stage3_task8_common import (
    cfg_get,
    load_config,
    module_available,
    read_json,
    reconstruction_summary_path,
    render_summary_path,
    rendered_views_dir,
    resolve_existing_stage3_path,
    save_image_grid,
    summary_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task8_3dface/a800_3ddfa_v2.py")
    parser.add_argument("--backend", choices=["auto", "pyrender", "matplotlib", "none"], default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-faces", type=int, default=None)
    parser.add_argument("--render-all", action="store_true", default=None, help="Render every successful reconstruction instead of the showcase subset.")
    parser.add_argument("--max-render-samples", type=int, default=None)
    return parser.parse_args()


def parse_angles(cfg: Dict[str, Any]) -> List[Dict[str, float]]:
    raw = cfg_get(cfg, "render", "angles", [])
    angles = []
    for item in raw:
        if isinstance(item, dict):
            angles.append(
                {
                    "name": str(item.get("name", "view")),
                    "yaw": float(item.get("yaw", 0.0)),
                    "pitch": float(item.get("pitch", 0.0)),
                }
            )
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            angles.append({"name": str(item[0]), "yaw": float(item[1]), "pitch": float(item[2]) if len(item) > 2 else 0.0})
    if not angles:
        angles = [
            {"name": "frontal", "yaw": 0.0, "pitch": 0.0},
            {"name": "left_yaw_30", "yaw": -30.0, "pitch": 0.0},
            {"name": "right_yaw_30", "yaw": 30.0, "pitch": 0.0},
        ]
    return angles


def read_obj(path: Path) -> Tuple["Any", "Any", Optional["Any"]]:
    import numpy as np

    vertices = []
    colors = []
    faces = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
            elif line.startswith("f "):
                face = []
                for token in line.strip().split()[1:4]:
                    face.append(int(token.split("/")[0]) - 1)
                if len(face) == 3:
                    faces.append(face)
    if not vertices or not faces:
        raise ValueError("OBJ has no usable vertices/faces: {}".format(path))
    color_array = np.asarray(colors, dtype=float) if len(colors) == len(vertices) else None
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int), color_array


def rotation_matrix(yaw: float, pitch: float) -> "Any":
    import numpy as np

    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    ry = np.asarray(
        [
            [math.cos(yaw_rad), 0.0, math.sin(yaw_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw_rad), 0.0, math.cos(yaw_rad)],
        ],
        dtype=float,
    )
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0.0, math.sin(pitch_rad), math.cos(pitch_rad)],
        ],
        dtype=float,
    )
    return rx.dot(ry)


def render_matplotlib(
    obj_path: Path,
    output_path: Path,
    yaw: float,
    pitch: float,
    image_size: int,
    max_faces: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices, faces, colors = read_obj(obj_path)
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    rotated = centered.dot(rotation_matrix(yaw, pitch).T)
    if faces.shape[0] > max_faces:
        stride = int(math.ceil(float(faces.shape[0]) / float(max_faces)))
        faces = faces[::stride]
    polys = rotated[faces]
    facecolors = None
    if colors is not None:
        facecolors = np.clip(colors[faces].mean(axis=1), 0.0, 1.0)

    fig = plt.figure(figsize=(image_size / 100.0, image_size / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        polys,
        linewidths=0.02,
        edgecolors=(0.15, 0.15, 0.15, 0.08),
        facecolors=facecolors if facecolors is not None else (0.78, 0.72, 0.66, 1.0),
    )
    ax.add_collection3d(collection)
    mins = rotated.min(axis=0)
    maxs = rotated.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.view_init(elev=8, azim=-90)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def render_pyrender(
    obj_path: Path,
    output_path: Path,
    yaw: float,
    pitch: float,
    image_size: int,
    max_faces: int,
) -> Path:
    import numpy as np
    import pyrender  # type: ignore
    import trimesh  # type: ignore
    from PIL import Image

    mesh = trimesh.load(str(obj_path), process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix(yaw, pitch)
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_transform(transform)
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.35, 0.35, 0.35])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    radius = max(float(mesh.extents.max()), 1.0)
    camera = pyrender.PerspectiveCamera(yfov=math.pi / 4.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = radius * 2.6
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)
    color, _ = renderer.render(scene)
    renderer.delete()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(str(output_path))
    return output_path


def choose_render_backend(requested: str) -> Tuple[str, str]:
    if requested == "none":
        return "none", "render disabled by user"
    if requested == "pyrender":
        return "pyrender", "requested explicitly"
    if requested == "matplotlib":
        return "matplotlib", "requested explicitly"
    if module_available("pyrender") and module_available("trimesh"):
        return "pyrender", "auto selected pyrender/trimesh"
    if module_available("matplotlib"):
        return "matplotlib", "auto selected matplotlib fallback"
    return "none", "no supported render backend is available"


def render_one(
    backend: str,
    obj_path: Path,
    output_path: Path,
    yaw: float,
    pitch: float,
    image_size: int,
    max_faces: int,
) -> Path:
    if backend == "pyrender":
        return render_pyrender(obj_path, output_path, yaw, pitch, image_size, max_faces)
    return render_matplotlib(obj_path, output_path, yaw, pitch, image_size, max_faces)


def successful_records(recon: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = []
    for sample in recon.get("records", []):
        obj_value = sample.get("obj_path")
        if not sample.get("success") or not obj_value:
            continue
        obj_path = resolve_existing_stage3_path(obj_value)
        if obj_path.exists():
            row = dict(sample)
            row["resolved_obj_path"] = str(obj_path)
            records.append(row)
    return records


def select_render_records(records: List[Dict[str, Any]], render_all: bool, max_render_samples: int, strategy: str) -> List[Dict[str, Any]]:
    if render_all:
        return records
    if strategy != "first_success":
        raise ValueError("Unsupported render_sample_strategy: {}".format(strategy))
    return records[:max_render_samples]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    recon_path = reconstruction_summary_path(cfg)
    if not recon_path.exists():
        raise FileNotFoundError("Missing reconstruction summary: {}. Run stage3_task8_run_reconstruction.py first.".format(recon_path))
    recon = read_json(recon_path)
    requested_backend = args.backend or str(cfg_get(cfg, "render", "backend", "auto"))
    backend, backend_reason = choose_render_backend(requested_backend)
    image_size = args.image_size or int(cfg_get(cfg, "render", "image_size", 640))
    max_faces = args.max_faces or int(cfg_get(cfg, "render", "max_faces", 8000))
    render_all = bool(cfg_get(cfg, "render", "render_all", False)) if args.render_all is None else bool(args.render_all)
    max_render_samples = args.max_render_samples if args.max_render_samples is not None else int(cfg_get(cfg, "render", "max_render_samples", 12))
    render_strategy = str(cfg_get(cfg, "render", "render_sample_strategy", "first_success"))
    angles = parse_angles(cfg)
    successful = successful_records(recon)
    render_targets = select_render_records(successful, render_all, max_render_samples, render_strategy)
    records = []
    for sample in render_targets:
        sample_id = sample.get("sample_id")
        obj_value = sample.get("resolved_obj_path") or sample.get("obj_path")
        out_dir = rendered_views_dir(cfg) / str(sample_id)
        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "obj_path": obj_value,
            "backend": backend,
            "backend_reason": backend_reason,
            "views": [],
            "available": False,
        }
        if backend == "none":
            record["failure_reason"] = backend_reason
            records.append(record)
            continue
        if not obj_value or not Path(str(obj_value)).exists():
            record["failure_reason"] = "missing obj file"
            records.append(record)
            continue
        obj_path = Path(str(obj_value))
        rendered_paths = []
        try:
            for angle in angles:
                output = out_dir / "{}.jpg".format(angle["name"])
                path = render_one(
                    backend=backend,
                    obj_path=obj_path,
                    output_path=output,
                    yaw=float(angle["yaw"]),
                    pitch=float(angle["pitch"]),
                    image_size=image_size,
                    max_faces=max_faces,
                )
                rendered_paths.append(path)
                record["views"].append({"name": angle["name"], "yaw": angle["yaw"], "pitch": angle["pitch"], "path": str(path)})
            grid = save_image_grid(rendered_paths, [a["name"] for a in angles], out_dir / "multiview_grid.jpg", thumb_size=180)
            record["multiview_grid"] = str(grid) if grid else None
            record["available"] = bool(rendered_paths)
        except Exception as exc:
            record["available"] = False
            record["failure_reason"] = "{}: {}".format(type(exc).__name__, str(exc))
        records.append(record)
    payload = {
        "task": cfg.get("task_name"),
        "ready": any(row.get("available") for row in records),
        "requested_backend": requested_backend,
        "selected_backend": backend,
        "backend_reason": backend_reason,
        "image_size": image_size,
        "max_faces": max_faces,
        "total_successful_reconstructions": len(successful),
        "rendered_count": sum(1 for row in records if row.get("available")),
        "attempted_render_count": len(records),
        "max_render_samples": max_render_samples,
        "render_all": render_all,
        "render_sample_strategy": render_strategy,
        "angles": angles,
        "records": records,
    }
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    write_json(render_summary_path(cfg), payload)
    if not payload["ready"]:
        print("Render unavailable; official reconstruction outputs are preserved. See {}".format(render_summary_path(cfg)))


if __name__ == "__main__":
    main()
