#!/usr/bin/env python3
"""Run MediaPipe Face Mesh landmarks and Task9 face effects on images or video."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from stage3_task9_common import (
    FACE_OVAL,
    INNER_LIPS,
    INSTALL_HINT,
    NO_VIDEO_HINT,
    OUTER_LIPS,
    cfg_get,
    compute_process_size,
    compute_glasses_transform,
    compute_hat_transform,
    debug_geometry_dir,
    demo_video_path,
    estimate_face_transform_from_landmarks,
    effects_summary_path,
    ensure_default_stickers,
    ensure_task9_dirs,
    expanded_bbox,
    glasses_path,
    hat_path,
    keyframes_dir,
    landmark_bbox,
    list_images,
    load_config,
    locate_user_video,
    outputs_dir,
    quantize_number,
    safe_stem,
    save_image_grid,
    static_contact_sheet_path,
    static_images_dir,
    summary_dir,
    videos_dir,
    write_json,
)


EFFECT_CHOICES = ["glasses", "hat", "smooth", "whiten", "lipstick"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task9_effects/a800_mediapipe_face_effects.py")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--camera", type=int, default=None, help="Optional webcam index. Headless cloud runs should use --video.")
    parser.add_argument("--effects", nargs="+", choices=EFFECT_CHOICES, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-keyframes", type=int, default=None)
    parser.add_argument("--process-width", type=int, default=None, help="Resize video frames to this width before detection/effects.")
    parser.add_argument("--process-height", type=int, default=None, help="Resize video frames to this height before detection/effects.")
    parser.add_argument("--fast-mode", action="store_true", help="Use realtime-friendly defaults such as 640x360 processing.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Optional experimental device flag. CUDA falls back to CPU for this OpenCV/MediaPipe pipeline.")
    parser.add_argument("--draw-landmarks", action="store_true", help="Also save landmark visualizations for static images.")
    parser.add_argument("--debug-sticker-geometry", action="store_true", help="Save sticker anchor/angle/scale debug visualizations.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def import_runtime_modules():
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        print("ERROR: Missing runtime dependency for Task9: {}: {}".format(type(exc).__name__, str(exc)))
        print("Install with: {}".format(INSTALL_HINT))
        raise SystemExit(1)
    return cv2, mp, np


def enabled_effects_from_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> Set[str]:
    if args.effects:
        return set(args.effects)
    enabled = set()
    mapping = {
        "glasses": "enable_glasses",
        "hat": "enable_hat",
        "smooth": "enable_smooth",
        "whiten": "enable_whiten",
        "lipstick": "enable_lipstick",
    }
    for effect, key in mapping.items():
        if bool(cfg_get(cfg, "effects", key, True)):
            enabled.add(effect)
    return enabled


def load_sticker_bgra(cv2: Any, np: Any, path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError("Could not read sticker PNG: {}".format(path))
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=2)
    return image


def clamp_strength(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return max(0.0, min(1.0, numeric))


class FaceEffectsProcessor:
    """Reusable MediaPipe + OpenCV processor for Task9 images, video, and benchmark."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        enabled_effects: Set[str],
        static_image_mode: bool = False,
        debug_sticker_geometry: bool = False,
    ) -> None:
        self.cfg = cfg
        self.enabled_effects = set(enabled_effects)
        self.debug_sticker_geometry = debug_sticker_geometry
        self.cv2, self.mp, self.np = import_runtime_modules()
        ensure_default_stickers(cfg, force=False)
        self.glasses = load_sticker_bgra(self.cv2, self.np, glasses_path(cfg))
        self.hat = load_sticker_bgra(self.cv2, self.np, hat_path(cfg))
        self.smooth_strength = clamp_strength(cfg_get(cfg, "effects", "smooth_strength", 0.55), 0.55)
        self.whiten_strength = clamp_strength(cfg_get(cfg, "effects", "whiten_strength", 0.35), 0.35)
        self.lipstick_alpha = clamp_strength(cfg_get(cfg, "effects", "lipstick_alpha", 0.45), 0.45)
        self.face_roi_margin = float(cfg_get(cfg, "effects", "face_roi_margin", 0.15))
        self.lips_roi_margin = float(cfg_get(cfg, "effects", "lips_roi_margin", 0.25))
        self.smooth_backend = str(cfg_get(cfg, "effects", "smooth_backend", "bilateral_fast"))
        self.smooth_downscale = max(0.1, min(1.0, float(cfg_get(cfg, "effects", "smooth_downscale", 0.5))))
        self.smooth_diameter = int(cfg_get(cfg, "effects", "smooth_diameter", 5) or 5)
        self.smooth_sigma_color = float(cfg_get(cfg, "effects", "smooth_sigma_color", 30) or 30)
        self.smooth_sigma_space = float(cfg_get(cfg, "effects", "smooth_sigma_space", 30) or 30)
        self.glasses_scale_factor = float(cfg_get(cfg, "effects", "glasses_scale_factor", 2.2))
        self.glasses_y_offset_factor = float(cfg_get(cfg, "effects", "glasses_y_offset_factor", 0.03))
        self.hat_scale_factor = float(cfg_get(cfg, "effects", "hat_scale_factor", 1.35))
        self.hat_y_offset_factor = float(cfg_get(cfg, "effects", "hat_y_offset_factor", 0.55))
        self.sticker_cache_enabled = bool(cfg_get(cfg, "stickers", "cache_enabled", True))
        self.angle_quantization = float(cfg_get(cfg, "stickers", "angle_quantization", 5) or 5)
        self.size_quantization = float(cfg_get(cfg, "stickers", "size_quantization", 8) or 8)
        self.sticker_cache: Dict[Tuple[str, int, int], Any] = {}
        rgb = list(cfg_get(cfg, "effects", "lipstick_color", (190, 35, 80)))
        if len(rgb) != 3:
            rgb = [190, 35, 80]
        self.lipstick_bgr = tuple(int(max(0, min(255, value))) for value in reversed(rgb))
        self.face_mesh = self.mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self.face_mesh.close()

    def detect_landmarks(self, frame_bgr: Any) -> Tuple[Optional[Any], float]:
        cv2 = self.cv2
        np = self.np
        h, w = frame_bgr.shape[:2]
        started = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.face_mesh.process(rgb)
        elapsed = time.perf_counter() - started
        if not result.multi_face_landmarks:
            return None, elapsed
        face = result.multi_face_landmarks[0]
        points = np.array([(lm.x * w, lm.y * h) for lm in face.landmark], dtype=np.float32)
        return points, elapsed

    def process_frame(self, frame_bgr: Any, save_landmark_frame: bool = False) -> Dict[str, Any]:
        points, detection_seconds = self.detect_landmarks(frame_bgr)
        output = frame_bgr.copy()
        landmark_frame = None
        debug_geometry_frame = None
        geometry = None
        sticker_boxes: List[Dict[str, Any]] = []
        render_seconds = 0.0
        beauty_seconds = 0.0
        lipstick_seconds = 0.0
        sticker_seconds = 0.0
        face_detected = points is not None
        if face_detected:
            h, w = frame_bgr.shape[:2]
            geometry = estimate_face_transform_from_landmarks(points, w, h)
            started = time.perf_counter()
            if "smooth" in self.enabled_effects:
                effect_started = time.perf_counter()
                output = self.apply_smooth(output, points)
                beauty_seconds += time.perf_counter() - effect_started
            if "whiten" in self.enabled_effects:
                effect_started = time.perf_counter()
                output = self.apply_whiten(output, points)
                beauty_seconds += time.perf_counter() - effect_started
            if "lipstick" in self.enabled_effects:
                effect_started = time.perf_counter()
                output = self.apply_lipstick(output, points)
                lipstick_seconds += time.perf_counter() - effect_started
            if geometry:
                if "hat" in self.enabled_effects:
                    effect_started = time.perf_counter()
                    output, box = self.apply_hat(output, geometry)
                    sticker_seconds += time.perf_counter() - effect_started
                    if box:
                        sticker_boxes.append(box)
                if "glasses" in self.enabled_effects:
                    effect_started = time.perf_counter()
                    output, box = self.apply_glasses(output, geometry)
                    sticker_seconds += time.perf_counter() - effect_started
                    if box:
                        sticker_boxes.append(box)
            render_seconds = time.perf_counter() - started
            if save_landmark_frame:
                landmark_frame = self.draw_landmarks(frame_bgr, points, geometry)
            if self.debug_sticker_geometry:
                debug_geometry_frame = self.draw_sticker_geometry(frame_bgr, points, geometry, sticker_boxes)
        return {
            "frame": output,
            "landmark_frame": landmark_frame,
            "debug_geometry_frame": debug_geometry_frame,
            "geometry": geometry,
            "sticker_boxes": sticker_boxes,
            "face_detected": face_detected,
            "detection_seconds": detection_seconds,
            "sticker_seconds": sticker_seconds,
            "beauty_seconds": beauty_seconds,
            "lipstick_seconds": lipstick_seconds,
            "render_seconds": render_seconds,
        }

    def point(self, points: Any, index: int) -> Tuple[float, float]:
        return float(points[index][0]), float(points[index][1])

    def polygon_mask(self, shape: Tuple[int, int, int], points: Any, indices: Sequence[int], blur: bool = True):
        cv2 = self.cv2
        np = self.np
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        if len(points) <= max(indices):
            return mask
        poly = points[indices].astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [poly], 1.0)
        if blur:
            sigma = max(5, int(min(h, w) * 0.018))
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        return np.clip(mask, 0.0, 1.0)

    def face_mask(self, shape: Tuple[int, int, int], points: Any, blur: bool = True):
        return self.polygon_mask(shape, points, FACE_OVAL, blur=blur)

    def blend_with_mask(self, base: Any, effect: Any, mask: Any, strength: float):
        np = self.np
        alpha = np.clip(mask[..., None] * strength, 0.0, 1.0)
        blended = base.astype(np.float32) * (1.0 - alpha) + effect.astype(np.float32) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def apply_smooth(self, frame: Any, points: Any):
        cv2 = self.cv2
        x1, y1, x2, y2 = self.landmark_roi(points, FACE_OVAL, frame.shape, self.face_roi_margin)
        if x2 <= x1 or y2 <= y1:
            return frame
        roi = frame[y1:y2, x1:x2]
        local_points = points.copy()
        local_points[:, 0] -= x1
        local_points[:, 1] -= y1
        mask = self.face_mask(roi.shape, local_points, blur=True)
        filtered = self.fast_bilateral_roi(roi)
        blended = self.blend_with_mask(roi, filtered, mask, self.smooth_strength)
        output = frame.copy()
        output[y1:y2, x1:x2] = blended
        return output

    def apply_whiten(self, frame: Any, points: Any):
        cv2 = self.cv2
        np = self.np
        x1, y1, x2, y2 = self.landmark_roi(points, FACE_OVAL, frame.shape, self.face_roi_margin)
        if x2 <= x1 or y2 <= y1:
            return frame
        roi = frame[y1:y2, x1:x2]
        local_points = points.copy()
        local_points[:, 0] -= x1
        local_points[:, 1] -= y1
        mask = self.face_mask(roi.shape, local_points, blur=True)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]
        lab[:, :, 0] = l_channel + (255.0 - l_channel) * (0.22 * self.whiten_strength)
        brightened = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(brightened, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.04 * self.whiten_strength), 0, 255)
        brightened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        blended = self.blend_with_mask(roi, brightened, mask, min(0.85, 0.85 * self.whiten_strength + 0.1))
        output = frame.copy()
        output[y1:y2, x1:x2] = blended
        return output

    def lip_mask(self, shape: Tuple[int, int, int], points: Any):
        cv2 = self.cv2
        np = self.np
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        if len(points) <= max(max(OUTER_LIPS), max(INNER_LIPS)):
            return mask
        outer = points[OUTER_LIPS].astype(np.int32)
        inner = points[INNER_LIPS].astype(np.int32)
        outer[:, 0] = np.clip(outer[:, 0], 0, w - 1)
        outer[:, 1] = np.clip(outer[:, 1], 0, h - 1)
        inner[:, 0] = np.clip(inner[:, 0], 0, w - 1)
        inner[:, 1] = np.clip(inner[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [outer], 1.0)
        cv2.fillPoly(mask, [inner], 0.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2, int(min(h, w) * 0.006)))
        return np.clip(mask, 0.0, 1.0)

    def apply_lipstick(self, frame: Any, points: Any):
        np = self.np
        x1, y1, x2, y2 = self.landmark_roi(points, OUTER_LIPS, frame.shape, self.lips_roi_margin, min_size=6)
        if x2 <= x1 or y2 <= y1:
            return frame
        roi = frame[y1:y2, x1:x2]
        local_points = points.copy()
        local_points[:, 0] -= x1
        local_points[:, 1] -= y1
        mask = self.lip_mask(roi.shape, local_points)
        color_layer = np.zeros_like(roi)
        color_layer[:, :] = self.lipstick_bgr
        alpha = np.clip(mask[..., None] * self.lipstick_alpha, 0.0, 1.0)
        blended = roi.astype(np.float32) * (1.0 - alpha) + color_layer.astype(np.float32) * alpha
        output = frame.copy()
        output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return output

    def landmark_roi(
        self,
        points: Any,
        indices: Sequence[int],
        shape: Tuple[int, int, int],
        margin: float,
        min_size: int = 8,
    ) -> Tuple[int, int, int, int]:
        h, w = shape[:2]
        return expanded_bbox(landmark_bbox(points, indices), w, h, margin=margin, min_size=min_size)

    def fast_bilateral_roi(self, roi: Any):
        cv2 = self.cv2
        if roi.size == 0:
            return roi
        d = max(1, int(self.smooth_diameter))
        if self.smooth_backend != "bilateral_fast" or self.smooth_downscale >= 0.98:
            return cv2.bilateralFilter(roi, d=d, sigmaColor=self.smooth_sigma_color, sigmaSpace=self.smooth_sigma_space)
        h, w = roi.shape[:2]
        small_w = max(8, int(round(w * self.smooth_downscale)))
        small_h = max(8, int(round(h * self.smooth_downscale)))
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
        filtered_small = cv2.bilateralFilter(
            small,
            d=d,
            sigmaColor=self.smooth_sigma_color,
            sigmaSpace=self.smooth_sigma_space,
        )
        return cv2.resize(filtered_small, (w, h), interpolation=cv2.INTER_LINEAR)

    def transform_sticker_rgba(self, sticker_name: str, sticker: Any, desired_width: float, angle_degrees: float):
        cv2 = self.cv2
        desired_width = max(10.0, float(desired_width))
        cached_width = int(quantize_number(desired_width, self.size_quantization))
        cached_angle = int(quantize_number(angle_degrees, self.angle_quantization))
        cache_key = (sticker_name, cached_width, cached_angle)
        if self.sticker_cache_enabled and cache_key in self.sticker_cache:
            return self.sticker_cache[cache_key]
        transform_width = float(cached_width if self.sticker_cache_enabled else desired_width)
        transform_angle = float(cached_angle if self.sticker_cache_enabled else angle_degrees)
        h, w = sticker.shape[:2]
        scale = transform_width / float(w)
        resized = cv2.resize(
            sticker,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        rh, rw = resized.shape[:2]
        center = (rw / 2.0, rh / 2.0)
        matrix = cv2.getRotationMatrix2D(center, transform_angle, 1.0)
        cos_v = abs(matrix[0, 0])
        sin_v = abs(matrix[0, 1])
        new_w = int((rh * sin_v) + (rw * cos_v))
        new_h = int((rh * cos_v) + (rw * sin_v))
        matrix[0, 2] += (new_w / 2.0) - center[0]
        matrix[1, 2] += (new_h / 2.0) - center[1]
        transformed = cv2.warpAffine(
            resized,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        if self.sticker_cache_enabled:
            self.sticker_cache[cache_key] = transformed
        return transformed

    def rotate_scale_sticker(self, sticker: Any, desired_width: float, angle_degrees: float):
        return self.transform_sticker_rgba("sticker", sticker, desired_width, angle_degrees)

    def overlay_rgba_at_center(
        self,
        frame: Any,
        overlay: Any,
        center_xy: Tuple[float, float],
        label: str = "sticker",
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        np = self.np
        h, w = frame.shape[:2]
        oh, ow = overlay.shape[:2]
        x1 = int(round(center_xy[0] - ow / 2.0))
        y1 = int(round(center_xy[1] - oh / 2.0))
        x2 = x1 + ow
        y2 = y1 + oh
        clip_x1 = max(0, x1)
        clip_y1 = max(0, y1)
        clip_x2 = min(w, x2)
        clip_y2 = min(h, y2)
        if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
            return frame, None
        ox1 = clip_x1 - x1
        oy1 = clip_y1 - y1
        ox2 = ox1 + (clip_x2 - clip_x1)
        oy2 = oy1 + (clip_y2 - clip_y1)
        sticker_roi = overlay[oy1:oy2, ox1:ox2]
        alpha = sticker_roi[:, :, 3:4].astype(np.float32) / 255.0
        roi = frame[clip_y1:clip_y2, clip_x1:clip_x2].astype(np.float32)
        blended = sticker_roi[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
        output = frame.copy()
        output[clip_y1:clip_y2, clip_x1:clip_x2] = np.clip(blended, 0, 255).astype(np.uint8)
        box = {
            "label": label,
            "center": (float(center_xy[0]), float(center_xy[1])),
            "box": (float(x1), float(y1), float(x2), float(y2)),
            "clipped_box": (float(clip_x1), float(clip_y1), float(clip_x2), float(clip_y2)),
            "width": float(ow),
            "height": float(oh),
        }
        return output, box

    def overlay_bgra(self, frame: Any, overlay: Any, center_xy: Tuple[float, float]):
        output, _ = self.overlay_rgba_at_center(frame, overlay, center_xy)
        return output

    def apply_glasses(self, frame: Any, geometry: Dict[str, Any]) -> Tuple[Any, Optional[Dict[str, Any]]]:
        pose = compute_glasses_transform(
            geometry,
            scale_factor=self.glasses_scale_factor,
            y_offset_factor=self.glasses_y_offset_factor,
        )
        sticker = self.transform_sticker_rgba("glasses", self.glasses, pose["width"], pose["angle_deg"])
        output, box = self.overlay_rgba_at_center(frame, sticker, (pose["center_x"], pose["center_y"]), label="glasses")
        if box:
            box["pose"] = pose
        return output, box

    def apply_hat(self, frame: Any, geometry: Dict[str, Any]) -> Tuple[Any, Optional[Dict[str, Any]]]:
        sticker_aspect = self.hat.shape[0] / float(self.hat.shape[1])
        pose = compute_hat_transform(
            geometry,
            sticker_aspect=sticker_aspect,
            scale_factor=self.hat_scale_factor,
            y_offset_factor=self.hat_y_offset_factor,
        )
        sticker = self.transform_sticker_rgba("hat", self.hat, pose["width"], pose["angle_deg"])
        output, box = self.overlay_rgba_at_center(frame, sticker, (pose["center_x"], pose["center_y"]), label="hat")
        if box:
            box["pose"] = pose
        return output, box

    def draw_landmarks(self, frame: Any, points: Any, geometry: Optional[Dict[str, Any]] = None):
        cv2 = self.cv2
        output = frame.copy()
        h, w = output.shape[:2]
        for idx in range(points.shape[0]):
            x = int(max(0, min(w - 1, points[idx][0])))
            y = int(max(0, min(h - 1, points[idx][1])))
            cv2.circle(output, (x, y), 1, (0, 210, 120), -1, lineType=cv2.LINE_AA)
        for indices, color in [(FACE_OVAL, (255, 180, 40)), (OUTER_LIPS, (80, 80, 255))]:
            poly = points[indices].astype("int32")
            cv2.polylines(output, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        if geometry:
            self.draw_geometry_overlays(output, geometry, [])
        return output

    def draw_geometry_overlays(self, image: Any, geometry: Dict[str, Any], sticker_boxes: Sequence[Dict[str, Any]]) -> None:
        cv2 = self.cv2

        def pt(value: Sequence[float]) -> Tuple[int, int]:
            return int(round(float(value[0]))), int(round(float(value[1])))

        left_eye = pt(geometry["left_eye_center"])
        right_eye = pt(geometry["right_eye_center"])
        eye_center = pt(geometry["eye_center"])
        face_center = pt(geometry["face_center"])
        brow_center = pt(geometry["brow_center"])
        forehead_anchor = pt(geometry["forehead_anchor"])
        cv2.line(image, left_eye, right_eye, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(image, left_eye, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(image, right_eye, 5, (0, 128, 255), -1, cv2.LINE_AA)
        cv2.circle(image, eye_center, 5, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(image, face_center, 6, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(image, brow_center, 5, (255, 120, 0), -1, cv2.LINE_AA)
        cv2.circle(image, forehead_anchor, 5, (80, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            "head_angle={:.1f}, sticker_angle={:.1f}".format(geometry["angle_deg"], geometry["sticker_angle_deg"]),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "head_angle={:.1f}, sticker_angle={:.1f}".format(geometry["angle_deg"], geometry["sticker_angle_deg"]),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        colors = {"glasses": (0, 255, 255), "hat": (80, 180, 255)}
        for box in sticker_boxes:
            x1, y1, x2, y2 = box["box"]
            label = str(box.get("label", "sticker"))
            color = colors.get(label, (255, 255, 255))
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
            cv2.circle(image, pt(box["center"]), 5, color, -1, cv2.LINE_AA)
            cv2.putText(image, label, (int(x1), max(18, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def draw_sticker_geometry(
        self,
        frame: Any,
        points: Any,
        geometry: Optional[Dict[str, Any]],
        sticker_boxes: Sequence[Dict[str, Any]],
    ):
        output = self.draw_landmarks(frame, points, None)
        if geometry:
            self.draw_geometry_overlays(output, geometry, sticker_boxes)
        return output


def labeled_before_after(cv2: Any, np: Any, before: Any, after: Any):
    h = max(before.shape[0], after.shape[0])
    w = max(before.shape[1], after.shape[1])

    def fit(image: Any):
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        y = (h - image.shape[0]) // 2
        x = (w - image.shape[1]) // 2
        canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
        return canvas

    left = fit(before)
    right = fit(after)
    label_h = 34
    canvas = np.full((h + label_h, w * 2, 3), 255, dtype=np.uint8)
    canvas[label_h:, :w] = left
    canvas[label_h:, w:] = right
    cv2.putText(canvas, "Before", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "After", (w + 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
    return canvas


def write_image(cv2: Any, path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError("Failed to write image: {}".format(path))


def process_static_images(
    cfg: Dict[str, Any],
    image_paths: Sequence[Path],
    enabled_effects: Set[str],
    draw_landmarks: bool,
    debug_sticker_geometry: bool,
) -> Dict[str, Any]:
    processor = FaceEffectsProcessor(cfg, enabled_effects, static_image_mode=True, debug_sticker_geometry=debug_sticker_geometry)
    cv2 = processor.cv2
    np = processor.np
    records: List[Dict[str, Any]] = []
    contact_candidates: List[Path] = []
    try:
        for idx, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path))
            if frame is None:
                records.append({"input_path": str(image_path), "success": False, "error": "cv2.imread returned None"})
                continue
            result = processor.process_frame(frame, save_landmark_frame=True)
            sample_id = "image_{:03d}_{}".format(idx, safe_stem(image_path.stem))
            output_path = outputs_dir(cfg) / "{}_effects.jpg".format(sample_id)
            landmark_path = outputs_dir(cfg) / "{}_landmarks.jpg".format(sample_id)
            compare_path = outputs_dir(cfg) / "{}_before_after.jpg".format(sample_id)
            debug_path = debug_geometry_dir(cfg) / "{}_debug_geometry.jpg".format(sample_id)
            write_image(cv2, output_path, result["frame"])
            if result["landmark_frame"] is not None:
                write_image(cv2, landmark_path, result["landmark_frame"])
            if result.get("debug_geometry_frame") is not None:
                write_image(cv2, debug_path, result["debug_geometry_frame"])
            comparison = labeled_before_after(cv2, np, frame, result["frame"])
            write_image(cv2, compare_path, comparison)
            contact_candidates.append(compare_path)
            records.append(
                {
                    "input_path": str(image_path),
                    "success": True,
                    "face_detected": result["face_detected"],
                    "output_path": str(output_path),
                    "landmark_path": str(landmark_path) if result["landmark_frame"] is not None else None,
                    "debug_geometry_path": str(debug_path) if result.get("debug_geometry_frame") is not None else None,
                    "geometry": result.get("geometry"),
                    "sticker_boxes": result.get("sticker_boxes"),
                    "before_after_path": str(compare_path),
                    "detection_seconds": result["detection_seconds"],
                    "render_seconds": result["render_seconds"],
                }
            )
    finally:
        processor.close()
    contact_sheet = save_image_grid(contact_candidates, ["before/after"] * len(contact_candidates), static_contact_sheet_path(cfg))
    return {
        "mode": "image_batch",
        "processed_images": len([row for row in records if row.get("success")]),
        "faces_detected": len([row for row in records if row.get("face_detected")]),
        "records": records,
        "static_contact_sheet": str(contact_sheet) if contact_sheet else None,
    }


def resolve_process_dimensions(
    cfg: Dict[str, Any],
    source_width: int,
    source_height: int,
    process_width: Optional[int] = None,
    process_height: Optional[int] = None,
    fast_mode: Optional[bool] = None,
) -> Tuple[int, int, bool]:
    configured_fast = bool(cfg_get(cfg, "video", "fast_mode", False))
    use_fast = configured_fast if fast_mode is None else bool(fast_mode)
    if use_fast and process_width is None and process_height is None:
        target_width, target_height = 640, 360
    else:
        target_width = int(process_width if process_width is not None else cfg_get(cfg, "video", "process_width", 0) or 0)
        target_height = int(process_height if process_height is not None else cfg_get(cfg, "video", "process_height", 0) or 0)
        if target_width <= 0 and target_height <= 0:
            target_width = int(cfg_get(cfg, "video", "width", 0) or 0)
            target_height = int(cfg_get(cfg, "video", "height", 0) or 0)
    keep_aspect = bool(cfg_get(cfg, "video", "keep_aspect_ratio", True))
    width, height = compute_process_size(source_width, source_height, target_width, target_height, keep_aspect)
    return width, height, use_fast


def resize_frame_to_config(
    cv2: Any,
    cfg: Dict[str, Any],
    frame: Any,
    process_width: Optional[int] = None,
    process_height: Optional[int] = None,
    fast_mode: Optional[bool] = None,
):
    height, width = frame.shape[:2]
    out_width, out_height, _ = resolve_process_dimensions(cfg, width, height, process_width, process_height, fast_mode)
    if out_width > 0 and out_height > 0 and (width != out_width or height != out_height):
        return cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
    return frame


def should_save_keyframe(frame_index: int, total_frames: int, max_keyframes: int, already_saved: int) -> bool:
    if already_saved >= max_keyframes:
        return False
    if frame_index == 0:
        return True
    interval = 30
    if total_frames > 0:
        interval = max(1, total_frames // max(1, max_keyframes))
    return frame_index % interval == 0


def process_video(
    cfg: Dict[str, Any],
    video_path: Path,
    enabled_effects: Set[str],
    output_video: Optional[Path],
    max_frames: Optional[int],
    fps_override: Optional[float],
    max_keyframes: Optional[int],
    process_width: Optional[int],
    process_height: Optional[int],
    fast_mode: Optional[bool],
    device: str,
    debug_sticker_geometry: bool = False,
    camera_index: Optional[int] = None,
) -> Dict[str, Any]:
    cv2, _, _ = import_runtime_modules()
    capture_source: Any = int(camera_index) if camera_index is not None else str(video_path)
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise FileNotFoundError("Could not open video source: {}".format(capture_source))
    processor = FaceEffectsProcessor(cfg, enabled_effects, static_image_mode=False, debug_sticker_geometry=debug_sticker_geometry)
    out_path = output_video or demo_video_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    key_dir = keyframes_dir(cfg)
    key_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir = debug_geometry_dir(cfg)
    dbg_dir.mkdir(parents=True, exist_ok=True)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    output_fps = float(fps_override or source_fps or cfg_get(cfg, "video", "fps", 20) or 20)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    configured_keyframes = int(max_keyframes or cfg_get(cfg, "video", "max_keyframes", 8) or 8)
    writer = None
    frame_index = 0
    processed_frames = 0
    faces_detected = 0
    detection_seconds = 0.0
    render_seconds = 0.0
    sticker_seconds = 0.0
    beauty_seconds = 0.0
    lipstick_seconds = 0.0
    write_seconds = 0.0
    process_size: Optional[Tuple[int, int]] = None
    keyframe_records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_frame_to_config(cv2, cfg, frame, process_width, process_height, fast_mode)
            if process_size is None:
                process_size = (int(frame.shape[1]), int(frame.shape[0]))
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (w, h))
                if not writer.isOpened():
                    raise IOError("Could not open output video writer: {}".format(out_path))
            save_keyframe = should_save_keyframe(frame_index, total_frames, configured_keyframes, len(keyframe_records))
            result = processor.process_frame(frame, save_landmark_frame=save_keyframe)
            detection_seconds += result["detection_seconds"]
            render_seconds += result["render_seconds"]
            sticker_seconds += result.get("sticker_seconds", 0.0)
            beauty_seconds += result.get("beauty_seconds", 0.0)
            lipstick_seconds += result.get("lipstick_seconds", 0.0)
            faces_detected += 1 if result["face_detected"] else 0
            write_started = time.perf_counter()
            writer.write(result["frame"])
            write_seconds += time.perf_counter() - write_started
            if save_keyframe:
                key_id = "keyframe_{:05d}".format(frame_index)
                before_path = key_dir / "{}_before.jpg".format(key_id)
                after_path = key_dir / "{}_after.jpg".format(key_id)
                landmark_path = key_dir / "{}_landmarks.jpg".format(key_id)
                debug_path = dbg_dir / "{}_debug_geometry.jpg".format(key_id)
                write_image(cv2, before_path, frame)
                write_image(cv2, after_path, result["frame"])
                if result["landmark_frame"] is not None:
                    write_image(cv2, landmark_path, result["landmark_frame"])
                if result.get("debug_geometry_frame") is not None:
                    write_image(cv2, debug_path, result["debug_geometry_frame"])
                keyframe_records.append(
                    {
                        "frame_index": frame_index,
                        "before_path": str(before_path),
                        "after_path": str(after_path),
                        "landmark_path": str(landmark_path) if result["landmark_frame"] is not None else None,
                        "debug_geometry_path": str(debug_path) if result.get("debug_geometry_frame") is not None else None,
                        "geometry": result.get("geometry"),
                        "sticker_boxes": result.get("sticker_boxes"),
                        "face_detected": result["face_detected"],
                    }
                )
            processed_frames += 1
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        processor.close()
    elapsed = time.perf_counter() - started
    return {
        "mode": "video",
        "processing_mode": "fast" if (fast_mode or bool(cfg_get(cfg, "video", "fast_mode", False))) else "quality",
        "device_requested": device,
        "device_used": "cpu",
        "device_note": "CUDA was requested but this standard MediaPipe/OpenCV pipeline remains CPU-bound." if device == "cuda" else "CPU MediaPipe/OpenCV pipeline.",
        "input_video": str(video_path) if camera_index is None else "camera:{}".format(camera_index),
        "output_video": str(out_path),
        "source_fps": source_fps,
        "output_fps": output_fps,
        "process_width": process_size[0] if process_size else None,
        "process_height": process_size[1] if process_size else None,
        "total_frames_in_source": total_frames,
        "processed_frames": processed_frames,
        "faces_detected_frames": faces_detected,
        "average_processing_fps": processed_frames / elapsed if elapsed > 0 else 0.0,
        "total_seconds": elapsed,
        "detection_seconds": detection_seconds,
        "sticker_seconds": sticker_seconds,
        "beauty_seconds": beauty_seconds,
        "lipstick_seconds": lipstick_seconds,
        "render_seconds": render_seconds,
        "write_seconds": write_seconds,
        "average_detection_ms": (detection_seconds / processed_frames * 1000.0) if processed_frames else None,
        "average_sticker_ms": (sticker_seconds / processed_frames * 1000.0) if processed_frames else None,
        "average_beauty_ms": (beauty_seconds / processed_frames * 1000.0) if processed_frames else None,
        "average_lipstick_ms": (lipstick_seconds / processed_frames * 1000.0) if processed_frames else None,
        "average_render_ms": (render_seconds / processed_frames * 1000.0) if processed_frames else None,
        "average_write_ms": (write_seconds / processed_frames * 1000.0) if processed_frames else None,
        "sticker_cache_size": len(processor.sticker_cache),
        "sticker_cache_enabled": processor.sticker_cache_enabled,
        "keyframes": keyframe_records,
        "debug_sticker_geometry": debug_sticker_geometry,
    }


def resolve_static_image_inputs(cfg: Dict[str, Any], image: Optional[Path], input_directory: Optional[Path]) -> List[Path]:
    if image:
        path = image.expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError("Image does not exist: {}".format(path))
        return [path.resolve()]
    directory = input_directory or static_images_dir(cfg)
    return list_images(directory, recursive=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_task9_dirs(cfg)
    enabled_effects = enabled_effects_from_args(args, cfg)
    payload: Dict[str, Any] = {
        "task": cfg.get("task_name"),
        "effects": sorted(enabled_effects),
        "synthetic_video_from_images": False,
    }
    if args.camera is not None:
        result = process_video(
            cfg,
            Path("camera"),
            enabled_effects,
            args.output_video,
            args.max_frames,
            args.fps,
            args.max_keyframes,
            args.process_width,
            args.process_height,
            True if args.fast_mode else None,
            args.device,
            debug_sticker_geometry=args.debug_sticker_geometry,
            camera_index=args.camera,
        )
        payload.update(result)
    elif args.video is not None:
        video_path = args.video.expanduser()
        if not video_path.is_absolute():
            video_path = Path.cwd() / video_path
        if not video_path.is_file():
            raise FileNotFoundError("Video does not exist: {}. {}".format(video_path, NO_VIDEO_HINT))
        result = process_video(
            cfg,
            video_path.resolve(),
            enabled_effects,
            args.output_video,
            args.max_frames,
            args.fps,
            args.max_keyframes,
            args.process_width,
            args.process_height,
            True if args.fast_mode else None,
            args.device,
            debug_sticker_geometry=args.debug_sticker_geometry,
        )
        payload.update(result)
    elif args.image is not None or args.input_dir is not None:
        image_paths = resolve_static_image_inputs(cfg, args.image, args.input_dir)
        result = process_static_images(
            cfg,
            image_paths,
            enabled_effects,
            draw_landmarks=True,
            debug_sticker_geometry=args.debug_sticker_geometry,
        )
        payload.update(result)
        payload["video_demo_status"] = "skipped_static_image_mode"
    else:
        video_path = locate_user_video(cfg)
        if video_path:
            result = process_video(
                cfg,
                video_path,
                enabled_effects,
                args.output_video,
                args.max_frames,
                args.fps,
                args.max_keyframes,
                args.process_width,
                args.process_height,
                True if args.fast_mode else None,
                args.device,
                debug_sticker_geometry=args.debug_sticker_geometry,
            )
            payload.update(result)
        else:
            print(NO_VIDEO_HINT)
            image_paths = resolve_static_image_inputs(cfg, None, None)
            result = process_static_images(
                cfg,
                image_paths,
                enabled_effects,
                draw_landmarks=True,
                debug_sticker_geometry=args.debug_sticker_geometry,
            )
            payload.update(result)
            payload["video_demo_status"] = "skipped_no_user_video"
            payload["video_hint"] = NO_VIDEO_HINT
    summary_dir(cfg).mkdir(parents=True, exist_ok=True)
    write_json(effects_summary_path(cfg), payload)


if __name__ == "__main__":
    main()
