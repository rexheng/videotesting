"""Landmark geometry used to align a face and to bound a torso."""

from __future__ import annotations

import math

import numpy as np

from face_shirt.config import HIP_VISIBILITY, MIN_TORSO_PX, SHOULDER_VISIBILITY, TORSO_SHRINK

# Subject's right eye, left eye, nose tip, right mouth corner, left mouth
# corner. The subject's right is image-left in the raw, unmirrored frame.
SFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def landmark_value(landmark, name: str, default: float = 0.0) -> float:
    if isinstance(landmark, dict):
        return float(landmark.get(name, default))
    return float(getattr(landmark, name, default))


def yaw_degrees(landmarks) -> float:
    """Positive when the person turns to their own left."""
    left = landmarks[234]
    right = landmarks[454]
    rise = landmark_value(right, "z") - landmark_value(left, "z")
    run = landmark_value(right, "x") - landmark_value(left, "x")
    return math.degrees(math.atan2(rise, run))


def five_points(landmarks, width: int, height: int) -> np.ndarray:
    """Pixel points in SFace template order."""

    def pixel(index: int) -> np.ndarray:
        landmark = landmarks[index]
        return np.array(
            [landmark_value(landmark, "x") * width, landmark_value(landmark, "y") * height],
            dtype=np.float64,
        )

    def mean(a: int, b: int) -> np.ndarray:
        return (pixel(a) + pixel(b)) / 2.0

    return np.stack([mean(33, 133), mean(362, 263), pixel(1), pixel(61), pixel(291)])


def estimate_similarity(src: np.ndarray, dst: np.ndarray) -> dict[str, float]:
    """Least-squares similarity, no reflection.

    Canvas order: x' = a·x + c·y + e, y' = b·x + d·y + f.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_delta = src - src_mean
    dst_delta = dst - dst_mean
    norm = float(np.sum(src_delta[:, 0] ** 2 + src_delta[:, 1] ** 2))
    if norm == 0.0:
        return {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0, "e": 0.0, "f": 0.0}
    align = float(np.sum(src_delta[:, 0] * dst_delta[:, 0] + src_delta[:, 1] * dst_delta[:, 1]) / norm)
    rotate = float(np.sum(src_delta[:, 0] * dst_delta[:, 1] - src_delta[:, 1] * dst_delta[:, 0]) / norm)
    translate_x = float(dst_mean[0] - (align * src_mean[0] - rotate * src_mean[1]))
    translate_y = float(dst_mean[1] - (rotate * src_mean[0] + align * src_mean[1]))
    return {
        "a": align,
        "b": rotate,
        "c": -rotate,
        "d": align,
        "e": translate_x,
        "f": translate_y,
    }


def apply_similarity(transform: dict[str, float], points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    out_x = transform["a"] * x + transform["c"] * y + transform["e"]
    out_y = transform["b"] * x + transform["d"] * y + transform["f"]
    return np.stack([out_x, out_y], axis=1)


def torso_quad(landmarks, width: int, height: int) -> np.ndarray | None:
    """Pixel corners: shoulder 11, shoulder 12, hip 24, hip 23.

    Hips below the visibility floor are placed one shoulder-width below
    the shoulders. The quad then shrinks toward its centre. Returns None
    when either shoulder is too uncertain or the box is under 40 px.
    """
    if landmarks is None or len(landmarks) < 25:
        return None
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    if landmark_value(left_shoulder, "visibility", 0.0) < SHOULDER_VISIBILITY:
        return None
    if landmark_value(right_shoulder, "visibility", 0.0) < SHOULDER_VISIBILITY:
        return None

    def pixel(landmark) -> np.ndarray:
        return np.array(
            [landmark_value(landmark, "x") * width, landmark_value(landmark, "y") * height],
            dtype=np.float64,
        )

    shoulder_left = pixel(left_shoulder)
    shoulder_right = pixel(right_shoulder)
    shoulder_width = float(np.linalg.norm(shoulder_right - shoulder_left))
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    if (
        landmark_value(left_hip, "visibility", 0.0) < HIP_VISIBILITY
        or landmark_value(right_hip, "visibility", 0.0) < HIP_VISIBILITY
    ):
        hip_left = shoulder_left + np.array([0.0, shoulder_width])
        hip_right = shoulder_right + np.array([0.0, shoulder_width])
    else:
        hip_left = pixel(left_hip)
        hip_right = pixel(right_hip)
    corners = np.stack([shoulder_left, shoulder_right, hip_right, hip_left])
    centre = corners.mean(axis=0)
    shrunk = centre + (corners - centre) * (1.0 - TORSO_SHRINK)
    box_w = float(shrunk[:, 0].max() - shrunk[:, 0].min())
    box_h = float(shrunk[:, 1].max() - shrunk[:, 1].min())
    if box_w < MIN_TORSO_PX or box_h < MIN_TORSO_PX:
        return None
    return shrunk
