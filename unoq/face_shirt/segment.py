"""Clothes mask, torso gate, and hue shift.

The multiclass model paints every clothes pixel with class 4. The shift is
limited to the pose torso when that quad is fresh, so one person's shirt
can change without recolouring someone else's.
"""

from __future__ import annotations

import numpy as np

from face_shirt.colour import shift_hue_bgr
from face_shirt.config import CLASS_CLOTHES


def category_mask_array(mask) -> np.ndarray:
    """Accept a numpy mask or a MediaPipe image and return uint8 HxW."""
    if hasattr(mask, "numpy_view"):
        array = np.asarray(mask.numpy_view())
    else:
        array = np.asarray(mask)
    if array.ndim == 3:
        array = array[:, :, 0]
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = category_mask_array(mask)
    if mask.shape == (height, width):
        return mask
    y_index = (np.arange(height) * mask.shape[0] / height).astype(np.int32)
    x_index = (np.arange(width) * mask.shape[1] / width).astype(np.int32)
    y_index = np.clip(y_index, 0, mask.shape[0] - 1)
    x_index = np.clip(x_index, 0, mask.shape[1] - 1)
    return mask[y_index][:, x_index]


def rasterize_convex_quad(height: int, width: int, quad: np.ndarray) -> np.ndarray:
    """True for pixels inside a convex four-point polygon."""
    corners = np.asarray(quad, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError("a torso quad has four (x, y) corners")
    centre = corners.mean(axis=0)
    start = corners[0]
    end = corners[1]
    centre_cross = (end[0] - start[0]) * (centre[1] - start[1]) - (end[1] - start[1]) * (
        centre[0] - start[0]
    )
    sign = 1.0 if centre_cross >= 0 else -1.0
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    inside = np.ones((height, width), dtype=bool)
    for index in range(4):
        edge_start = corners[index]
        edge_end = corners[(index + 1) % 4]
        cross = (edge_end[0] - edge_start[0]) * (y_grid - edge_start[1]) - (
            edge_end[1] - edge_start[1]
        ) * (x_grid - edge_start[0])
        inside &= sign * cross >= -1.0
    return inside


def recolor_clothes(
    bgr: np.ndarray,
    category_mask,
    hue_degrees: float,
    quad: np.ndarray | None = None,
) -> np.ndarray:
    """Shift the hue of clothes pixels. Brightness and saturation stay."""
    height, width = bgr.shape[:2]
    mask = resize_mask_nearest(category_mask, height, width)
    clothes = mask == CLASS_CLOTHES
    if quad is not None:
        clothes &= rasterize_convex_quad(height, width, quad)
    if not np.any(clothes):
        return np.array(bgr, copy=True)
    shifted = shift_hue_bgr(bgr, hue_degrees)
    output = np.array(bgr, copy=True)
    output[clothes] = shifted[clothes]
    return output
