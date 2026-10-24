"""Torso colour histogram. This is the shirt recogniser, not the segmenter."""

from __future__ import annotations

import numpy as np

from face_shirt.colour import bgr_to_hsv_unit
from face_shirt.config import HISTOGRAM_BINS, MAX_SHIRT_SAMPLES
from face_shirt.segment import rasterize_convex_quad


def _sample(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    area = int(mask.sum())
    if area == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    stride = max(1, int(np.ceil(np.sqrt(area / MAX_SHIRT_SAMPLES))))
    ys, xs = np.nonzero(mask[::stride, ::stride])
    return ys * stride, xs * stride


def shirt_histogram(bgr: np.ndarray, quad: np.ndarray) -> dict | None:
    """28-bin HSV histogram of the torso, normalised to sum to 1.

    Grey bins 0–3 hold pixels with saturation under 0.2 or value under
    0.15, split by brightness. Other pixels use 12 hue bins times 2
    saturation bins. `swatch` is the mean colour of the fullest bin.
    """
    height, width = bgr.shape[:2]
    inside = rasterize_convex_quad(height, width, quad)
    ys, xs = _sample(inside)
    if len(xs) == 0:
        return None
    pixels = bgr[ys, xs]
    hue, saturation, value = bgr_to_hsv_unit(pixels)
    grey = (saturation < 0.2) | (value < 0.15)
    bins = np.zeros(len(xs), dtype=np.int32)
    grey_bin = np.zeros(len(xs), dtype=np.int32)
    grey_bin[value >= 0.25] = 1
    grey_bin[value >= 0.5] = 2
    grey_bin[value >= 0.75] = 3
    chromatic = 4 + np.floor(hue / 30.0).astype(np.int32) * 2 + (saturation >= 0.5).astype(np.int32)
    bins[grey] = grey_bin[grey]
    bins[~grey] = chromatic[~grey]
    histogram = np.bincount(bins, minlength=HISTOGRAM_BINS).astype(np.float64)
    total = float(histogram.sum())
    if total <= 0.0:
        return None
    histogram /= total
    fullest = int(np.argmax(histogram))
    chosen = pixels[bins == fullest]
    mean = chosen.mean(axis=0)
    # BGR mean -> #rrggbb
    red, green, blue = int(round(mean[2])), int(round(mean[1])), int(round(mean[0]))
    swatch = f"#{red:02x}{green:02x}{blue:02x}"
    return {"histogram": histogram.astype(np.float32), "swatch": swatch}


def histogram_similarity(left, right) -> float:
    """Bhattacharyya coefficient."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(np.sum(np.sqrt(left * right)))


def _normalise(histogram: np.ndarray) -> np.ndarray:
    total = float(np.sum(histogram))
    if total <= 0.0:
        return histogram.astype(np.float32)
    return (histogram / total).astype(np.float32)


def average_histograms(histograms) -> np.ndarray:
    stacked = np.stack([np.asarray(item, dtype=np.float64) for item in histograms])
    return _normalise(stacked.mean(axis=0))


def blend_histograms(old, fresh, weight: float) -> np.ndarray:
    mixed = (1.0 - weight) * np.asarray(old, dtype=np.float64) + weight * np.asarray(fresh, dtype=np.float64)
    return _normalise(mixed)
