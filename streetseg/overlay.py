"""Waymo-style frame: class colors, boxes, and the cue the wearer hears."""

from __future__ import annotations

import numpy as np

from .labels import PALETTE
from .types import Box, Cue

_BANNER = 64


def draw_overlay(
    bgr: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    boxes: list[Box],
    cue: Cue,
) -> np.ndarray:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected a BGR frame")
    frame = bgr
    if labels.shape[:2] != frame.shape[:2]:
        labels = _nearest(labels, frame.shape[:2])
    color = np.zeros_like(frame)
    for idx, name in names.items():
        paint = PALETTE.get(name)
        if paint is None:
            continue
        color[labels == idx] = paint
    out = (frame.astype(np.float32) * 0.55 + color.astype(np.float32) * 0.45)
    out = np.clip(out, 0, 255).astype(np.uint8)
    for box in boxes:
        _rectangle(out, box)
    _banner(out, cue)
    _legend(out)
    return out


def _nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    ys = np.linspace(0, mask.shape[0] - 1, height).astype(np.int32)
    xs = np.linspace(0, mask.shape[1] - 1, width).astype(np.int32)
    return mask[ys][:, xs]


def _rectangle(image: np.ndarray, box: Box) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy)
    x1, y1 = max(0, x1), max(_BANNER, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    color = _box_color(box)
    t = 2
    image[y1:y1 + t, x1:x2] = color
    image[max(y1, y2 - t):y2, x1:x2] = color
    image[y1:y2, x1:x1 + t] = color
    image[y1:y2, max(x1, x2 - t):x2] = color


def _box_color(box: Box) -> tuple[int, int, int]:
    if box.signal == "red":
        return (40, 40, 230)
    if box.signal == "green":
        return (40, 200, 40)
    if box.signal == "yellow":
        return (0, 220, 255)
    name = box.label.lower()
    if "person" in name or "pedestrian" in name:
        return (220, 80, 240)
    if any(word in name for word in ("car", "truck", "bus", "vehicle")):
        return (40, 40, 230)
    return (255, 220, 0)


def _banner(image: np.ndarray, cue: Cue) -> None:
    image[:_BANNER, :] = (18, 18, 22)
    _glyph(image, cue.direction)
    _text(image, cue.speech, x=62, y=38)


def _glyph(image: np.ndarray, direction: str) -> None:
    if direction == "stop":
        image[14:50, 14:50] = (30, 30, 210)
        image[22:42, 22:42] = (245, 245, 245)
        return
    color = (245, 245, 245)
    if direction == "straight":
        for y in range(12, 52):
            half = max(0, 18 - abs(y - 18))
            if y < 30:
                image[y, 32 - half:32 + half] = color
            else:
                image[y, 26:38] = color
        return
    point_left = direction == "left"
    for y in range(14, 50):
        inset = abs(y - 32)
        if point_left:
            image[y, 12 + inset:46] = color
        else:
            image[y, 14:48 - inset] = color


def _legend(image: np.ndarray) -> None:
    items = ("road", "sidewalk", "person", "car", "vegetation", "traffic light")
    y1 = image.shape[0] - 18
    y2 = image.shape[0] - 6
    if y1 <= _BANNER:
        return
    x = 8
    for name in items:
        image[y1:y2, x:x + 16] = PALETTE[name]
        x += 22
        if x > image.shape[1] - 20:
            break


def _text(image: np.ndarray, text: str, x: int, y: int) -> None:
    try:
        import cv2
    except ImportError:
        return
    scale = 0.62
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_width = image.shape[1] - x - 12
    while scale > 0.35:
        (tw, _th), _ = cv2.getTextSize(text, font, scale, 1)
        if tw <= max_width:
            break
        scale -= 0.05
    cv2.putText(image, text, (x, y), font, scale, (245, 245, 245), 1, cv2.LINE_AA)
