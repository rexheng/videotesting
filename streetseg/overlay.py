"""Waymo-style frame: class colors, boxes, and the line the earpiece would say."""

from __future__ import annotations

import numpy as np

from .labels import is_light
from .mapillary import paint_bgr
from .types import Box, Cue

_BANNER = 64


def draw_overlay(
    bgr: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    boxes: list[Box],
    cue: Cue,
    role: str = "",
    status: str = "",
) -> np.ndarray:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected a BGR frame")
    frame = bgr
    if labels.shape[:2] != frame.shape[:2]:
        labels = _nearest(labels, frame.shape[:2])
    color = np.zeros_like(frame)
    for idx, name in names.items():
        paint = paint_bgr(name)
        if paint is None:
            continue
        color[labels == idx] = paint
    out = frame.astype(np.float32) * 0.55 + color.astype(np.float32) * 0.45
    out = np.clip(out, 0, 255).astype(np.uint8)
    for box in boxes:
        _rectangle(out, box)
    _banner(out, cue, role, status)
    for box in boxes:
        _light_label(out, box)
    _legend(out, names)
    return out


def draw_chest(bgr: np.ndarray, boxes: list[Box], speech: str, stop: bool) -> np.ndarray:
    out = bgr.copy()
    for box in boxes:
        _rectangle(out, box)
    out[:_BANNER, :] = (18, 18, 22)
    if stop:
        out[:4, :] = (30, 30, 210)
    _text(out, speech or "Chest clear.", 16, 40)
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


def _banner(image: np.ndarray, cue: Cue, role: str = "", status: str = "") -> None:
    image[:_BANNER, :] = (18, 18, 22)
    if cue.direction == "stop":
        image[:4, :] = (30, 30, 210)
    _glyph(image, cue.direction)
    if status:
        _text(image, status, 62, 24, scale=0.46, margin=96 if role else 12)
        _text(image, cue.speech, 62, 50, scale=0.55)
    else:
        _text(image, cue.speech, x=62, y=40)
    if role:
        _text(image, role, max(62, image.shape[1] - 88), 22, scale=0.45)


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


def _legend(image: np.ndarray, names: dict[int, str] | None = None) -> None:
    try:
        import cv2
    except ImportError:
        return
    known = set(names.values()) if names else set()
    items = []
    for caption, options in (
        ("walk", ("Sidewalk", "sidewalk")),
        ("crossing", ("Crosswalk - Plain", "Lane Marking - Crosswalk")),
        ("person", ("Person", "person")),
        ("vehicle", ("Car", "car")),
        ("signal", ("Traffic Light", "traffic light")),
    ):
        name = next((option for option in options if option in known or not known), None)
        if known and name not in known:
            continue
        items.append((name, caption))
    bar = 28
    if image.shape[0] <= _BANNER + bar:
        return
    y1 = image.shape[0] - bar
    image[y1:, :] = (18, 18, 22)
    x = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    for name, caption in items:
        if x + 80 >= image.shape[1]:
            break
        swatch = paint_bgr(name)
        if swatch is None:
            continue
        image[y1 + 7:y1 + 21, x:x + 16] = swatch
        cv2.putText(image, caption, (x + 22, y1 + 18), font, scale, (245, 245, 245), 1, cv2.LINE_AA)
        (tw, _th), _ = cv2.getTextSize(caption, font, scale, 1)
        x += 22 + tw + 18


def _light_label(image: np.ndarray, box: Box) -> None:
    caption = _light_caption(box)
    if not caption:
        return
    try:
        import cv2
    except ImportError:
        return
    height, width = image.shape[:2]
    x1, y1, _x2, _y2 = (int(round(v)) for v in box.xyxy)
    x1 = min(max(0, x1), width - 1)
    y1 = min(max(0, y1), height - 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    (tw, th), baseline = cv2.getTextSize(caption, font, scale, 1)
    pad = 3
    if y1 - th - baseline - pad > _BANNER:
        text_y = y1 - pad
    else:
        text_y = min(height - baseline - pad, max(_BANNER + th + pad, y1 + th + pad))
    left = min(x1, max(0, width - tw - 2 * pad))
    right = min(width, left + tw + 2 * pad)
    top = max(_BANNER, text_y - th - pad)
    bottom = min(height, text_y + baseline + pad)
    if bottom <= top or right <= left:
        return
    image[top:bottom, left:right] = (18, 18, 22)
    cv2.putText(
        image,
        caption,
        (left + pad, min(text_y, bottom - baseline)),
        font,
        scale,
        _box_color(box),
        1,
        cv2.LINE_AA,
    )


def _light_caption(box: Box) -> str:
    if box.source == "pvtl" and box.label:
        return box.label
    if box.signal and is_light(box.label):
        return f"{box.label} {box.signal}"
    return ""


def _text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.62,
    margin: int = 12,
) -> None:
    try:
        import cv2
    except ImportError:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_width = image.shape[1] - x - margin
    floor = 0.28
    while scale > floor and max_width > 0:
        (tw, _th), _ = cv2.getTextSize(text, font, scale, 1)
        if tw <= max_width:
            break
        scale -= 0.04
    cv2.putText(image, text, (x, y), font, scale, (245, 245, 245), 1, cv2.LINE_AA)
