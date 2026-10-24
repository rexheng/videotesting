"""BGR colour helpers shared by the shirt histogram and the hue shift."""

from __future__ import annotations

import numpy as np


def bgr_to_hsv_unit(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hue in degrees [0, 360), saturation and value in [0, 1]."""
    blue = bgr[..., 0].astype(np.float32)
    green = bgr[..., 1].astype(np.float32)
    red = bgr[..., 2].astype(np.float32)
    maxc = np.maximum(np.maximum(red, green), blue)
    minc = np.minimum(np.minimum(red, green), blue)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    nonzero = delta != 0
    red_max = nonzero & (maxc == red)
    green_max = nonzero & (maxc == green) & (maxc != red)
    blue_max = nonzero & (maxc == blue) & (maxc != red) & (maxc != green)
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6.0)
    hue[green_max] = ((blue[green_max] - red[green_max]) / delta[green_max]) + 2.0
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4.0
    hue_deg = np.mod(hue * 60.0, 360.0)
    saturation = np.zeros_like(maxc)
    visible = maxc != 0
    saturation[visible] = delta[visible] / maxc[visible]
    value = maxc / 255.0
    return hue_deg, saturation, value


def shift_hue_bgr(bgr: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate hue and keep saturation and value, so shading stays put."""
    hue, saturation, value = bgr_to_hsv_unit(bgr)
    hue = np.mod(hue + float(degrees), 360.0)
    return hsv_unit_to_bgr(hue, saturation, value)


def hsv_unit_to_bgr(hue_deg: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    value_255 = value * 255.0
    chroma = value_255 * saturation
    sector_pos = (hue_deg / 60.0) % 2.0
    x_value = chroma * (1.0 - np.abs(sector_pos - 1.0))
    match = value_255 - chroma
    sector = np.floor(np.mod(hue_deg, 360.0) / 60.0).astype(np.int32) % 6
    red = np.zeros_like(value_255)
    green = np.zeros_like(value_255)
    blue = np.zeros_like(value_255)
    pairs = (
        (chroma, x_value, 0.0),
        (x_value, chroma, 0.0),
        (0.0, chroma, x_value),
        (0.0, x_value, chroma),
        (x_value, 0.0, chroma),
        (chroma, 0.0, x_value),
    )
    for index, (red_part, green_part, blue_part) in enumerate(pairs):
        chosen = sector == index
        red[chosen] = _part(red_part, chosen)
        green[chosen] = _part(green_part, chosen)
        blue[chosen] = _part(blue_part, chosen)
    red = np.clip(np.rint(red + match), 0, 255)
    green = np.clip(np.rint(green + match), 0, 255)
    blue = np.clip(np.rint(blue + match), 0, 255)
    return np.stack([blue, green, red], axis=-1).astype(np.uint8)


def _part(part, chosen: np.ndarray):
    if isinstance(part, float):
        return np.full(int(chosen.sum()), part, dtype=np.float32)
    return part[chosen]
