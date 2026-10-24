"""Text the STM32 may display. Frames and masks stay on the MPU."""

from __future__ import annotations

from face_shirt.config import BRIDGE_TEXT_MAX, MCU_FRAME_BYTES, MCU_SRAM_BYTES


def bridge_text(label: str) -> str:
    text = " ".join(label.split())
    if len(text) > BRIDGE_TEXT_MAX:
        text = text[: BRIDGE_TEXT_MAX - 1] + "…"
    return text


def frame_fits_on_mcu() -> bool:
    return MCU_FRAME_BYTES < MCU_SRAM_BYTES
