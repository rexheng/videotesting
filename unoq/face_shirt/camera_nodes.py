"""Pick the USB camera. The first video nodes on a UNO Q are not cameras."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoNode:
    path: str
    card: str
    text: str


def parse_v4l2(path: str, text: str, card: str = "") -> VideoNode:
    found = card
    for line in text.splitlines():
        if "Card type" in line and ":" in line:
            found = line.split(":", 1)[1].strip()
    return VideoNode(path=path, card=found, text=text)


def _blob(node: VideoNode) -> str:
    return f"{node.card}\n{node.text}".lower()


def is_codec(node: VideoNode) -> bool:
    blob = _blob(node)
    return "venus" in blob or "encoder" in blob or "decoder" in blob


def is_metadata(node: VideoNode) -> bool:
    return "metadata" in _blob(node)


def is_infrared(node: VideoNode) -> bool:
    blob = _blob(node)
    return "infrared" in blob or " ir " in f" {blob} "


def is_capture(node: VideoNode) -> bool:
    if is_codec(node) or is_metadata(node):
        return False
    if "Video Capture" in node.text:
        return True
    if "Video Output" in node.text:
        return False
    # A sysfs name with no capability block is still a candidate when it
    # is not the Qualcomm codec.
    return True


def _index(node: VideoNode) -> int:
    name = node.path.rsplit("/", 1)[-1]
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 999


def choose_capture_node(nodes: list[VideoNode]) -> str | None:
    usable = [node for node in nodes if is_capture(node)]
    colour = [node for node in usable if not is_infrared(node)]
    pool = colour or usable
    if not pool:
        return None
    return sorted(pool, key=_index)[0].path


def camera_error(nodes: list[VideoNode]) -> str:
    codecs = [node.path for node in nodes if is_codec(node)]
    codec_text = ", ".join(codecs) if codecs else "/dev/video0 and /dev/video1"
    return (
        "No camera found. Plug a USB camera into a powered USB-C hub, then try again. "
        f"On the UNO Q, {codec_text} belong to the Qualcomm Venus codec, not the webcam."
    )
