"""Download the pinned model files and reject a file of the wrong size."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from face_shirt.config import (
    FACE_BYTES,
    FACE_FILE,
    FACE_URL,
    POSE_BYTES,
    POSE_FILE,
    POSE_URL,
    SEGMENT_BYTES,
    SEGMENT_FILE,
    SEGMENT_URL,
    SFACE_BYTES,
    SFACE_FILE,
    SFACE_URL,
)

FILES = (
    (SEGMENT_FILE, SEGMENT_URL, SEGMENT_BYTES),
    (FACE_FILE, FACE_URL, FACE_BYTES),
    (POSE_FILE, POSE_URL, POSE_BYTES),
    (SFACE_FILE, SFACE_URL, SFACE_BYTES),
)


def fetch_models(directory: Path, *, force: bool = False) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, url, expected in FILES:
        target = directory / name
        if target.is_file() and target.stat().st_size == expected and not force:
            written.append(target)
            continue
        temporary = target.with_suffix(target.suffix + ".partial")
        urllib.request.urlretrieve(url, temporary)
        size = temporary.stat().st_size
        if size != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"{name} is {size} bytes, expected {expected}. Left it uninstalled.")
        temporary.replace(target)
        written.append(target)
    return written
