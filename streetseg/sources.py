"""Frame sources. The Tello source only turns the camera on. It never flies."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VideoFileSource:
    def __init__(self, path: str | Path):
        import cv2

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self._cv2 = cv2

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    @property
    def fps(self) -> float:
        value = float(self.cap.get(self._cv2.CAP_PROP_FPS) or 0.0)
        return value if value > 1 else 20.0

    @property
    def size(self) -> tuple[int, int]:
        width = int(self.cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def close(self) -> None:
        self.cap.release()


class WebcamSource:
    def __init__(self, index: int = 0):
        import cv2

        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {index}")

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        self.cap.release()


class TelloSource:
    """DJI Tello video via djitellopy. Connects and calls streamon only."""

    def __init__(self):
        from djitellopy import Tello

        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.reader = self.tello.get_frame_read()

    def read(self) -> np.ndarray | None:
        frame = self.reader.frame
        if frame is None:
            return None
        return frame.copy()

    def close(self) -> None:
        try:
            self.tello.streamoff()
        finally:
            self.tello.end()
