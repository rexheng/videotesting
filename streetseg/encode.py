"""Write overlay frames as H.264 so a browser or a phone can play them.

OpenCV's mp4v writer is not H.264. This pipes raw BGR into ffmpeg
(libx264, yuv420p, +faststart) and keeps every frame at the source rate.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class H264Writer:
    def __init__(self, path: str, width: int, height: int, fps: float, crf: int = 23):
        if width < 2 or height < 2 or width % 2 or height % 2:
            raise ValueError("yuv420p needs a positive even width and height")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 0 <= int(crf) <= 51:
            raise ValueError("crf must be from 0 to 51")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is not on PATH")
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.frames = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                f"{self.fps:.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(int(crf)),
                "-preset",
                "veryfast",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._closed or self._proc.stdin is None:
            raise RuntimeError("writer is closed")
        if frame.ndim != 3 or frame.shape[0] != self.height or frame.shape[1] != self.width or frame.shape[2] != 3:
            raise ValueError(f"expected {(self.height, self.width, 3)}, got {frame.shape}")
        if self._proc.poll() is not None:
            raise RuntimeError(self._failure())
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(self._failure()) from exc
        self.frames += 1

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            _stdout, stderr = self._proc.communicate(timeout=180)
        except subprocess.TimeoutExpired as exc:
            self._proc.kill()
            raise RuntimeError("ffmpeg timed out while finishing the mp4") from exc
        if self._proc.returncode != 0:
            tail = (stderr or b"").decode("utf-8", errors="replace")[-800:]
            raise RuntimeError(f"ffmpeg exited {self._proc.returncode}: {tail}")

    def close(self) -> None:
        self.release()

    def __enter__(self) -> H264Writer:
        return self

    def __exit__(self, *_args) -> None:
        self.release()

    def _failure(self) -> str:
        self._closed = True
        try:
            _stdout, stderr = self._proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            stderr = b""
        tail = (stderr or b"").decode("utf-8", errors="replace")[-800:]
        return f"ffmpeg exited {self._proc.returncode}: {tail}"
