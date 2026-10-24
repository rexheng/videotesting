"""Runs segmenter + detectors and holds the spoken direction steady.

Stop is immediate. A new left/right/straight has to agree for a few frames
so the AirPods do not chatter every time a person crosses a column boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fuse import FuseConfig, assess
from .overlay import draw_overlay
from .signal import paint_signal
from .types import Box, Cue, FrameResult


@dataclass
class DirectionHold:
    frames: int = 3
    committed: str | None = None
    pending: str | None = None
    count: int = 0

    def update(self, direction: str) -> str:
        if direction == "stop":
            self.committed = "stop"
            self.pending = None
            self.count = 0
            return "stop"
        if self.committed is None:
            self.committed = direction
            return direction
        if direction == self.committed:
            self.pending = None
            self.count = 0
            return self.committed
        if direction != self.pending:
            self.pending = direction
            self.count = 1
        else:
            self.count += 1
        if self.count >= self.frames:
            self.committed = direction
            self.pending = None
            self.count = 0
        return self.committed


class Orchestrator:
    def __init__(
        self,
        segmenter,
        detector,
        body_detector=None,
        sign_detector=None,
        depth_estimator=None,
        cfg: FuseConfig | None = None,
        hold_frames: int = 3,
        depth_stride: int = 4,
    ):
        self.segmenter = segmenter
        self.detector = detector
        self.body_detector = body_detector
        self.sign_detector = sign_detector
        self.depth_estimator = depth_estimator
        self.cfg = cfg or FuseConfig()
        self.hold = DirectionHold(hold_frames)
        self.depth_stride = max(1, depth_stride)
        self._last: Cue | None = None
        self._depth = None
        self._depth_i = 0

    def step(self, drone_bgr: np.ndarray, body_bgr: np.ndarray | None = None) -> FrameResult:
        labels = self.segmenter.predict(drone_bgr)
        if labels.shape[:2] != drone_bgr.shape[:2]:
            raise ValueError("segmenter returned a different frame size")
        names = _names(self.segmenter)
        boxes = [paint_signal(box, drone_bgr) for box in self.detector.predict(drone_bgr)]
        if self.sign_detector is not None:
            boxes.extend(paint_signal(box, drone_bgr) for box in self.sign_detector.predict(drone_bgr))
        body_boxes: list[Box] = []
        depth = None
        body_shape = None
        if body_bgr is not None:
            body_shape = body_bgr.shape[:2]
            detector = self.body_detector or self.detector
            body_boxes = list(detector.predict(body_bgr))
            depth = self._maybe_depth(body_bgr)
        cue = assess(
            labels,
            names,
            boxes,
            body_boxes=body_boxes,
            depth=depth,
            body_shape=body_shape,
            cfg=self.cfg,
        )
        shown = self.hold.update(cue.direction)
        if shown != cue.direction and self._last is not None and self._last.direction == shown:
            cue = self._last
        else:
            self._last = cue
        image = draw_overlay(drone_bgr, labels, names, boxes, cue)
        return FrameResult(cue, labels, boxes, body_boxes, image)

    def _maybe_depth(self, frame: np.ndarray):
        if self.depth_estimator is None:
            return None
        if self._depth is None or self._depth_i % self.depth_stride == 0:
            self._depth = self.depth_estimator.predict(frame)
        self._depth_i += 1
        return self._depth


def _names(segmenter) -> dict[int, str]:
    raw = segmenter.names
    if callable(raw):
        raw = raw()
    return {int(key): str(value) for key, value in dict(raw).items()}
