"""Shared records. The voice and glasses loops should depend on Cue only."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """One detector box. xyxy is in pixels, x2/y2 exclusive."""

    label: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None
    signal: str | None = None
    source: str = "drone"

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass(frozen=True)
class ColumnScore:
    name: str
    walkable: float
    blocked: float
    people_pixels: float
    people: int
    vehicles: int
    score: float
    adjusted: float
    selectable: bool


@dataclass(frozen=True)
class Cue:
    """What the AirPods should say, and what the glasses pads should do.

    ``turn`` is the wearer's left/right, matching the ReachGlass glasses
    contract: -1 left, 0 straight or stop, +1 right. The firmware crosses
    the temple pads itself, so this value is not pre-crossed. ``stop`` is
    separate: a stop must not nudge either pad.
    """

    direction: str
    speech: str
    reason: str
    crowd: str
    signal: str
    crosswalk: bool
    hazards: tuple[str, ...] = ()
    columns: tuple[ColumnScore, ColumnScore, ColumnScore] | None = None
    wearer_excluded: bool = False

    @property
    def stop(self) -> bool:
        return self.direction == "stop"

    @property
    def turn(self) -> int:
        if self.direction == "left":
            return -1
        if self.direction == "right":
            return 1
        return 0

    def to_dict(self) -> dict:
        columns = {}
        if self.columns is not None:
            for col in self.columns:
                columns[col.name] = {
                    "walkable": round(col.walkable, 3),
                    "blocked": round(col.blocked, 3),
                    "people": col.people,
                    "vehicles": col.vehicles,
                    "score": round(col.score, 3),
                    "selectable": col.selectable,
                }
        return {
            "direction": self.direction,
            "speech": self.speech,
            "reason": self.reason,
            "crowd": self.crowd,
            "signal": self.signal,
            "crosswalk": self.crosswalk,
            "hazards": list(self.hazards),
            "turn": self.turn,
            "stop": self.stop,
            "wearer_excluded": self.wearer_excluded,
            "columns": columns,
        }


@dataclass
class FrameResult:
    cue: Cue
    labels: np.ndarray
    boxes: list[Box] = field(default_factory=list)
    body_boxes: list[Box] = field(default_factory=list)
    image: np.ndarray | None = None
