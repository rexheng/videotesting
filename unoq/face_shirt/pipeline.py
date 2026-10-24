"""One camera frame: plan, run the allowed models, recolour from the held mask."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from face_shirt.config import (
    HUE_SHIFT_DEGREES,
    MAX_EMBEDS,
    RECOLOR_MS,
    TRACK_TTL_BOTH_MS,
    TRACK_TTL_MS,
)
from face_shirt.geometry import torso_quad, yaw_degrees
from face_shirt.scheduler import BoardState, FramePlan, plan_frame
from face_shirt.segment import recolor_clothes
from face_shirt.shirt import shirt_histogram


@dataclass
class StepResult:
    plan: FramePlan
    frame: np.ndarray
    faces: int
    yaw: float | None
    segmented: bool
    recolored: bool
    swatch: str | None
    spent_ms: int


@dataclass
class Pipeline:
    runners: object
    mode: str = "recolor"
    hue_degrees: float = HUE_SHIFT_DEGREES
    state: BoardState = field(init=False)
    mask: np.ndarray | None = None
    mask_ms: int = -10**9
    quad: np.ndarray | None = None
    quad_ms: int = -10**9
    last_swatch: str | None = None
    last_faces: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = BoardState(mode=self.mode)

    def quad_ttl(self) -> int:
        if self.mode == "both":
            return TRACK_TTL_BOTH_MS
        return TRACK_TTL_MS

    def fresh_quad(self, now_ms: int) -> np.ndarray | None:
        if self.quad is None:
            return None
        if now_ms - self.quad_ms > self.quad_ttl():
            return None
        return self.quad

    def step(self, frame: np.ndarray, now_ms: int, capture_ms: int | None = None) -> StepResult:
        self.state.mode = self.mode
        self.state.now_ms = now_ms
        self.state.has_mask = self.mask is not None
        plan = plan_frame(self.state)
        started = time.perf_counter()
        faces = self.state.faces_seen
        yaw = None
        segmented = False
        swatch = self.last_swatch
        height, width = frame.shape[:2]

        if "face" in plan.stages:
            landmarks = self.runners.face(frame, now_ms) or []
            self.last_faces = list(landmarks)
            faces = len(landmarks)
            self.state.faces_seen = faces
            self.state.last_face_ms = now_ms
            if landmarks:
                try:
                    yaw = yaw_degrees(landmarks[0])
                except (IndexError, TypeError, KeyError):
                    yaw = None
        if "embed" in plan.stages:
            embed_one = getattr(self.runners, "embed_landmarks", None)
            if embed_one is not None and self.last_faces:
                for face in self.last_faces[:MAX_EMBEDS]:
                    embed_one(frame, face)
            else:
                self.runners.embed(frame, now_ms)
            self.state.pending_embed = False
        if "pose" in plan.stages:
            pose_landmarks = self.runners.pose(frame, now_ms) or []
            self.state.poses_seen = len(pose_landmarks)
            self.state.last_pose_ms = now_ms
            if pose_landmarks:
                quad = torso_quad(pose_landmarks[0], width, height)
                if quad is not None:
                    self.quad = quad
                    self.quad_ms = now_ms
                    described = shirt_histogram(frame, quad)
                    if described is not None:
                        swatch = described["swatch"]
                        self.last_swatch = swatch
        if "segment" in plan.stages:
            raw_mask = self.runners.segment(frame, now_ms)
            self.mask = np.asarray(raw_mask)
            self.mask_ms = now_ms
            self.state.last_segment_ms = now_ms
            self.state.has_mask = True
            segmented = True

        output = frame
        recolored = False
        quad = self.fresh_quad(now_ms)
        if self.mode in ("recolor", "both") and self.state.enrolment_needs is None and self.mask is not None:
            output = recolor_clothes(frame, self.mask, self.hue_degrees, quad)
            recolored = True

        measured = int((time.perf_counter() - started) * 1000)
        extra = 0 if capture_ms is None else capture_ms
        # Tests inject model costs. A real run uses the wall clock, which
        # already includes the stages above, plus the capture time.
        injected = 0
        cost_of = getattr(self.runners, "cost_ms", None)
        if cost_of is not None:
            for stage in plan.stages:
                override = cost_of(stage)
                if override is not None:
                    injected += int(override)
        if injected:
            spent = (0 if capture_ms is None else capture_ms) + injected
            if recolored:
                recolor_cost = cost_of("recolor") if cost_of is not None else None
                spent += RECOLOR_MS if recolor_cost is None else int(recolor_cost)
        else:
            spent = measured + extra
        self.state.last_frame_ms = spent
        return StepResult(
            plan=plan,
            frame=output,
            faces=faces,
            yaw=yaw,
            segmented=segmented,
            recolored=recolored,
            swatch=swatch,
            spent_ms=spent,
        )
