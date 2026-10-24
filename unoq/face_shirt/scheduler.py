"""Choose the models for one camera frame.

The clothes segmenter and the face landmarker never run in the same frame.
Pose is added to a face frame only when one person was seen last time and
the sum still fits in the one-second name budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from face_shirt.budget import embed_cost, face_cost, pose_cost
from face_shirt.config import (
    CAPTURE_MS,
    FACE_INTERVAL_BOTH_MS,
    FACE_INTERVAL_MS,
    FRAME_HARD_MS,
    MAX_EMBEDS,
    MAX_FACES,
    MAX_POSES,
    NAME_BUDGET_MS,
    POSE_INTERVAL_MS,
    RECOLOR_MS,
    SEGMENT_FRAME_BUDGET_MS,
    SEGMENT_INTERVAL_MS,
    SEGMENT_MS,
    SHIRT_SAMPLE_MS,
)


@dataclass
class BoardState:
    mode: str = "recolor"
    now_ms: int = 0
    identified: bool = False
    pending_embed: bool = False
    embeds_wanted: int = 1
    faces_seen: int = 0
    poses_seen: int = 0
    last_face_ms: int = -10**9
    last_pose_ms: int = -10**9
    last_segment_ms: int = -10**9
    last_frame_ms: int = 0
    has_mask: bool = False
    enrolment_needs: str | None = None


@dataclass(frozen=True)
class FramePlan:
    stages: tuple[str, ...]
    reason: str
    estimated_ms: int
    limit_ms: int


def face_interval_ms(state: BoardState) -> int:
    if state.mode == "both":
        return FACE_INTERVAL_BOTH_MS
    return FACE_INTERVAL_MS


def _recoloring(state: BoardState) -> bool:
    return state.enrolment_needs is None and state.mode in ("recolor", "both")


def _faces(state: BoardState) -> int:
    return max(1, min(state.faces_seen or 1, MAX_FACES))


def _poses(state: BoardState) -> int:
    hinted = state.poses_seen or state.faces_seen or 1
    return max(1, min(hinted, MAX_POSES))


def _embeds(state: BoardState) -> int:
    return max(1, min(state.embeds_wanted or 1, MAX_EMBEDS))


def estimated_ms(stages: tuple[str, ...] | list[str], state: BoardState) -> int:
    total = CAPTURE_MS
    stage_set = set(stages)
    if "face" in stage_set:
        total += face_cost(_faces(state))
    if "embed" in stage_set:
        total += embed_cost(_embeds(state))
    if "pose" in stage_set:
        total += pose_cost(_poses(state)) + SHIRT_SAMPLE_MS
    if "segment" in stage_set:
        total += SEGMENT_MS
    if _recoloring(state):
        total += RECOLOR_MS
    return total


def plan_frame(state: BoardState) -> FramePlan:
    if state.last_frame_ms > FRAME_HARD_MS:
        estimate = estimated_ms((), state)
        return FramePlan((), "cooldown", estimate, NAME_BUDGET_MS)

    if state.enrolment_needs == "pose":
        stages = ("pose",)
        estimate = estimated_ms(stages, state)
        if estimate > NAME_BUDGET_MS:
            raise AssertionError(f"pose enrolment frame is {estimate} ms")
        return FramePlan(stages, "pose", estimate, NAME_BUDGET_MS)

    face_due = False
    if state.enrolment_needs == "face":
        face_due = True
    elif state.mode in ("tag", "both"):
        age = state.now_ms - state.last_face_ms
        face_due = state.pending_embed or state.last_face_ms < 0 or age >= face_interval_ms(state)

    segment_due = (
        state.enrolment_needs is None
        and state.mode in ("recolor", "both")
        and (not state.has_mask or state.now_ms - state.last_segment_ms >= SEGMENT_INTERVAL_MS)
    )
    pose_due = (
        state.enrolment_needs is None
        and state.mode in ("tag", "both")
        and state.now_ms - state.last_pose_ms >= POSE_INTERVAL_MS
    )

    if face_due:
        stages = ["face", "embed"]
        # Only attach pose when the previous frame actually saw a single
        # person. An unknown count, or two people, would blow the name budget.
        if pose_due and state.faces_seen == 1:
            with_pose = stages + ["pose"]
            if estimated_ms(with_pose, state) <= NAME_BUDGET_MS:
                stages = with_pose
        estimate = estimated_ms(stages, state)
        if estimate > NAME_BUDGET_MS:
            raise AssertionError(f"identify frame is {estimate} ms")
        if "segment" in stages:
            raise AssertionError("face and clothes segmentation shared a frame")
        return FramePlan(tuple(stages), "identify", estimate, NAME_BUDGET_MS)

    if segment_due:
        stages = ("segment",)
        estimate = estimated_ms(stages, state)
        if estimate > SEGMENT_FRAME_BUDGET_MS:
            raise AssertionError(f"segment frame is {estimate} ms")
        return FramePlan(stages, "segment", estimate, SEGMENT_FRAME_BUDGET_MS)

    if pose_due:
        stages = ("pose",)
        estimate = estimated_ms(stages, state)
        if estimate > NAME_BUDGET_MS:
            raise AssertionError(f"pose frame is {estimate} ms")
        return FramePlan(stages, "pose", estimate, NAME_BUDGET_MS)

    estimate = estimated_ms((), state)
    return FramePlan((), "track", estimate, NAME_BUDGET_MS)
