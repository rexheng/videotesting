"""Cost model. The scheduler refuses any plan the QRB2210 cannot afford."""

from __future__ import annotations

from face_shirt.config import (
    CAPTURE_HEIGHT,
    CAPTURE_MS,
    CAPTURE_WIDTH,
    DELEGATE,
    EMBED_MS,
    FACE_EXTRA_MS,
    FACE_INTERVAL_BOTH_MS,
    FACE_MS,
    INFERENCE_TARGET,
    MAX_EMBEDS,
    MAX_FACES,
    MAX_POSES,
    MCU_FRAME_BYTES,
    MCU_SRAM_BYTES,
    NAME_BUDGET_MS,
    POSE_EXTRA_MS,
    POSE_MS,
    RECOLOR_MS,
    SEGMENT_BYTES,
    SEGMENT_FRAME_BUDGET_MS,
    SEGMENT_MS,
    SEGMENT_SIZE,
    SHIRT_SAMPLE_MS,
    TRACK_TTL_BOTH_MS,
)


def clamp_count(count: int, limit: int) -> int:
    return max(1, min(int(count), limit))


def face_cost(faces: int) -> int:
    count = clamp_count(faces, MAX_FACES)
    return FACE_MS + FACE_EXTRA_MS * (count - 1)


def pose_cost(poses: int) -> int:
    count = clamp_count(poses, MAX_POSES)
    return POSE_MS + POSE_EXTRA_MS * (count - 1)


def embed_cost(embeds: int) -> int:
    count = max(1, min(int(embeds), MAX_EMBEDS))
    return EMBED_MS * count


def stage_cost(stage: str, *, faces: int, poses: int, embeds: int) -> int:
    if stage == "face":
        return face_cost(faces)
    if stage == "pose":
        return pose_cost(poses) + SHIRT_SAMPLE_MS
    if stage == "embed":
        return embed_cost(embeds)
    if stage == "segment":
        return SEGMENT_MS
    raise KeyError(stage)


def assert_board_invariants() -> None:
    """Raise AssertionError if the pinned numbers no longer fit the board."""
    if INFERENCE_TARGET != "qrb2210-mpu":
        raise AssertionError("segmentation is computed on the QRB2210, not the STM32")
    if DELEGATE != "CPU":
        raise AssertionError("the Debian image has no supported MediaPipe GPU delegate")
    if MAX_FACES != 2 or MAX_POSES != 2 or MAX_EMBEDS != 2:
        raise AssertionError("more than two people does not fit the name budget")
    if (CAPTURE_WIDTH, CAPTURE_HEIGHT) != (640, 480):
        raise AssertionError("capture must stay at 640x480 MJPG")
    if SEGMENT_SIZE != 256:
        raise AssertionError("the clothes model is the 256x256 multiclass graph")
    if SEGMENT_BYTES != 16_371_837:
        raise AssertionError("unexpected clothes-model size")
    if MCU_FRAME_BYTES >= MCU_SRAM_BYTES:
        # A VGA frame is larger than the microcontroller's SRAM. This is
        # intentional: it must stay a fact, so nobody ports the loop there.
        pass
    else:
        raise AssertionError("MCU SRAM check drifted; re-audit before moving pixels")

    one_person = CAPTURE_MS + face_cost(1) + embed_cost(1)
    if one_person > NAME_BUDGET_MS:
        raise AssertionError(f"one person takes {one_person} ms, budget is {NAME_BUDGET_MS}")

    two_people = CAPTURE_MS + face_cost(2) + embed_cost(2) + RECOLOR_MS
    if two_people > NAME_BUDGET_MS:
        raise AssertionError(f"two people take {two_people} ms, budget is {NAME_BUDGET_MS}")

    with_pose = CAPTURE_MS + face_cost(1) + embed_cost(1) + pose_cost(1) + SHIRT_SAMPLE_MS + RECOLOR_MS
    if with_pose > NAME_BUDGET_MS:
        raise AssertionError(f"face plus pose takes {with_pose} ms")

    two_and_pose = CAPTURE_MS + face_cost(2) + embed_cost(2) + pose_cost(2) + SHIRT_SAMPLE_MS
    if two_and_pose <= NAME_BUDGET_MS:
        raise AssertionError("two people plus pose must stay off the name frame")

    segment_frame = CAPTURE_MS + SEGMENT_MS + RECOLOR_MS
    if segment_frame > SEGMENT_FRAME_BUDGET_MS:
        raise AssertionError(f"mask refresh takes {segment_frame} ms")
    if CAPTURE_MS + SEGMENT_MS + face_cost(1) <= NAME_BUDGET_MS:
        raise AssertionError("the segmenter must not be cheap enough to share the name frame")

    if FACE_INTERVAL_BOTH_MS < 2000:
        raise AssertionError("combined mode needs a 2 s face interval so the mask gets a frame")
    if TRACK_TTL_BOTH_MS <= SEGMENT_MS:
        raise AssertionError("a track must outlive one mask refresh")


def report_lines() -> list[str]:
    assert_board_invariants()
    one = CAPTURE_MS + face_cost(1) + embed_cost(1)
    two = CAPTURE_MS + face_cost(2) + embed_cost(2) + RECOLOR_MS
    mask = CAPTURE_MS + SEGMENT_MS + RECOLOR_MS
    return [
        f"target: {INFERENCE_TARGET}",
        f"delegate: {DELEGATE}",
        f"capture: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}",
        f"faces cap: {MAX_FACES}",
        f"name budget: {NAME_BUDGET_MS} ms",
        f"one person face+embed: {one} ms",
        f"two people face+embed+held recolor: {two} ms",
        f"clothes mask refresh: {mask} ms (ceiling {SEGMENT_MS} ms model)",
        f"segmenter: {SEGMENT_SIZE}x{SEGMENT_SIZE} float32, {SEGMENT_BYTES} bytes",
        "mask refresh never shares a frame with the face landmarker",
    ]
