import numpy as np

from streetseg.labels import CITYSCAPES, NAMES
from streetseg.overlay import draw_overlay
from streetseg.pipeline import DirectionHold, Orchestrator
from streetseg.scene import synthetic_street
from streetseg.types import Box


class _Script:
    def __init__(self, frames):
        self.frames = list(frames)
        self.names = dict(NAMES)

    def predict(self, _bgr):
        return self.frames.pop(0)


class _Boxes:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, _bgr):
        return list(self.boxes)


def _image(labels):
    image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    image[:, :] = (80, 90, 100)
    return image


def test_direction_holds_until_three_frames_agree_but_stop_is_immediate():
    hold = DirectionHold(frames=3)
    assert hold.update("left") == "left"
    assert hold.update("right") == "left"
    assert hold.update("right") == "left"
    assert hold.update("right") == "right"
    assert hold.update("stop") == "stop"
    assert hold.update("straight") == "stop"
    assert hold.update("straight") == "stop"
    assert hold.update("straight") == "straight"


def test_orchestrator_keeps_the_previous_cue_while_the_direction_settles():
    sidewalk = np.full((80, 60), CITYSCAPES.index("sidewalk"), dtype=np.int32)
    left_only = np.full((80, 60), CITYSCAPES.index("building"), dtype=np.int32)
    left_only[:, :20] = CITYSCAPES.index("sidewalk")
    segmenter = _Script([sidewalk, left_only, left_only, left_only])
    frame = _image(sidewalk)
    orch = Orchestrator(segmenter, _Boxes([]), hold_frames=3)
    first = orch.step(frame)
    second = orch.step(frame)
    third = orch.step(frame)
    fourth = orch.step(frame)
    assert first.cue.direction == "straight"
    assert second.cue.direction == "straight"
    assert third.cue.direction == "straight"
    assert fourth.cue.direction == "left"


def test_overlay_matches_the_frame_and_draws_the_banner():
    image, labels, boxes = synthetic_street(width=320, height=180)
    from streetseg.fuse import assess

    cue = assess(labels, NAMES, boxes)
    vis = draw_overlay(image, labels, NAMES, boxes, cue)
    assert vis.shape == image.shape
    assert vis.dtype == np.uint8
    assert int(vis[:8].mean()) < int(image[:8].mean())
    assert vis[32, 20, 0] > 200


def test_light_box_is_colored_from_the_pixels():
    labels = np.full((80, 60), CITYSCAPES.index("road"), dtype=np.int32)
    frame = _image(labels)
    frame[10:28, 24:36] = (20, 20, 240)
    box = Box("traffic light", 0.9, (22, 8, 38, 40))
    orch = Orchestrator(_Script([labels]), _Boxes([box]), hold_frames=1)
    result = orch.step(frame)
    assert result.boxes[0].signal == "red"
    assert result.cue.direction == "stop"
