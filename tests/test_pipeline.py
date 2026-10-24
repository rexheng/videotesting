import numpy as np

from streetseg.camera import DEMO_POSE, TELLO_CAMERA, Camera, Pose
from streetseg.fuse import FuseConfig
from streetseg.labels import CITYSCAPES, NAMES
from streetseg.overlay import draw_overlay
from streetseg.pipeline import DirectionHold, Perception
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


def _kinds(events):
    return [event.kind for event in events]


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


def test_a_new_direction_waits_and_a_stop_is_said_at_once():
    sidewalk = np.full((720, 960), CITYSCAPES.index("sidewalk"), dtype=np.int32)
    left_only = np.full((720, 960), CITYSCAPES.index("building"), dtype=np.int32)
    left_only[:, :320] = CITYSCAPES.index("sidewalk")
    blocked = np.full((720, 960), CITYSCAPES.index("building"), dtype=np.int32)
    segmenter = _Script([sidewalk, left_only, left_only, left_only, blocked])
    frame = _image(sidewalk)
    perception = Perception(segmenter, _Boxes([]), hold_frames=3)
    camera = Camera(fx=682, fy=682, role="drone")
    pose = Pose(0.0, 0.0)
    said = []
    pending = []
    for _ in range(5):
        events = perception.process(frame, camera, pose)
        said.extend(event for event in events if event.kind == "say")
        hint = next(event for event in events if event.kind == "hint")
        pending.append(hint.detail["pending"])
    assert [event.direction for event in said] == ["straight", "left", "stop"]
    assert said[0].priority == "nav"
    assert said[1].priority == "nav"
    assert said[2].priority == "hazard"
    assert pending[1] == "left"
    assert pending[3] == ""
    assert "turn" not in said[0].to_dict()


def test_sim_street_says_bear_left_and_keeps_the_walker_out_of_the_crowd():
    image, labels, boxes = synthetic_street()
    perception = Perception(_Script([labels]), _Boxes(boxes), hold_frames=1)
    events = perception.process(image, TELLO_CAMERA.with_role("sim"), DEMO_POSE)
    say = next(event for event in events if event.kind == "say")
    hint = next(event for event in events if event.kind == "hint")
    scene = next(event for event in events if event.kind == "scene")
    assert say.priority == "nav"
    assert say.source == "sim"
    assert say.text.lower().startswith("bear left")
    assert hint.text == "bear left"
    assert hint.priority == "nav"
    assert "hazard" not in _kinds(events)
    assert "obstacle" not in _kinds(events)
    assert scene.detail["wearer_excluded"] is True
    assert scene.detail["columns"]["center"]["people"] == 0
    assert perception.image.shape == image.shape
    assert "turn" not in scene.to_dict()


def test_pitch_moves_the_ground_band():
    height, width = 720, 960
    camera = Camera(fx=682, fy=682, role="drone", cx=width / 2, cy=height / 2)
    from streetseg.camera import ground_row

    level_row = ground_row(height, camera, Pose(1.5, 0.0), 4.0)
    down_row = ground_row(height, camera, Pose(1.5, 35.0), 4.0)
    assert down_row < level_row - 50
    labels = np.full((height, width), CITYSCAPES.index("building"), dtype=np.int32)
    labels[level_row:, :] = CITYSCAPES.index("sidewalk")
    frame = _image(labels)
    cfg = FuseConfig(near_distance_m=4.0)
    level = Perception(_Script([labels.copy()]), _Boxes([]), cfg=cfg).process(frame, camera, Pose(1.5, 0.0))
    steep = Perception(_Script([labels.copy()]), _Boxes([]), cfg=cfg).process(frame, camera, Pose(1.5, 35.0))
    assert next(event.direction for event in level if event.kind == "hint") == "straight"
    assert next(event for event in steep if event.kind == "hazard").reason == "blocked"


def test_chest_person_is_an_obstacle_and_does_not_segment():
    class Boom:
        names = NAMES

        def predict(self, _frame):
            raise AssertionError("chest view must not segment the street")

    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    box = Box("person", 0.9, (300, 80, 700, 700))
    events = Perception(Boom(), _Boxes([box])).process(
        frame, Camera(fx=682, fy=682, role="chest"), Pose(1.3, 20)
    )
    assert "obstacle" in _kinds(events)
    assert "scene" not in _kinds(events)
    say = next(event for event in events if event.kind == "say")
    assert say.priority == "hazard"
    assert say.source == "chest"
    assert "person" in say.text.lower()


def test_same_direction_is_not_said_twice():
    labels = np.full((720, 960), CITYSCAPES.index("sidewalk"), dtype=np.int32)
    perception = Perception(_Script([labels, labels]), _Boxes([]), hold_frames=3)
    camera = Camera(fx=682, fy=682, role="drone")
    pose = Pose(0.0, 0.0)
    frame = _image(labels)
    first = perception.process(frame, camera, pose)
    second = perception.process(frame, camera, pose)
    assert sum(event.kind == "say" for event in first) == 1
    assert sum(event.kind == "say" for event in second) == 0
    assert any(event.kind == "hint" for event in second)


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
    labels = np.full((720, 960), CITYSCAPES.index("road"), dtype=np.int32)
    frame = _image(labels)
    frame[80:160, 450:510] = (20, 20, 240)
    box = Box("traffic light", 0.9, (440, 60, 520, 200))
    perception = Perception(_Script([labels]), _Boxes([box]), hold_frames=1)
    events = perception.process(frame, Camera(fx=682, fy=682, role="drone"), Pose(0, 0))
    assert perception.boxes[0].signal == "red"
    assert any(event.kind == "hazard" and event.reason == "signal" for event in events)


def test_odd_sized_frame_keeps_its_own_shape():
    class _AsFrame:
        names = dict(NAMES)

        def predict(self, bgr):
            return np.full(bgr.shape[:2], CITYSCAPES.index("sidewalk"), dtype=np.int32)

    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    camera = Camera(fx=200, fy=150, role="drone")
    perception = Perception(_AsFrame(), _Boxes([]))
    perception.process(frame, camera, Pose(0, 0))
    assert perception.image.shape == (180, 320, 3)
