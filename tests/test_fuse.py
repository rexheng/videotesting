import numpy as np

from streetseg.fuse import FuseConfig, assess
from streetseg.labels import CITYSCAPES, NAMES
from streetseg.scene import synthetic_street
from streetseg.types import Box


def _sidewalk(height=120, width=90):
    labels = np.full((height, width), CITYSCAPES.index("sidewalk"), dtype=np.int32)
    return labels


def _fill(labels, name, x0, y0, x1, y1):
    labels[y0:y1, x0:x1] = CITYSCAPES.index(name)


def test_demo_street_bears_left_toward_the_empty_sidewalk():
    _image, labels, boxes = synthetic_street()
    cue = assess(labels, NAMES, boxes)
    assert cue.direction == "left"
    assert cue.turn == -1
    assert cue.stop is False
    assert cue.crosswalk is True
    assert cue.signal == "green"
    assert cue.reason == "clear"
    assert cue.speech == "Bear left. The way looks clear. The crossing light looks green."
    assert cue.wearer_excluded is True
    assert cue.columns[0].selectable
    assert not cue.columns[1].selectable


def test_clear_road_stays_straight():
    labels = _sidewalk()
    cue = assess(labels, NAMES, [])
    assert cue.direction == "straight"
    assert cue.turn == 0
    assert cue.speech == "Continue straight. The way looks clear."


def test_red_light_ahead_stops_and_a_side_light_does_not():
    labels = _sidewalk()
    red_center = Box("traffic light", 0.9, (40, 10, 55, 40), signal="red")
    stopped = assess(labels, NAMES, [red_center])
    assert stopped.direction == "stop"
    assert stopped.reason == "signal"
    assert stopped.speech == "Stop. The light looks red."
    assert stopped.turn == 0

    red_right = Box("traffic light", 0.9, (70, 10, 85, 40), signal="red")
    walking = assess(labels, NAMES, [red_right])
    assert walking.direction == "straight"
    assert walking.signal == "none"


def test_stop_sign_in_the_near_lane():
    labels = _sidewalk(height=100, width=90)
    sign = Box("stop sign", 0.8, (35, 60, 55, 95))
    cue = assess(labels, NAMES, [sign])
    assert cue.direction == "stop"
    assert cue.reason == "stop-sign"
    assert "stop sign" in cue.speech


def test_quieter_side_wins_when_the_scores_are_close():
    labels = _sidewalk(height=100, width=90)
    # The center is still walkable, but two people stand in it. The count
    # penalty is kept smaller than the center bias so the center wins on
    # score and the crowd tie-break is what actually moves us left.
    boxes = [
        Box("person", 0.7, (34, 70, 44, 96)),
        Box("person", 0.7, (48, 72, 58, 96)),
    ]
    cue = assess(
        labels,
        NAMES,
        boxes,
        cfg=FuseConfig(person_count_penalty=0.01, crowd_window=0.12),
    )
    assert cue.direction == "left"
    assert cue.reason == "crowd"
    assert cue.speech.startswith("Bear left. Fewer people that way.")


def test_blocked_center_uses_the_open_side():
    labels = _sidewalk(height=100, width=90)
    _fill(labels, "car", 30, 55, 60, 100)
    cue = assess(labels, NAMES, [Box("car", 0.9, (32, 58, 58, 98))])
    assert cue.direction == "left"
    assert cue.columns[1].selectable is False


def test_wearer_cut_off_by_the_frame_is_not_a_crowd():
    labels = _sidewalk(height=100, width=90)
    _fill(labels, "building", 0, 0, 30, 100)
    _fill(labels, "person", 30, 70, 60, 100)
    wearer = Box("person", 0.95, (30, 70, 60, 99))
    other = Box("person", 0.8, (64, 60, 82, 92))
    _fill(labels, "person", 64, 60, 82, 92)
    kept = assess(labels, NAMES, [wearer, other])
    assert kept.wearer_excluded is True
    assert kept.direction == "straight"
    assert kept.columns[1].people == 0

    counted = assess(labels, NAMES, [wearer, other], cfg=FuseConfig(exclude_wearer=False))
    assert counted.direction == "right"
    assert counted.columns[1].people == 1


def test_body_camera_person_in_front_stops_immediately():
    labels = _sidewalk()
    body = [Box("person", 0.9, (30, 20, 60, 110))]
    cue = assess(labels, NAMES, [], body_boxes=body, body_shape=(120, 90))
    assert cue.direction == "stop"
    assert cue.reason == "sudden"
    assert cue.speech == "Stop. A person is close in front of you."


def test_hands_at_the_bottom_of_the_body_camera_are_ignored():
    labels = _sidewalk()
    hands = [Box("person", 0.6, (10, 100, 40, 118))]
    cue = assess(labels, NAMES, [], body_boxes=hands, body_shape=(120, 90))
    assert cue.direction == "straight"
    assert cue.hazards == ()


def test_depth_marks_a_small_box_as_close():
    labels = _sidewalk()
    depth = np.zeros((120, 90), dtype=np.float32)
    depth[48:88, 30:55] = 10
    box = Box("chair", 0.8, (32, 50, 52, 86))
    cue = assess(labels, NAMES, [], body_boxes=[box], depth=depth, body_shape=(120, 90))
    assert cue.direction == "stop"
    assert "chair" in cue.speech


def test_nothing_walkable_stops():
    labels = np.full((80, 60), CITYSCAPES.index("building"), dtype=np.int32)
    cue = assess(labels, NAMES, [])
    assert cue.direction == "stop"
    assert cue.reason == "blocked"
    assert cue.speech == "Stop. The path ahead looks blocked."
