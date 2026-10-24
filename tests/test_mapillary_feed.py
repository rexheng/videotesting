"""Mapillary crossings and the PVTL light head on a forward frame."""

import threading
import time

import numpy as np

from streetseg.camera import Camera, Pose
from streetseg.lights import LightReading, LightVote, crop_traffic_lights
from streetseg.mapillary import MAPILLARY, NAMES, crosswalk_in_band, ground_kind
from streetseg.pipeline import Perception
from streetseg.types import Box


def test_crosswalk_classes_are_walkable_and_a_crossing():
    assert ground_kind("Crosswalk - Plain") == "walk"
    assert ground_kind("Lane Marking - Crosswalk") == "walk"
    assert ground_kind("Sidewalk") == "walk"
    assert ground_kind("Person") == "person"
    assert ground_kind("Pothole") == "block"
    assert MAPILLARY.index("Traffic Light") == 48
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[20:, 10:40] = MAPILLARY.index("Crosswalk - Plain")
    assert crosswalk_in_band(labels, NAMES, y0=20, minimum=80)


def test_a_crossing_in_the_mask_is_said_and_a_red_pedestrian_light_stops():
    sidewalk = np.full((720, 960), MAPILLARY.index("Sidewalk"), dtype=np.int32)
    crossing = sidewalk.copy()
    crossing[400:, 300:660] = MAPILLARY.index("Lane Marking - Crosswalk")

    class _Seg:
        names = dict(NAMES)

        def predict(self, _frame):
            return crossing

    class _Det:
        def predict(self, _frame):
            return []

    class _Lights:
        def read(self, _frame, labels, names):
            assert crosswalk_in_band(labels, names, 400)
            box = Box("pedestrian-red", 1.0, (10, 10, 30, 40), signal="red", source="pvtl")
            return LightReading("pedestrian-red", ("pedestrian-red",), (box,))

    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    events = Perception(_Seg(), _Det(), lights=_Lights(), hold_frames=1).process(
        frame, Camera(682, 682, "drone"), Pose(0, 0)
    )
    scene = next(event for event in events if event.kind == "scene")
    said = next(event for event in events if event.kind == "say")
    assert scene.detail["crosswalk"] is True
    assert "crossing" in scene.text
    assert scene.detail["pedestrian_light"] == "pedestrian-red"
    assert said.text == "Stop. The pedestrian light is red."
    assert said.priority == "hazard"
    assert said.direction == "stop"


def test_green_waits_for_seven_frames_and_red_does_not():
    vote = LightVote(7)
    assert vote.push("pedestrian-green") is None
    for _ in range(5):
        assert vote.push("pedestrian-green") is None
    assert vote.push("pedestrian-green") == "pedestrian-green"
    assert vote.push("pedestrian-red") == "pedestrian-red"


def test_traffic_light_blobs_are_cropped_largest_first():
    labels = np.zeros((120, 160), dtype=np.int32)
    labels[10:60, 10:50] = MAPILLARY.index("Traffic Light")
    labels[8:20, 90:110] = MAPILLARY.index("Traffic Light")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[10:60, 10:50] = (0, 0, 255)
    crops, boxes = crop_traffic_lights(frame, labels, NAMES)
    assert len(crops) == 1
    assert boxes[0][2] - boxes[0][0] >= 32


def test_segmentation_and_boxes_run_together():
    labels = np.full((720, 960), MAPILLARY.index("Sidewalk"), dtype=np.int32)
    started = []
    release = threading.Event()

    class _Seg:
        names = dict(NAMES)

        def predict(self, _frame):
            started.append("seg")
            assert release.wait(2)
            return labels

    class _Det:
        def predict(self, _frame):
            started.append("det")
            assert release.wait(2)
            return [Box("person", 0.9, (10, 400, 40, 700))]

    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    perception = Perception(_Seg(), _Det(), parallel=True, hold_frames=1)
    worker = threading.Thread(
        target=perception.process,
        args=(frame, Camera(682, 682, "drone"), Pose(0, 0)),
    )
    begun = time.monotonic()
    worker.start()
    while len(started) < 2 and time.monotonic() - begun < 2:
        time.sleep(0.01)
    assert sorted(started) == ["det", "seg"]
    release.set()
    worker.join(3)
    assert not worker.is_alive()
    perception.close()
