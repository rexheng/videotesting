import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from face_shirt.config import CLASS_CLOTHES
from face_shirt.pipeline import Pipeline


class FakeRunners:
    def __init__(self):
        self.segment_calls = 0
        self.face_calls = 0
        self.costs = {}

    def face(self, frame, timestamp_ms):
        del frame, timestamp_ms
        self.face_calls += 1
        return []

    def embed(self, frame, timestamp_ms):
        del frame, timestamp_ms
        return 0

    def pose(self, frame, timestamp_ms):
        del frame, timestamp_ms
        return []

    def segment(self, frame, timestamp_ms):
        del timestamp_ms
        self.segment_calls += 1
        height, width = frame.shape[:2]
        mask = np.full((height, width), CLASS_CLOTHES, dtype=np.uint8)
        return mask

    def cost_ms(self, stage):
        return self.costs.get(stage)


class PipelineTest(unittest.TestCase):
    def test_recolor_holds_the_mask_between_refreshes(self):
        runners = FakeRunners()
        pipeline = Pipeline(runners, mode="recolor", hue_degrees=120)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 255)
        first = pipeline.step(frame, 0)
        self.assertTrue(first.segmented)
        self.assertEqual(tuple(int(v) for v in first.frame[1, 1]), (0, 255, 0))
        second = pipeline.step(frame, 200)
        self.assertFalse(second.segmented)
        self.assertEqual(runners.segment_calls, 1)
        self.assertEqual(tuple(int(v) for v in second.frame[1, 1]), (0, 255, 0))
        third = pipeline.step(frame, 2000)
        self.assertTrue(third.segmented)
        self.assertEqual(runners.segment_calls, 2)

    def test_a_slow_mask_forces_one_cooldown_frame(self):
        runners = FakeRunners()
        runners.costs["segment"] = 2500
        pipeline = Pipeline(runners, mode="recolor", hue_degrees=0)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        pipeline.step(frame, 0)
        cooled = pipeline.step(frame, 10)
        self.assertEqual(cooled.plan.reason, "cooldown")
        self.assertEqual(runners.segment_calls, 1)
        # The mask from the slow frame is still fresh, so the next due
        # refresh is the interval, not the cooldown itself.
        again = pipeline.step(frame, 2000)
        self.assertEqual(again.plan.reason, "segment")
        self.assertEqual(runners.segment_calls, 2)

    def test_tagging_does_not_recolor_or_segment(self):
        runners = FakeRunners()
        pipeline = Pipeline(runners, mode="tag")
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 255)
        result = pipeline.step(frame, 0)
        self.assertEqual(result.plan.stages, ("face", "embed"))
        self.assertEqual(runners.segment_calls, 0)
        self.assertFalse(result.recolored)
        self.assertTrue(np.array_equal(result.frame, frame))

    def test_stale_torso_no_longer_gates_the_shirt(self):
        runners = FakeRunners()
        pipeline = Pipeline(runners, mode="recolor", hue_degrees=120)
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 255)
        pipeline.mask = np.full((20, 20), CLASS_CLOTHES, dtype=np.uint8)
        pipeline.quad = np.array([[0, 0], [8, 0], [8, 19], [0, 19]], dtype=np.float64)
        pipeline.quad_ms = 0
        pipeline.state.has_mask = True
        pipeline.state.last_segment_ms = 0
        held = pipeline.step(frame, 500)
        self.assertEqual(held.plan.reason, "track")
        self.assertEqual(tuple(int(v) for v in held.frame[10, 18]), (0, 0, 255))
        expired = pipeline.step(frame, 5000)
        self.assertEqual(tuple(int(v) for v in expired.frame[10, 18]), (0, 255, 0))


if __name__ == "__main__":
    unittest.main()
