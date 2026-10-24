import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from face_shirt.geometry import (
    SFACE_TEMPLATE,
    apply_similarity,
    estimate_similarity,
    five_points,
    torso_quad,
    yaw_degrees,
)


def _blank(count):
    return [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0} for _ in range(count)]


class GeometryTest(unittest.TestCase):
    def test_similarity_round_trip(self):
        theta = math.radians(30)
        rotation = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=np.float64,
        )
        src = (rotation @ (SFACE_TEMPLATE * 2).T).T + np.array([100.0, 50.0])
        transform = estimate_similarity(src, SFACE_TEMPLATE)
        back = apply_similarity(transform, src)
        self.assertTrue(np.allclose(back, SFACE_TEMPLATE, atol=1e-6))

    def test_similarity_identity(self):
        transform = estimate_similarity(SFACE_TEMPLATE, SFACE_TEMPLATE)
        for key, expected in {"a": 1, "b": 0, "c": 0, "d": 1, "e": 0, "f": 0}.items():
            self.assertAlmostEqual(transform[key], expected, places=9)

    def test_yaw_sign_and_magnitude(self):
        for degrees in (0, 30, -30):
            theta = math.radians(degrees)
            landmarks = _blank(478)
            landmarks[234] = {"x": 0.5 - 0.1 * math.cos(theta), "y": 0.5, "z": -0.1 * math.sin(theta)}
            landmarks[454] = {"x": 0.5 + 0.1 * math.cos(theta), "y": 0.5, "z": 0.1 * math.sin(theta)}
            self.assertAlmostEqual(yaw_degrees(landmarks), degrees, delta=0.01)

    def test_five_points_average_the_eye_corners(self):
        landmarks = _blank(478)
        landmarks[33] = {"x": 0.2, "y": 0.4, "z": 0.0}
        landmarks[133] = {"x": 0.3, "y": 0.4, "z": 0.0}
        point = five_points(landmarks, 1000, 500)[0]
        self.assertTrue(np.allclose(point, [250, 200]))

    def test_hidden_shoulder_rejects_the_torso(self):
        landmarks = _blank(33)
        landmarks[11]["visibility"] = 0.3
        landmarks[12]["visibility"] = 1
        self.assertIsNone(torso_quad(landmarks, 1000, 1000))

    def test_missing_hips_are_estimated_then_shrunk(self):
        landmarks = _blank(33)
        landmarks[11] = {"x": 0.1, "y": 0.1, "z": 0.0, "visibility": 1}
        landmarks[12] = {"x": 0.2, "y": 0.1, "z": 0.0, "visibility": 1}
        quad = torso_quad(landmarks, 1000, 1000)
        self.assertIsNotNone(quad)
        self.assertTrue(np.allclose(quad[0], [107.5, 107.5]))


if __name__ == "__main__":
    unittest.main()
