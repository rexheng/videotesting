import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from face_shirt.shirt import (
    average_histograms,
    blend_histograms,
    histogram_similarity,
    shirt_histogram,
)


def solid(colour, size=8):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = colour
    quad = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float64)
    return frame, quad


class ShirtTest(unittest.TestCase):
    def test_primary_colours_land_in_the_expected_bins(self):
        red, quad = solid((0, 0, 255))
        described = shirt_histogram(red, quad)
        self.assertAlmostEqual(float(described["histogram"][5]), 1.0, places=6)
        self.assertEqual(described["swatch"], "#ff0000")

        white, _ = solid((255, 255, 255))
        self.assertAlmostEqual(float(shirt_histogram(white, quad)["histogram"][3]), 1.0, places=6)

        black, _ = solid((0, 0, 0))
        self.assertAlmostEqual(float(shirt_histogram(black, quad)["histogram"][0]), 1.0, places=6)

    def test_similarity(self):
        red, quad = solid((0, 0, 255))
        blue, _ = solid((255, 0, 0))
        red_hist = shirt_histogram(red, quad)["histogram"]
        blue_hist = shirt_histogram(blue, quad)["histogram"]
        self.assertLess(histogram_similarity(red_hist, blue_hist), 0.1)
        self.assertAlmostEqual(histogram_similarity(red_hist, red_hist), 1.0, places=6)

        mixed, _ = solid((0, 0, 255), size=8)
        mixed[:, 4:] = (255, 0, 0)
        mixed_hist = shirt_histogram(mixed, quad)["histogram"]
        self.assertAlmostEqual(histogram_similarity(mixed_hist, red_hist), 0.7071, delta=0.02)

    def test_blend(self):
        blended = blend_histograms([1, 0], [0, 1], 0.3)
        self.assertTrue(np.allclose(blended, [0.7, 0.3]))
        averaged = average_histograms([[1, 0], [0, 1]])
        self.assertTrue(np.allclose(averaged, [0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
