import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from face_shirt.colour import shift_hue_bgr
from face_shirt.config import CLASS_CLOTHES, CLASS_NAMES
from face_shirt.segment import category_mask_array, rasterize_convex_quad, recolor_clothes, resize_mask_nearest


class SegmentTest(unittest.TestCase):
    def test_class_ids_match_the_model_label_list(self):
        self.assertEqual(CLASS_NAMES[CLASS_CLOTHES], "clothes")
        self.assertEqual(CLASS_NAMES[0], "background")

    def test_category_mask_drops_the_channel_axis(self):
        raw = np.zeros((4, 5, 1), dtype=np.uint8)
        raw[1, 2, 0] = CLASS_CLOTHES
        mask = category_mask_array(raw)
        self.assertEqual(mask.shape, (4, 5))
        self.assertEqual(int(mask[1, 2]), CLASS_CLOTHES)

    def test_nearest_resize_keeps_class_ids(self):
        mask = np.array([[CLASS_CLOTHES, 0], [0, 1]], dtype=np.uint8)
        scaled = resize_mask_nearest(mask, 4, 4)
        self.assertEqual(int(scaled[0, 0]), CLASS_CLOTHES)
        self.assertEqual(int(scaled[0, 1]), CLASS_CLOTHES)
        self.assertEqual(int(scaled[3, 3]), 1)

    def test_red_shifts_toward_yellow_and_unmasked_blue_stays(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 255)
        frame[0, 0] = (255, 0, 0)
        mask = np.full((2, 2), CLASS_CLOTHES, dtype=np.uint8)
        mask[0, 0] = 0
        output = recolor_clothes(frame, mask, 60)
        self.assertEqual(tuple(int(v) for v in output[0, 0]), (255, 0, 0))
        self.assertEqual(tuple(int(v) for v in output[1, 1]), (0, 255, 255))

    def test_grey_has_no_hue_to_shift(self):
        frame = np.full((2, 2, 3), 180, dtype=np.uint8)
        mask = np.full((2, 2), CLASS_CLOTHES, dtype=np.uint8)
        output = recolor_clothes(frame, mask, 120)
        self.assertTrue(np.array_equal(output, frame))

    def test_quad_keeps_the_shift_off_other_clothes(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 255)
        mask = np.full((20, 20), CLASS_CLOTHES, dtype=np.uint8)
        quad = np.array([[0, 0], [8, 0], [8, 19], [0, 19]], dtype=np.float64)
        output = recolor_clothes(frame, mask, 120, quad)
        self.assertEqual(tuple(int(v) for v in output[10, 2]), (0, 255, 0))
        self.assertEqual(tuple(int(v) for v in output[10, 18]), (0, 0, 255))

    def test_shift_preserves_value_of_a_dark_red(self):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        frame[0, 0] = (0, 0, 80)
        shifted = shift_hue_bgr(frame, 120)
        self.assertEqual(int(shifted[0, 0, 1]), 80)
        self.assertEqual(int(shifted[0, 0, 0]), 0)
        self.assertEqual(int(shifted[0, 0, 2]), 0)

    def test_convex_quad_contains_its_centre(self):
        quad = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        inside = rasterize_convex_quad(12, 12, quad)
        self.assertTrue(inside[5, 5])
        self.assertFalse(inside[11, 11])


if __name__ == "__main__":
    unittest.main()
