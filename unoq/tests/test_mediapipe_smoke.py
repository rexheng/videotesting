"""Loads the real models when this interpreter has MediaPipe and the files."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from face_shirt.config import CLASS_NAMES, FACE_FILE, POSE_FILE, SEGMENT_FILE

MODEL_DIR = Path("/tmp/unoq-smoke")


def _mediapipe_ready() -> bool:
    try:
        import mediapipe  # noqa: F401
    except (ImportError, OSError):
        return False
    needed = (SEGMENT_FILE, FACE_FILE, POSE_FILE)
    return all((MODEL_DIR / name).is_file() for name in needed)


@unittest.skipUnless(_mediapipe_ready(), "MediaPipe and the pinned model files are not installed here")
class MediaPipeSmokeTest(unittest.TestCase):
    def test_cpu_segmenter_returns_a_frame_sized_mask(self):
        from face_shirt.runners import VisionRunners

        runners = VisionRunners(MODEL_DIR, need_face=True, need_segment=True)
        try:
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            mask = runners.segment(frame, 1)
            self.assertEqual(mask.shape, (120, 160))
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(set(np.unique(mask)).issubset(set(range(len(CLASS_NAMES)))))
            self.assertEqual(runners.face(frame, 1), [])
            self.assertEqual(runners.pose(frame, 1), [])
        finally:
            runners.close()


if __name__ == "__main__":
    unittest.main()
