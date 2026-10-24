import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_shirt.budget import assert_board_invariants, report_lines
from face_shirt.config import DELEGATE, MAX_FACES, MCU_FRAME_BYTES, MCU_SRAM_BYTES
from face_shirt.mcu import bridge_text, frame_fits_on_mcu


class BudgetTest(unittest.TestCase):
    def test_invariants_hold(self):
        assert_board_invariants()

    def test_report_names_the_board_and_the_cpu(self):
        text = "\n".join(report_lines())
        self.assertIn("qrb2210-mpu", text)
        self.assertIn("delegate: CPU", text)
        self.assertIn("name budget: 1000 ms", text)
        self.assertIn("never shares a frame", text)
        self.assertEqual(DELEGATE, "CPU")
        self.assertEqual(MAX_FACES, 2)

    def test_a_vga_frame_does_not_fit_in_mcu_sram(self):
        self.assertGreater(MCU_FRAME_BYTES, MCU_SRAM_BYTES)
        self.assertFalse(frame_fits_on_mcu())

    def test_bridge_payload_is_short_text(self):
        self.assertEqual(bridge_text("  Ada   Lovelace  "), "Ada Lovelace")
        self.assertLessEqual(len(bridge_text("x" * 80)), 30)


class InstallContractTest(unittest.TestCase):
    def test_requirements_do_not_install_a_second_opencv(self):
        root = Path(__file__).resolve().parents[1]
        requirements = (root / "requirements.txt").read_text(encoding="utf-8")
        requirement_lines = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        script = (root / "install.sh").read_text(encoding="utf-8")
        self.assertIn("opencv-python-headless==4.11.0.86", requirement_lines)
        self.assertFalse(any("opencv-contrib" in line for line in requirement_lines))
        self.assertIn("--no-deps", script)
        self.assertIn("mediapipe==1.1.1", script)
        self.assertIn("libegl1", script)
        self.assertIn("libgles2", script)


if __name__ == "__main__":
    unittest.main()
