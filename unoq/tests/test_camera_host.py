import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_shirt.camera_nodes import choose_capture_node, parse_v4l2
from face_shirt.host import HostFacts, blocking_issues, parse_meminfo
from face_shirt.store import write_private_json


VENUS = """
Driver name   : qcom-venus
Card type     : Qualcomm Venus video decoder
Device caps   : 0x04200001
                Video Capture
"""

CAMERA = """
Driver name   : uvcvideo
Card type     : w300: w300
Device caps   : 0x04200001
                Video Capture
"""

IR = """
Card type     : BRIO IR
Device caps   : 0x04200001
                Video Capture
"""


class CameraTest(unittest.TestCase):
    def test_skips_venus_and_picks_the_usb_camera(self):
        nodes = [
            parse_v4l2("/dev/video0", VENUS),
            parse_v4l2("/dev/video1", VENUS.replace("decoder", "encoder")),
            parse_v4l2("/dev/video2", CAMERA),
            parse_v4l2("/dev/video3", "Card type: w300 metadata\nMetadata Capture"),
            parse_v4l2("/dev/video4", IR),
        ]
        self.assertEqual(choose_capture_node(nodes), "/dev/video2")

    def test_sysfs_name_without_a_capability_block_still_skips_venus(self):
        nodes = [
            parse_v4l2("/dev/video0", "qcom-venus-decoder", card="qcom-venus-decoder"),
            parse_v4l2("/dev/video2", "HP w300", card="HP w300"),
        ]
        self.assertEqual(choose_capture_node(nodes), "/dev/video2")


class HostTest(unittest.TestCase):
    def test_meminfo(self):
        total, available = parse_meminfo("MemTotal:        2048000 kB\nMemAvailable:     512000 kB\n")
        self.assertEqual(total, 2000)
        self.assertEqual(available, 500)

    def test_foreign_arch_blocks_a_run(self):
        issues = blocking_issues(HostFacts(arch="x86_64", python=(3, 13)), purpose="run")
        self.assertTrue(any("aarch64" in issue for issue in issues))

    def test_low_ram_and_desktop_on_the_2gb_board(self):
        facts = HostFacts(
            arch="aarch64",
            python=(3, 13),
            mem_total_mb=1900,
            mem_available_mb=400,
            display_session=True,
            nodes=(parse_v4l2("/dev/video2", CAMERA),),
        )
        issues = blocking_issues(facts, purpose="run")
        self.assertTrue(any("700" in issue for issue in issues))
        self.assertTrue(any("ABX00173" in issue for issue in issues))

    def test_4gb_board_with_a_camera_can_run(self):
        facts = HostFacts(
            arch="aarch64",
            python=(3, 13),
            mem_total_mb=3800,
            mem_available_mb=2200,
            display_session=True,
            disk_free_mb=4000,
            nodes=(parse_v4l2("/dev/video2", CAMERA),),
        )
        self.assertEqual(blocking_issues(facts, purpose="run"), [])

    def test_explicit_camera_overrides_an_empty_listing(self):
        facts = HostFacts(arch="aarch64", python=(3, 13), mem_total_mb=3800, mem_available_mb=2200)
        self.assertEqual(blocking_issues(facts, purpose="run", camera_path="/dev/video2"), [])

    def test_explicit_venus_node_is_still_rejected(self):
        facts = HostFacts(
            arch="aarch64",
            python=(3, 13),
            mem_total_mb=3800,
            mem_available_mb=2200,
            nodes=(parse_v4l2("/dev/video0", VENUS),),
        )
        issues = blocking_issues(facts, purpose="run", camera_path="/dev/video0")
        self.assertTrue(any("Venus" in issue for issue in issues))

    def test_only_venus_nodes_explain_the_codec(self):
        facts = HostFacts(
            arch="aarch64",
            python=(3, 13),
            mem_total_mb=3800,
            mem_available_mb=2200,
            nodes=(parse_v4l2("/dev/video0", VENUS),),
        )
        issues = blocking_issues(facts, purpose="run")
        self.assertTrue(any("Venus" in issue for issue in issues))

    def test_old_python_is_rejected(self):
        issues = blocking_issues(HostFacts(arch="aarch64", python=(3, 10)), purpose="fetch")
        self.assertTrue(any("3.10" in issue for issue in issues))

    def test_fetch_needs_a_gigabyte_free(self):
        facts = HostFacts(arch="aarch64", python=(3, 13), disk_free_mb=500, mem_available_mb=900)
        issues = blocking_issues(facts, purpose="fetch")
        self.assertTrue(any("/opt" in issue for issue in issues))


class StoreTest(unittest.TestCase):
    def test_people_file_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "people.json"
            write_private_json(path, [{"name": "Ada"}])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertIn("Ada", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
