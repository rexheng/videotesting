import subprocess

import numpy as np

from streetseg.camera import Camera, Pose
from streetseg.display import Display
from streetseg.labels import CITYSCAPES, NAMES
from streetseg.live import play
from streetseg.pipeline import Perception


class _Script:
    def __init__(self, labels):
        self.names = dict(NAMES)
        self._labels = labels

    def predict(self, frame):
        if frame.shape[:2] != self._labels.shape[:2]:
            raise ValueError("size")
        return self._labels


class _Boxes:
    def predict(self, _frame):
        return []


class _Frames:
    def __init__(self, frames):
        self.frames = list(frames)
        self.ended = False
        self.closed = False

    def read(self):
        if not self.frames:
            self.ended = True
            return None
        return self.frames.pop(0)

    def close(self):
        self.closed = True


class _Memory:
    def __init__(self):
        self.shown = []
        self.closed = False

    def show(self, image, title="Kite street"):
        self.shown.append((title, image.shape))

    def poll_quit(self):
        return False

    def close(self):
        self.closed = True


def test_play_shows_each_frame_and_prints_events():
    labels = np.full((720, 960), CITYSCAPES.index("sidewalk"), dtype=np.int32)
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    source = _Frames([frame, frame])
    display = _Memory()
    events = []
    seen = play(
        Perception(_Script(labels), _Boxes(), hold_frames=1),
        source,
        Camera(682, 682, "drone"),
        Pose(1.6, 12),
        display,
        on_event=events.append,
    )
    assert seen == 2
    assert [title for title, _shape in display.shown] == ["Kite street", "Kite street"]
    assert any(event.kind == "say" and event.priority == "nav" for event in events)
    assert source.closed
    assert display.closed


def test_display_can_stay_off():
    display = Display(enabled=False)
    display.show(np.zeros((32, 48, 3), dtype=np.uint8))
    assert display.wait(1) == -1
    display.close()
    assert display.headless


def test_mp4_writer_is_h264(tmp_path):
    from streetseg.__main__ import _writer

    path = tmp_path / "clip.mp4"
    writer = _writer(str(path), (24, 32, 3), 10.0)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[:, :, 1] = 200
    writer.write(frame)
    writer.write(frame)
    writer.release()
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "h264,32,24"
