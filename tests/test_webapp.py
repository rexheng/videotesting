"""The browser page shows the C270 and the Tello, input beside output."""

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

from streetseg.camera import Camera, Pose, focal_length
from streetseg.webapp import LiveBoards, camera_index, droidcam_url, make_server


class _Paint:
    def __init__(self):
        self.image = None
        self.closed = False

    def process(self, frame, camera, pose):
        image = np.array(frame, copy=True)
        image[:, :, 1] = 255
        self.image = image
        return ()

    def close(self):
        self.closed = True


class _Loop:
    def __init__(self, frame):
        self.frame = frame
        self.closed = False
        self.ended = False
        self.failed_open = False
        self.sent = False

    def read(self):
        if not self.sent:
            self.sent = True
            return np.array(self.frame, copy=True)
        time.sleep(0.02)
        return None

    def close(self):
        self.closed = True


def test_page_puts_each_input_feed_beside_its_output():
    from streetseg.webapp import _PAGE

    assert "grid-template-columns: 1fr 1fr" in _PAGE
    assert _PAGE.index("Logitech C270") < _PAGE.index("DJI Tello")
    assert _PAGE.index("DJI Tello") < _PAGE.index("Phone URL")
    for board in ("board-c270", "board-tello"):
        block = _PAGE.split(f'id="{board}"', 1)[1].split("<section", 1)[0]
        assert block.index("Input feed") < block.index("Output feed")
    assert "1280×720" in _PAGE
    assert "stays on the ground" in _PAGE


def test_droidcam_url_and_camera_index():
    assert droidcam_url("192.168.0.20") == "http://192.168.0.20:4747/video"
    assert droidcam_url("192.168.0.20:4748") == "http://192.168.0.20:4748/video"
    kept = "http://10.0.0.8:4747/mjpegfeed?1280x720"
    assert droidcam_url(kept) == kept
    with pytest.raises(ValueError):
        droidcam_url("  ")
    assert camera_index("") == 0
    assert camera_index("2") == 2
    with pytest.raises(ValueError):
        camera_index("-1")


def _post(url, fields=None):
    data = urllib.parse.urlencode(fields or {}).encode()
    request = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _wait_frames(boards, name):
    deadline = time.time() + 3
    while time.time() < deadline:
        if boards.status()[name]["frames"] >= 1:
            return
        time.sleep(0.02)
    raise AssertionError(f"{name} produced no frame")


def test_c270_and_tello_serve_full_input_and_output(monkeypatch):
    import cv2

    calls = []
    c270_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    c270_frame[0, 0] = (0, 0, 220)
    tello_frame = np.zeros((720, 960, 3), dtype=np.uint8)
    tello_frame[0, 0] = (220, 0, 0)

    def c270_factory(index=0):
        calls.append(("c270", index))
        return _Loop(c270_frame)

    def tello_factory():
        calls.append("tello")
        return _Loop(tello_frame)

    made = []

    def perception_factory():
        made.append(_Paint())
        return made[-1]

    boards = LiveBoards(
        perception_factory,
        Camera(100, 100, "drone"),
        Pose(1.6, 12),
        camera_fns={
            "c270": lambda frame, camera: Camera(focal_length(frame.shape[1], frame.shape[0], 60.0), focal_length(frame.shape[1], frame.shape[0], 60.0), camera.role),
            "tello": lambda frame, camera: Camera(focal_length(frame.shape[1], frame.shape[0], 82.6), focal_length(frame.shape[1], frame.shape[0], 82.6), camera.role),
        },
        tello_factory=tello_factory,
        c270_factory=c270_factory,
    )
    server = make_server(boards, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            page = response.read().decode()
        assert "Input feed" in page and "Output feed" in page
        assert "Logitech C270" in page and "DJI Tello" in page

        code, payload = _post(base + "/c270", {"index": "2"})
        assert code == 200 and payload["index"] == 2
        code, payload = _post(base + "/tello")
        assert code == 200 and payload["source"] == "Tello"
        bad, error = _post(base + "/c270", {"index": "nope"})
        assert bad == 400 and "camera number" in error["error"]

        _wait_frames(boards, "c270")
        _wait_frames(boards, "tello")
        status = json.loads(urllib.request.urlopen(base + "/status", timeout=5).read())
        assert status["c270"]["width"] == 1280 and status["c270"]["height"] == 720
        assert status["tello"]["width"] == 960 and status["tello"]["height"] == 720
        assert status["c270"]["source"] == "Logitech C270"
        assert status["tello"]["source"] == "Tello"

        for name, shape in (("c270", (720, 1280, 3)), ("tello", (720, 960, 3))):
            raw = urllib.request.urlopen(base + f"/boards/{name}/input", timeout=5).read()
            painted = urllib.request.urlopen(base + f"/boards/{name}/output", timeout=5).read()
            assert raw != painted
            left = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            right = cv2.imdecode(np.frombuffer(painted, np.uint8), cv2.IMREAD_COLOR)
            assert left.shape == shape
            assert right.shape == shape
            assert int(right[:, :, 1].mean()) > int(left[:, :, 1].mean())

        assert calls == [("c270", 2), "tello"]
        assert "takeoff" not in calls
        assert len(made) == 2
        assert boards._boards["c270"].camera.fx == pytest.approx(focal_length(1280, 720, 60.0))
        assert boards._boards["tello"].camera.fx == pytest.approx(focal_length(960, 720, 82.6))
    finally:
        server.shutdown()
        server.server_close()
        boards.close()
    assert all(item.closed for item in made)


def test_cli_web_connects_c270_and_tello_without_takeoff(monkeypatch):
    calls = []

    def c270(index=0):
        calls.append(("c270", index))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        return _Loop(frame)

    def tello():
        calls.append("tello")
        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        return _Loop(frame)

    monkeypatch.setattr("streetseg.sources.open_c270", c270)
    monkeypatch.setattr("streetseg.sources.open_tello", tello)
    monkeypatch.setattr("streetseg.__main__._models", lambda args, need_segmenter=True: _Paint())
    held = {}

    def serve(boards, host, port):
        deadline = time.time() + 3
        while time.time() < deadline:
            status = boards.status()
            if status["c270"]["frames"] and status["tello"]["frames"]:
                break
            time.sleep(0.02)
        held["status"] = boards.status()
        held["c270_fx"] = boards._boards["c270"].camera.fx
        held["tello_fx"] = boards._boards["tello"].camera.fx
        held["host"] = host

    monkeypatch.setattr("streetseg.webapp.serve", serve)
    from streetseg.__main__ import main

    assert main(["--web", "--webcam", "0", "--tello", "--web-port", "9"]) == 0
    assert ("c270", 0) in calls
    assert "tello" in calls
    assert "takeoff" not in calls
    assert held["status"]["c270"]["width"] == 1280
    assert held["status"]["c270"]["height"] == 720
    assert held["status"]["tello"]["width"] == 960
    assert held["status"]["tello"]["height"] == 720
    assert held["c270_fx"] == pytest.approx(focal_length(1280, 720, 60.0))
    assert held["tello_fx"] == pytest.approx(focal_length(960, 720, 82.6))
    assert held["host"] == "127.0.0.1"
