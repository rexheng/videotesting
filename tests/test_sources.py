import asyncio
import base64
import json
import sys
import types

import numpy as np
import pytest

from streetseg.sources import SimSocketSource, VideoFileSource, as_frame, decode_jpeg_message, open_file


def test_video_file_reads_a_frame(tmp_path):
    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("opencv has no MJPG writer")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 180)
    writer.write(frame)
    writer.release()
    source = VideoFileSource(path)
    try:
        got = source.read()
        assert got is not None
        assert got.shape == (48, 64, 3)
        assert got[:, :, 2].mean() > 100
    finally:
        source.close()


def test_kite_file_reader_is_used_when_it_is_installed(monkeypatch):
    sent = {}

    class Fake:
        def __init__(self, path):
            sent["path"] = path
            self.reads = 0

        def read(self):
            self.reads += 1
            if self.reads > 1:
                return None
            return np.zeros((4, 6, 3), dtype=np.uint8)

        def close(self):
            sent["closed"] = True

    package = types.ModuleType("kite")
    sources = types.ModuleType("kite.sources")
    sources.FileSource = Fake
    monkeypatch.setitem(sys.modules, "kite", package)
    monkeypatch.setitem(sys.modules, "kite.sources", sources)
    source = open_file("clip.mp4")
    try:
        frame = source.read()
        assert sent["path"] == "clip.mp4"
        assert frame.shape == (4, 6, 3)
        assert source.read() is None
        assert source.ended
    finally:
        source.close()
    assert sent["closed"]


def test_packet_with_an_image_attribute_is_a_frame():
    class Packet:
        image = np.zeros((2, 2, 3), dtype=np.uint8)

    assert as_frame(Packet()).shape == (2, 2, 3)
    assert as_frame(None) is None


def _jpeg(image):
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return buf.tobytes()


def test_sim_socket_accepts_a_binary_jpeg():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :] = (40, 180, 40)
    source = SimSocketSource(host="127.0.0.1", port=0, timeout=2)
    try:
        payload = _jpeg(image)

        async def push():
            import websockets

            async with websockets.connect(f"ws://127.0.0.1:{source.port}") as ws:
                await ws.send(payload)

        asyncio.run(push())
        frame = source.read()
        assert frame is not None
        assert frame.shape[0] == 48
        assert frame[:, :, 1].mean() > 100
    finally:
        source.close()


def test_sim_socket_accepts_json_with_intrinsics_and_pose():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[4:20, 4:20] = (0, 0, 220)
    source = SimSocketSource(host="127.0.0.1", port=0, timeout=2)
    try:
        body = json.dumps({
            "jpeg": base64.b64encode(_jpeg(image)).decode("ascii"),
            "fx": 100,
            "fy": 110,
            "height": 1.5,
            "pitch": 8,
        })

        async def push():
            import websockets

            async with websockets.connect(f"ws://127.0.0.1:{source.port}") as ws:
                await ws.send(body)

        asyncio.run(push())
        frame = source.read()
        assert frame is not None
        assert source.fx == 100
        assert source.fy == 110
        assert source.height_m == 1.5
        assert source.pitch == 8
        assert frame[:, :, 2].mean() > 20
    finally:
        source.close()


def test_decode_rejects_a_text_frame_without_jpeg():
    try:
        decode_jpeg_message(json.dumps({"fx": 1}))
    except ValueError as exc:
        assert "jpeg" in str(exc)
    else:
        raise AssertionError("expected ValueError")
