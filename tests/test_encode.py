import json
import subprocess

import numpy as np

from streetseg.__main__ import _writer


def test_mp4_writer_keeps_every_frame(tmp_path):
    path = tmp_path / "overlay.mp4"
    frames = []
    for index in range(5):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (index * 30, 40, 220 - index * 20)
        frame[10:20, 8:24] = (255, 255, 255)
        frames.append(frame)
    writer = _writer(str(path), frames[0].shape, 12.0, crf=28)
    for frame in frames:
        writer.write(frame)
    writer.release()

    data = path.read_bytes()
    assert data.find(b"moov") != -1
    assert data.find(b"moov") < data.find(b"mdat")

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,nb_frames,width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["width"] == 64
    assert stream["height"] == 48
    assert stream["r_frame_rate"] == "12/1"
    assert int(stream["nb_frames"]) == len(frames)

    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        read = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read.append(frame)
    finally:
        cap.release()
    assert len(read) == len(frames)
    assert read[0].shape == frames[0].shape
    assert int(read[0][:, :, 2].mean()) > int(read[-1][:, :, 2].mean())
