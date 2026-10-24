# Face and shirt segmentation on the Arduino UNO Q

The camera loop runs on the Qualcomm QRB2210 (the Linux side of the UNO Q). The STM32 microcontroller cannot hold a camera frame, and a browser on the board is not the inference runtime. The audit that fixes those limits is [`docs/superpowers/specs/2026-10-24-uno-q-segmentation-audit.md`](../docs/superpowers/specs/2026-10-24-uno-q-segmentation-audit.md).

## What runs

* **Recolour** (`--mode recolor`): the 256×256 multiclass selfie model labels clothes. Hue is shifted and brightness is kept. The mask is reused on later frames, and a new mask is computed about every 1.5 s, on its own frame.
* **Tag** (`--mode tag`): face landmarks for at most two people, an optional SFace embedding, and a torso colour histogram. This is the path that has to put a name up within about a second.
* **Both**: the face frame keeps the one-second budget. The clothes model runs only when the face is not due.

A powered USB-C hub is required. `/dev/video0` and `/dev/video1` are the Qualcomm Venus codec, not the webcam.

## Install on the board

```sh
cd unoq
sh install.sh
. .venv/bin/activate
python -m face_shirt --run --mode recolor
```

`install.sh` installs `libegl1` and `libgles2` (MediaPipe needs them even for the CPU delegate), creates a virtualenv, installs MediaPipe 1.1.1 without the conflicting OpenCV contrib package, and downloads the models.

Check the budget without a camera:

```sh
python -m face_shirt --self-check
```

Useful flags: `--mode tag|recolor|both`, `--hue 120`, `--camera /dev/video2`, `--frames 30`, `--save /tmp/frame.jpg`.

People records, when the enrolment UI is added, go in a mode-`0600` file on the board. Nothing in this loop uploads frames or embeddings. A browser preview is not started.

## Tests

From the repository root, with NumPy available:

```sh
python3 -m unittest discover -s unoq/tests -v
```

The MediaPipe smoke test runs only when this interpreter can import MediaPipe and `/tmp/unoq-smoke` holds the model files.
