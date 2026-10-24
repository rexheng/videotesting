# Arduino UNO Q audit — face, shirt, and clothes segmentation

Date: 2026-10-24
Status: Binding for any build that runs this product on Arduino Q. The experience in `2026-10-15-face-shirt-tagger-design.md` still holds. The browser runtime in that document does not.

## Verdict

The segmentation loop can run on an Arduino UNO Q, on the Qualcomm QRB2210 Linux processor, with the schedule in `unoq/`. It cannot run on the STM32, and it cannot run as the browser pipeline in the original plan (MediaPipe Tasks in the page, `onnxruntime-web`, four faces, the clothes model on the same frames as the face landmarker).

What "can run" means here:

| Job | On the UNO Q |
|---|---|
| One person recognised by face, then named, within about a second | Yes. Planning cost 490 ms before the shirt sample. |
| Two people in the same frame | Yes, if pose stays on a later frame. Planning cost 800 ms including a recolour of the mask already in memory. |
| Four people, as the original plan allowed | No. The cap is 2. |
| Shirt colour as a histogram, for recognition | Yes. A few milliseconds, on the pose frame. |
| Clothes segmentation (class 4) and a hue shift that keeps shading | Yes, as its own frame. Planning cost 1.5 s. The previous mask recolours the frames in between. |
| That segmentation stacked on the face landmarker every frame | No. The sum is past the one-second name budget, and the model is a float32 transformer, not a tiny quantised net. |
| The same loop on the STM32U585 | No. One 640×480 frame is 921,600 bytes. The microcontroller has 786 KB of SRAM. |

These millisecond figures are ceilings for a 4× Cortex-A53 at 2.0 GHz, not a stopwatch reading from a UNO Q. The loop records the real time and, if a frame takes more than 2 s, spends the next frame on capture and the held mask only. The first run on a board should be checked against that.

## The board

Arduino Q is the UNO Q. Two processors share the board:

* **MPU.** Qualcomm Dragonwing QRB2210. 4× Arm Cortex-A53 at up to 2.0 GHz, Adreno 702 GPU at 845 MHz (OpenGL ES 3.1, OpenCL 2.0, Vulkan 1.1), dual ISPs, Debian. RAM is 2 GB (ABX00162, 16 GB eMMC) or 4 GB (ABX00173, 32 GB eMMC). There is a Hexagon DSP for always-on audio and sensors. Qualcomm's own product page says inference on this chip is CPU and GPU. It is not the Hexagon tensor processor that Edge Impulse's QNN path documents for the QCS6490.
* **MCU.** STM32U585, Cortex-M33 at up to 160 MHz, 2 MB flash, 786 KB SRAM, Zephyr. It can show a short name. It cannot own the camera or the models.

The single USB-C port also carries power. Arduino's own face-detection example requires a powered USB-C hub (5 V, 3 A) and a UVC webcam. Plug the camera in there. `v4l2` then exposes the Qualcomm Venus codec as `/dev/video0` and `/dev/video1`, and the webcam further along, often `/dev/video2`. Opening video0 does not open the camera. The node picker in `unoq/face_shirt/camera_nodes.py` drops anything whose name contains Venus, encoder, decoder, or metadata, and prefers a colour capture node over an infrared one.

Capture is MJPG at 640×480. Uncompressed YUYV above 640×480 falls off the USB bandwidth. The models do not need 720p or 1080p.

## What the original process assumed

The approved design is a page: MediaPipe Face Landmarker and Pose Landmarker from `@mediapipe/tasks-vision`, SFace through `onnxruntime-web` on WASM, up to four faces and four poses, a shirt histogram, and a later fork that runs MediaPipe's multiclass selfie segmenter and shifts the hue of the clothes pixels inside the torso quad.

That fork is the segmentation this audit is about. The rest of the loop matters because it shares the camera. A segmenter that only works when the face model is switched off does not ship the product.

## Why the browser runtime does not ship on this board

1. **The heavy work would sit in Chromium.** MediaPipe's web tasks want WASM SIMD and WebGL. The Adreno 702 can do graphics, but the Python GPU delegate that ships with MediaPipe is documented as Ubuntu-only, and the UNO Q image is Debian. A page that asks for the GPU delegate and then falls back to WASM CPU is the slow path, on top of a desktop. The supported vision examples from Arduino and Edge Impulse run the model in Python on the MPU and use the browser as a window.
2. **The page would be the wrong place to keep face data.** If someone opens the UI from a laptop, `localStorage` is on the laptop. The privacy rule is that faces stay on the device. Enrolment belongs in a mode-`0600` file on the eMMC (`unoq/face_shirt/store.py`). This loop does not start a preview server. Turning one on later is an explicit choice to send pixels across the LAN.
3. **App Lab's default bricks are a poor fit.** Bricks target Python 3.13 containers. MediaPipe 0.10.x has no 3.13 wheel. MediaPipe 1.1.1 (released 14 October 2026) publishes `mediapipe-1.1.1-py3-none-manylinux_2_28_aarch64.whl`, and the 1.1.0 notes add Python 3.13 and 3.14 to the build. Use 1.1.1. Do not use a brick graph that forces 0.10.x.
4. **MediaPipe 1.1.1 declares `opencv-contrib-python`.** That wheel is about 58 MB compressed and it is a second `cv2`. Installing it next to `opencv-python-headless` made `cv2.__version__` report 5.0.0 in a test environment. The clothes model, the face landmarker, and the pose landmarker all ran on `opencv-python-headless==4.11.0.86` alone. `install.sh` installs MediaPipe with `--no-deps` for that reason.
5. **The native library needs EGL even on the CPU path.** `libmediapipe.so` links `libEGL.so.1` and `libGLESv2.so.2`. Importing it failed until `libegl1` and `libgles2` were installed. `matplotlib` is imported by `mediapipe.tasks.vision` at import time, so it is a real dependency. `sounddevice` is declared and is not imported by this loop; it is left out so PortAudio is not required.

Checked on 24 October 2026, on an x86 Linux host, with MediaPipe 1.1.1 and the CPU delegate (XNNPACK logged itself as created):

* Model labels were exactly `background`, `hair`, `body-skin`, `face-skin`, `clothes`, `others`. Class 4 is clothes.
* A 640×480 input came back as a uint8 category mask of the same size (a trailing channel of 1, which the pipeline squeezes).
* A blank frame produced no faces and no poses. Those calls returned in a few milliseconds because the detector gave up. That timing is not a budget.
* A solid 640×480 frame ran the segmenter in about 60 ms on that host. That host is not a Cortex-A53. The 60 ms figure is evidence the graph loads, not evidence of the board's speed.

## The clothes model

Pinned file, fetched from Google's model bucket (HTTP 200, 16,371,837 bytes):

`selfie_multiclass_256x256.tflite`
`https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite`

There is no `.task` at the usual MediaPipe path (the `latest` and `1` task URLs 404). The `.tflite` file carries the label list and loads in `ImageSegmenter`. It is float32. There is no official int8 or float16 multiclass file. The graph names include `xeno_former` and `linear_attention`, so this is a transformer-style segmenter, which is why it is large and slow next to the other selfie model.

The binary selfie segmenter at the same bucket is 249,537 bytes and Google's published Pixel 6 CPU time is 33 ms. It only has background and person. It cannot drive a clothes recolour. Do not swap it in to make the frame rate look better.

Google's published Pixel 6 times for the multiclass model are 217.76 ms on CPU and 71.24 ms on GPU. A Pixel 6 CPU is several times the multicore capacity of 4× A53 at 2.0 GHz. The planning ceiling is **1400 ms** (about 6× the Pixel 6 CPU number, with a little room for the UNO Q's small board to warm up). The Adreno 702 is a small GPU, and this pipeline does not depend on it.

The model paints every person it sees with the same class ids. One inference covers the frame. Per-person recolour is the torso quad (shoulders and hips, shrunk 15%), not a second inference. The quad is forgotten after the track TTL (1 s, or 2.5 s while recolour is also on) so a person who has left does not keep a stale shirt window.

Grey and black shirts have no hue. The shift leaves them alone. That is the right result; a brightness change would wipe out folds, which the design explicitly keeps.

## The other models

| Model | File | Bytes | Ceiling |
|---|---|---|---|
| Face landmarker, float16, blendshapes off | `face_landmarker.task` | 3,758,596 | 400 ms for one face, +220 ms for the second |
| Pose landmarker lite, float16 | `pose_landmarker_lite.task` | 5,777,746 | 280 ms for one pose, +150 ms for the second, plus 10 ms to sample the shirt |
| SFace int8 | `face_recognition_sface_2021dec_int8.onnx` | 9,896,933 | 50 ms each, at most two |

The face ceiling comes from a public face-mesh measurement of about 330 ms on an i.MX8 with a similar amount of A53-class compute (4× A53 at 0.9 GHz plus 2× A72 at 1.2 GHz), rounded up. SFace is a 112×112 network. The shirt histogram looks at most 2,000 pixels.

Capture is budgeted at 40 ms. Recolouring a held mask across 640×480 is budgeted at 40 ms.

## The frame schedule

Name budget: **1000 ms**. Mask-refresh budget: **2000 ms**. A measured frame over 2000 ms forces one cooldown frame with no new inference.

| Frame | Work | Planning total |
|---|---|---|
| First look, one person | Capture, face, one embedding | 490 ms |
| One person, pose as well, held recolour | Those, plus pose, shirt sample, hue shift | 820 ms |
| Two people, embeddings, held recolour, no pose | Capture, face, two embeddings, hue shift | 800 ms |
| Two people plus pose | — | 1200 ms, refused |
| Clothes mask | Capture, segmenter, hue shift | 1480 ms |
| Face and segmenter together | — | About 1.8 s, refused |

Rules the code enforces:

* Recolour mode never loads a face frame. Tag mode never loads the segmenter.
* Combined mode gives the face the frame whenever the face is due. The mask waits.
* After a face pass, the next face pass waits 1 s (tag) or 2 s (combined). An unidentified person does not keep the face model running on every frame, or the mask would never get a turn. The first frame still runs the face immediately.
* Pose is added to a face frame only when the previous frame saw exactly one person and the sum still fits. Otherwise pose is its own frame.
* Enrolment of a face runs the face every frame and does not segment. Enrolment of a shirt runs pose only.
* While combined mode is on, face recheck is 2 s and the track TTL is 2.5 s, so a name survives one mask refresh. The original 1 s TTL would drop the person during the segmenter.

The pure enrolment, matching, and tracking rules in the October 15 plan are unchanged. They are ordinary Python or JavaScript and they are not what fills the second. They are not reimplemented in `unoq/` yet. The scheduler already reserves their time (`enrolment_needs`, embed stage, shirt sample, TTL). Porting them onto this loop must keep those hooks. It must not put the segmenter back on the face frame, and it must not raise `MAX_FACES` above 2.

## Memory, disk, and the desktop

* Refuse to load models when `MemAvailable` is under 700 MiB.
* On a board with under 3 GB of RAM, refuse to run while `DISPLAY` or `WAYLAND_DISPLAY` is set. That is the 2 GB SKU with a monitor session. Use the 4 GB board (ABX00173) for a local desktop, or stop the graphical session and run headless over SSH. The pipeline's own UI is the terminal log.
* `install.sh` refuses to start when the filesystem has under 1024 MiB free. App Lab layouts often leave `/home/arduino` near 3.6 GB, and a virtualenv plus App Lab containers can fill it. `/opt` is the usual escape, after `arduino-app-cli system cleanup`.
* Model files together are about 36 MB. The tight resource is the virtualenv and the desktop, not the checkpoint files.

## What was verified, and what a board still has to show

Verified without a UNO Q on the bench:

* The model URLs, sizes, and label list.
* MediaPipe 1.1.1 CPU inference of the segmenter, face landmarker, and pose landmarker, including the EGL dependency and the OpenCV packaging clash.
* The schedule's arithmetic, the hue shift, the torso quad, the histogram bins, the camera-node filter, and the host gates. `python3 -m unittest discover -s unoq/tests -v`.

Not verified, because this session is not a QRB2210:

* Wall-clock time of a real face and a real shirt on the A53s.
* Whether the 1.1.1 aarch64 wheel imports under the board's Python 3.13 without a rebuild. The wheel tag and the release notes say it should. If `import mediapipe` fails, the fallback is a Python 3.12 virtualenv, still with `--no-deps`, not an older MediaPipe.
* Thermal behaviour after several minutes. The cooldown exists for that, and it is not a measurement.

On the board, after `sh install.sh`:

```sh
. .venv/bin/activate
python -m face_shirt --run --mode both --frames 30
```

Read `spent=` against `estimate=`. One person, face frame, should stay under 1000 ms. A segment frame should stay under 2000 ms. If the segmenter is past 2000 ms the loop still runs: it cools down on the next frame and keeps the last mask. That is the degradation, not a crash and not a silent drop of the clothes class.
