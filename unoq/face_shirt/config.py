"""Budgets and model pins for the Arduino UNO Q (Qualcomm QRB2210).

The millisecond figures are planning ceilings for the quad Cortex-A53 at
2.0 GHz, not measurements from a board. They are derived in
docs/superpowers/specs/2026-10-24-uno-q-segmentation-audit.md.
The running loop replaces them with measured stage times.
"""

INFERENCE_TARGET = "qrb2210-mpu"
# MediaPipe's Python GPU delegate is documented as Ubuntu-only. The UNO Q
# image is Debian, so the supported path is the CPU / XNNPACK delegate.
DELEGATE = "CPU"

MEDIAPIPE_VERSION = "1.1.1"
OPENCV_HEADLESS_VERSION = "4.11.0.86"
MIN_PYTHON = (3, 11)

# Capture. MJPG at this size stays inside USB 2.0 bandwidth on a hub.
# Do not raise it: the landmarkers and the segmenter do not need 720p.
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FOURCC = "MJPG"

MAX_FACES = 2
MAX_POSES = 2
MAX_EMBEDS = 2

# Planning ceilings, milliseconds, one person unless noted.
CAPTURE_MS = 40
FACE_MS = 400
FACE_EXTRA_MS = 220
POSE_MS = 280
POSE_EXTRA_MS = 150
EMBED_MS = 50
SHIRT_SAMPLE_MS = 10
SEGMENT_MS = 1400
RECOLOR_MS = 40

# A signed-up person must be nameable inside this window. Segmentation is
# not allowed to spend any of it.
NAME_BUDGET_MS = 1000
# One clothes-mask refresh may take this long. It never shares a frame
# with the face landmarker.
SEGMENT_FRAME_BUDGET_MS = 2000
# A measured frame slower than this forces an inference-free cooldown.
FRAME_HARD_MS = 2000

FACE_INTERVAL_MS = 1000
# Combined tagging + recolor. A mask refresh takes longer than the
# original one-second recheck, so the face recheck waits long enough for
# the mask to get a frame.
FACE_INTERVAL_BOTH_MS = 2000
POSE_INTERVAL_MS = 1000
SEGMENT_INTERVAL_MS = 1500
TRACK_TTL_MS = 1000
TRACK_TTL_BOTH_MS = 2500

# Host gates.
MIN_FREE_RAM_MB = 700
DESKTOP_RAM_FLOOR_MB = 3000
MIN_FREE_DISK_MB = 1024

# Shirt descriptor, same numbers as the design spec.
SHOULDER_VISIBILITY = 0.6
HIP_VISIBILITY = 0.5
MIN_TORSO_PX = 40
TORSO_SHRINK = 0.15
MAX_SHIRT_SAMPLES = 2000
HISTOGRAM_BINS = 28

# Selfie multiclass. Class ids are the model's label list, confirmed by
# loading selfie_multiclass_256x256.tflite.
SEGMENT_SIZE = 256
CLASS_BACKGROUND = 0
CLASS_HAIR = 1
CLASS_BODY_SKIN = 2
CLASS_FACE_SKIN = 3
CLASS_CLOTHES = 4
CLASS_OTHER = 5
CLASS_NAMES = (
    "background",
    "hair",
    "body-skin",
    "face-skin",
    "clothes",
    "others",
)

SFACE_SIZE = 112
HUE_SHIFT_DEGREES = 120

# Byte sizes checked against the files fetched on 2026-10-24.
SEGMENT_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite"
)
SEGMENT_FILE = "selfie_multiclass_256x256.tflite"
SEGMENT_BYTES = 16_371_837

FACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_FILE = "face_landmarker.task"
FACE_BYTES = 3_758_596

POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_FILE = "pose_landmarker_lite.task"
POSE_BYTES = 5_777_746

SFACE_URL = (
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/"
    "face_recognition_sface_2021dec_int8.onnx"
)
SFACE_FILE = "face_recognition_sface_2021dec_int8.onnx"
SFACE_BYTES = 9_896_933

# The binary selfie model has no clothes class. It is recorded here so a
# later change does not "speed the pipeline up" by swapping it in.
BINARY_SELFIE_BYTES = 249_537

MCU_SRAM_BYTES = 786 * 1024
MCU_FRAME_BYTES = CAPTURE_WIDTH * CAPTURE_HEIGHT * 3
BRIDGE_TEXT_MAX = 30
