"""Reference SFace embeddings for the face fixtures, computed with OpenCV's own pipeline.

Detects each face with FaceDetectorYN (YuNet), aligns and embeds it with FaceRecognizerSF
(fp32 SFace), L2-normalises, writes tests/fixtures/faces/reference.json and prints the
pairwise cosines. The browser pipeline is checked against this file by
tests/e2e/faceEmbedding.spec.js.

Setup (the venv and .cache/ are git-ignored):
    python3 -m venv .venv && .venv/bin/pip install opencv-python-headless numpy
    .venv/bin/python scripts/reference_embeddings.py

Missing model files are downloaded into .cache/models/ on first run.
"""

import itertools
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FACES = ROOT / "tests" / "fixtures" / "faces"
NAMES = ["a1", "a2", "b1"]
CACHE = ROOT / ".cache" / "models"
MODELS = {
    "yunet": (
        "face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    ),
    "sface": (
        "face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    ),
}


def model_path(key):
    filename, url = MODELS[key]
    path = CACHE / filename
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return str(path)


def embed(detector, recognizer, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Cannot read {image_path}")
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        raise SystemExit(f"No face found in {image_path}")
    face = max(faces, key=lambda f: f[-1])  # highest detection score
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned).flatten().astype(np.float64)
    return feature / np.linalg.norm(feature)


def main():
    detector = cv2.FaceDetectorYN.create(model_path("yunet"), "", (320, 320), 0.9)
    recognizer = cv2.FaceRecognizerSF.create(model_path("sface"), "")
    embeddings = {name: embed(detector, recognizer, FACES / f"{name}.jpg") for name in NAMES}

    out = FACES / "reference.json"
    out.write_text(json.dumps({name: [round(float(v), 7) for v in vec] for name, vec in embeddings.items()}) + "\n")
    print(f"OpenCV {cv2.__version__}: wrote {out.relative_to(ROOT)}")
    for first, second in itertools.combinations(NAMES, 2):
        print(f"cos({first}, {second}) = {float(embeddings[first] @ embeddings[second]):.4f}")


if __name__ == "__main__":
    main()
