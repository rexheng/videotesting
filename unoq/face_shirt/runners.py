"""MediaPipe and OpenCV runners. Imported only when a model is loaded."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from face_shirt.config import (
    DELEGATE,
    FACE_FILE,
    MAX_FACES,
    MAX_POSES,
    POSE_FILE,
    SEGMENT_FILE,
    SFACE_FILE,
    SFACE_SIZE,
)
from face_shirt.geometry import SFACE_TEMPLATE, estimate_similarity, five_points
from face_shirt.segment import category_mask_array


def _rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame_bgr[:, :, ::-1])


class VisionRunners:
    """CPU-only landmarkers and the clothes segmenter.

    Each model keeps its own clock. MediaPipe rejects a video timestamp
    that does not increase.
    """

    def __init__(self, model_dir: Path, *, need_face: bool, need_segment: bool):
        if DELEGATE != "CPU":
            raise RuntimeError("refusing to construct a GPU delegate")
        self.model_dir = Path(model_dir)
        self._mp = None
        self._face = None
        self._pose = None
        self._segment = None
        self._recogniser = None
        self._last_ts = {"face": -1, "pose": -1, "segment": -1}
        self._load(need_face=need_face, need_segment=need_segment)

    def _import_mediapipe(self):
        if self._mp is not None:
            return self._mp
        try:
            import mediapipe as mp
        except OSError as exc:
            raise RuntimeError(
                "MediaPipe failed to load its native library. On the UNO Q install "
                "libegl1 and libgles2, then try again. "
                f"Original error: {exc}"
            ) from exc
        self._mp = mp
        return mp

    def _base(self, filename: str):
        mp = self._import_mediapipe()
        path = self.model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path.name}. Run `python -m face_shirt --fetch` on the board."
            )
        options = mp.tasks.BaseOptions(
            model_asset_path=str(path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        return mp, options

    def _load(self, *, need_face: bool, need_segment: bool) -> None:
        if need_face:
            mp, base = self._base(FACE_FILE)
            self._face = mp.tasks.vision.FaceLandmarker.create_from_options(
                mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=base,
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_faces=MAX_FACES,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
            )
            mp, pose_base = self._base(POSE_FILE)
            self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(
                mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=pose_base,
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_poses=MAX_POSES,
                )
            )
        if need_segment:
            mp, base = self._base(SEGMENT_FILE)
            self._segment = mp.tasks.vision.ImageSegmenter.create_from_options(
                mp.tasks.vision.ImageSegmenterOptions(
                    base_options=base,
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    output_category_mask=True,
                    output_confidence_masks=False,
                )
            )
            labels = list(self._segment.labels)
            expected = ["background", "hair", "body-skin", "face-skin", "clothes", "others"]
            if labels != expected:
                raise RuntimeError(f"clothes model labels are {labels}, expected {expected}")
        sface = self.model_dir / SFACE_FILE
        if sface.is_file():
            import cv2

            self._recogniser = cv2.FaceRecognizerSF.create(str(sface), "")

    def _stamp(self, key: str, timestamp_ms: int) -> int:
        timestamp_ms = max(1, int(timestamp_ms))
        if timestamp_ms <= self._last_ts[key]:
            timestamp_ms = self._last_ts[key] + 1
        self._last_ts[key] = timestamp_ms
        return timestamp_ms

    def _image(self, frame_bgr: np.ndarray):
        mp = self._import_mediapipe()
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=_rgb(frame_bgr))

    def face(self, frame_bgr: np.ndarray, timestamp_ms: int):
        if self._face is None:
            return []
        stamp = self._stamp("face", timestamp_ms)
        result = self._face.detect_for_video(self._image(frame_bgr), stamp)
        return list(result.face_landmarks)

    def pose(self, frame_bgr: np.ndarray, timestamp_ms: int):
        if self._pose is None:
            return []
        stamp = self._stamp("pose", timestamp_ms)
        result = self._pose.detect_for_video(self._image(frame_bgr), stamp)
        return list(result.pose_landmarks)

    def segment(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        if self._segment is None:
            raise RuntimeError("clothes segmenter is not loaded")
        stamp = self._stamp("segment", timestamp_ms)
        result = self._segment.segment_for_video(self._image(frame_bgr), stamp)
        return category_mask_array(result.category_mask)

    def embed(self, frame_bgr: np.ndarray, timestamp_ms: int) -> int:
        if self._recogniser is None or self._face is None:
            return 0
        # The face stage already ran in this frame when the scheduler asked
        # for an embedding. Re-reading it would advance the landmarker clock
        # a second time, so embeddings are computed from the crop helper
        # only when the caller passes landmarks through embed_landmarks.
        del timestamp_ms
        return 0

    def embed_landmarks(self, frame_bgr: np.ndarray, landmarks) -> np.ndarray | None:
        if self._recogniser is None:
            return None
        import cv2

        points = five_points(landmarks, frame_bgr.shape[1], frame_bgr.shape[0])
        transform = estimate_similarity(points, SFACE_TEMPLATE)
        matrix = np.array(
            [
                [transform["a"], transform["c"], transform["e"]],
                [transform["b"], transform["d"], transform["f"]],
            ],
            dtype=np.float32,
        )
        aligned = cv2.warpAffine(frame_bgr, matrix, (SFACE_SIZE, SFACE_SIZE))
        feature = self._recogniser.feature(aligned)
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    def close(self) -> None:
        for model in (self._face, self._pose, self._segment):
            if model is not None:
                model.close()
        self._face = self._pose = self._segment = None

    def cost_ms(self, stage: str) -> int | None:
        del stage
        return None
