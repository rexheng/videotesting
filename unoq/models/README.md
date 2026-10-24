Model files are downloaded by `python -m face_shirt --fetch`. They are not committed.

| File | Role | Bytes |
|---|---|---|
| `selfie_multiclass_256x256.tflite` | Clothes, hair, skin, background | 16371837 |
| `face_landmarker.task` | Face landmarks, at most 2 faces | 3758596 |
| `pose_landmarker_lite.task` | Shoulders and hips for the torso | 5777746 |
| `face_recognition_sface_2021dec_int8.onnx` | 128-d face embedding | 9896933 |

The MediaPipe models are Apache-2.0. SFace is the OpenCV Zoo model, also Apache-2.0. The binary selfie segmenter is intentionally not used: it has no clothes class.
