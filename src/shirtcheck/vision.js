// Camera-side models for Shirt Check: a face landmarker (is there a face?), a pose landmarker
// (where is the torso?) and the selfie multiclass segmenter (skin or cloth in that torso?).
// SFace is not loaded. This page does not recognise who someone is, only whether the shirt is off.
//
// The segmenter always runs on the CPU. On iOS Safari the GPU delegate has been seen to scramble
// the category numbers (background stays 0, but skin and cloth swap), which would make the call
// wrong. Face and pose still try the GPU and fall back to the CPU if that delegate fails.

import { torsoQuad } from '../recognition/shirt.js';
import { isFacePresent, judgeMask, scaleQuad } from './bare.js';

const FACE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const SEGMENTER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite';

async function createTask(Task, fileset, modelAssetPath, options) {
  try {
    return await Task.createFromOptions(fileset, {
      ...options,
      baseOptions: { modelAssetPath, delegate: 'GPU' },
    });
  } catch {
    return Task.createFromOptions(fileset, {
      ...options,
      baseOptions: { modelAssetPath, delegate: 'CPU' },
    });
  }
}

/** Download the face, pose and shirt-or-skin models. Rejects with the original error on failure. */
export async function loadShirtCheckModels({ runningMode = 'VIDEO' } = {}) {
  const { FaceLandmarker, PoseLandmarker, ImageSegmenter, FilesetResolver, MEDIAPIPE_WASM_URL } =
    await import('../vision/libs.js');
  const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);

  const settled = await Promise.allSettled([
    createTask(FaceLandmarker, fileset, FACE_MODEL_URL, {
      runningMode,
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    }),
    createTask(PoseLandmarker, fileset, POSE_MODEL_URL, {
      runningMode,
      numPoses: 1,
    }),
    ImageSegmenter.createFromOptions(fileset, {
      runningMode,
      outputCategoryMask: true,
      outputConfidenceMasks: false,
      baseOptions: { modelAssetPath: SEGMENTER_MODEL_URL, delegate: 'CPU' },
    }),
  ]);

  const failure = settled.find((result) => result.status === 'rejected');
  if (failure) {
    for (const result of settled) {
      if (result.status !== 'fulfilled') continue;
      try {
        result.value.close?.();
      } catch {
        // Already failing; report the original error below.
      }
    }
    throw failure.reason;
  }
  const [face, pose, segmenter] = settled.map((result) => result.value);
  return { face, pose, segmenter };
}

function normalizedBox(landmarks) {
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const point of landmarks) {
    if (point.x < minX) minX = point.x;
    if (point.y < minY) minY = point.y;
    if (point.x > maxX) maxX = point.x;
    if (point.y > maxY) maxY = point.y;
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

function readTorso(poseLandmarks, width, height, segment) {
  const quad = torsoQuad(poseLandmarks, width, height);
  if (!quad) return { polygon: null, reading: null };
  const polygon = quad.map(([x, y]) => [x / width, y / height]);
  const segmented = segment();
  try {
    const mask = segmented?.categoryMask;
    if (!mask) return { polygon, reading: null };
    const scaled = scaleQuad(quad, width, height, mask.width, mask.height);
    return { polygon, reading: judgeMask(mask.getAsUint8Array(), mask.width, mask.height, scaled) };
  } finally {
    segmented?.close?.();
  }
}

/**
 * Live detector. `detect(video, now)` returns
 * { face, faceBox, polygon, reading } in normalised frame coordinates.
 * Timestamps passed to MediaPipe are strictly increasing.
 */
export function createShirtDetector(models) {
  let lastTimestamp = -Infinity;

  function timestamp(now) {
    lastTimestamp = Math.max(lastTimestamp + 1, Math.round(now));
    return lastTimestamp;
  }

  return {
    detect(video, now) {
      const width = video.videoWidth;
      const height = video.videoHeight;
      if (!(width > 0) || !(height > 0)) {
        return { face: false, faceBox: null, polygon: null, reading: null };
      }
      const ts = timestamp(now);
      const faces = models.face.detectForVideo(video, ts);
      const poses = models.pose.detectForVideo(video, ts);
      const faceLandmarks = faces.faceLandmarks?.[0];
      const faceBox = faceLandmarks ? normalizedBox(faceLandmarks) : null;
      const poseLandmarks = poses.landmarks?.[0];
      const torso = poseLandmarks
        ? readTorso(poseLandmarks, width, height, () => models.segmenter.segmentForVideo(video, ts))
        : { polygon: null, reading: null };
      return { face: isFacePresent(faceBox), faceBox, ...torso };
    },
  };
}

/**
 * One still image (IMAGE running mode). Used by the browser test to check a known portrait.
 * `source` is an HTMLImageElement (or anything MediaPipe accepts) with naturalWidth/naturalHeight.
 */
export function classifyStill(models, source) {
  const width = source.naturalWidth || source.videoWidth || source.width;
  const height = source.naturalHeight || source.videoHeight || source.height;
  const poses = models.pose.detect(source);
  const poseLandmarks = poses.landmarks?.[0];
  if (!poseLandmarks) return { pose: false, torso: false, reading: null };
  const torso = readTorso(poseLandmarks, width, height, () => models.segmenter.segment(source));
  return { pose: true, torso: !!torso.polygon, reading: torso.reading, polygon: torso.polygon };
}
