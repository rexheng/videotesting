// Loads the three models: MediaPipe Face Landmarker, MediaPipe Pose Landmarker, and SFace (ONNX).
import { MAX_FACES, MAX_POSES } from '../config.js';

/** SFace face-recognition model, relative to the app root. */
export const SFACE_MODEL_URL = 'models/face_recognition_sface_2021dec.onnx';

const FACE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

// The app root is two folders up from this file, so pages in sub-folders (test harnesses) work too.
const appRoot = new URL('../../', import.meta.url);

/** Create a landmarker on the GPU, or on the CPU if the GPU delegate cannot be created. */
async function createLandmarker(Landmarker, fileset, modelAssetPath, options) {
  try {
    return await Landmarker.createFromOptions(fileset, {
      ...options,
      baseOptions: { modelAssetPath, delegate: 'GPU' },
    });
  } catch {
    return Landmarker.createFromOptions(fileset, {
      ...options,
      baseOptions: { modelAssetPath, delegate: 'CPU' },
    });
  }
}

/**
 * Download and initialise all models. Rejects with the original error if any of them fails.
 * `sfaceUrl` (relative to the app root) swaps in another SFace file, for testing.
 */
export async function loadModels({ runningMode = 'VIDEO', sfaceUrl = SFACE_MODEL_URL } = {}) {
  const { FaceLandmarker, PoseLandmarker, FilesetResolver, MEDIAPIPE_WASM_URL, ort } = await import('./libs.js');
  const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);

  const settled = await Promise.allSettled([
    createLandmarker(FaceLandmarker, fileset, FACE_MODEL_URL, {
      runningMode,
      numFaces: MAX_FACES,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    }),
    createLandmarker(PoseLandmarker, fileset, POSE_MODEL_URL, {
      runningMode,
      numPoses: MAX_POSES,
    }),
    ort.InferenceSession.create(new URL(sfaceUrl, appRoot).href, {
      executionProviders: ['wasm'],
      // Errors only: SFace triggers hundreds of harmless "initializer appears in graph inputs" warnings.
      logSeverityLevel: 3,
    }),
  ]);

  const failure = settled.find((result) => result.status === 'rejected');
  if (failure) {
    // Free whatever did load, so a retry does not pile up WebGL contexts and Wasm memory.
    // Landmarkers have close(); the ONNX session has release().
    for (const result of settled) {
      if (result.status !== 'fulfilled') continue;
      const model = result.value;
      try {
        await (model.close ? model.close() : model.release());
      } catch {
        // Already failing; report the original error below.
      }
    }
    throw failure.reason;
  }
  const [face, pose, sface] = settled.map((result) => result.value);
  return { face, pose, sface };
}
