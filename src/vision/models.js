// Loads the three models: MediaPipe Face Landmarker, MediaPipe Pose Landmarker, and SFace (ONNX).
import { MAX_FACES, MAX_POSES } from '../config.js';

/**
 * SFace face-recognition model, stored as 4 MiB parts (relative to the app root). The whole file is
 * 38.7 MB. Cloudflare Pages allows 25 MiB per file, and a Workers static-assets upload allows 5 MiB
 * per file, so neither limit can hold it in one piece. The browser downloads every part and joins them.
 */
export const SFACE_MODEL_PARTS = Object.freeze([
  'models/face_recognition_sface_2021dec.onnx.part01',
  'models/face_recognition_sface_2021dec.onnx.part02',
  'models/face_recognition_sface_2021dec.onnx.part03',
  'models/face_recognition_sface_2021dec.onnx.part04',
  'models/face_recognition_sface_2021dec.onnx.part05',
  'models/face_recognition_sface_2021dec.onnx.part06',
  'models/face_recognition_sface_2021dec.onnx.part07',
  'models/face_recognition_sface_2021dec.onnx.part08',
  'models/face_recognition_sface_2021dec.onnx.part09',
  'models/face_recognition_sface_2021dec.onnx.part10',
]);
/** Size and SHA-256 of the joined model, checked after download so a bad or half-cached part is caught. */
export const SFACE_MODEL_BYTES = 38696353;
export const SFACE_MODEL_SHA256 = '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79';

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

/** Join ArrayBuffers end to end, in order, into one Uint8Array. */
export function concatParts(buffers) {
  const joined = new Uint8Array(buffers.reduce((total, buffer) => total + buffer.byteLength, 0));
  let offset = 0;
  for (const buffer of buffers) {
    joined.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  }
  return joined;
}

const toHex = (bytes) => Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');

/**
 * Download the SFace parts in parallel and join them. For the shipped model (the default parts) the
 * joined file must have the expected size and SHA-256; custom parts, used in tests, are not checked.
 */
async function fetchSfaceBytes(parts) {
  const buffers = await Promise.all(
    parts.map(async (part) => {
      const response = await fetch(new URL(part, appRoot).href);
      if (!response.ok) throw new Error(`Could not download ${part} (HTTP ${response.status}).`);
      return response.arrayBuffer();
    }),
  );
  const bytes = concatParts(buffers);
  if (parts === SFACE_MODEL_PARTS) {
    if (bytes.length !== SFACE_MODEL_BYTES) {
      throw new Error(`The face model is corrupt: expected ${SFACE_MODEL_BYTES} bytes, got ${bytes.length}.`);
    }
    const sha256 = toHex(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)));
    if (sha256 !== SFACE_MODEL_SHA256) {
      throw new Error('The face model is corrupt: its SHA-256 does not match the expected value.');
    }
  }
  return bytes;
}

/**
 * Download and initialise all models. Rejects with the original error if any of them fails.
 * `sfaceParts` (URLs relative to the app root, joined in order) swaps in another SFace file, for testing.
 */
export async function loadModels({ runningMode = 'VIDEO', sfaceParts = SFACE_MODEL_PARTS } = {}) {
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
    fetchSfaceBytes(sfaceParts).then((bytes) =>
      ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        // Errors only: SFace triggers hundreds of harmless "initializer appears in graph inputs" warnings.
        logSeverityLevel: 3,
      }),
    ),
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
