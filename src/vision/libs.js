// The only module that imports from a CDN. Version numbers here must match VERSIONS in
// src/config.js (a unit test checks this). Import this module lazily (`await import('./libs.js')`)
// so a failed CDN download rejects a promise instead of breaking the importing page.
export {
  FaceLandmarker,
  PoseLandmarker,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';

/** Folder holding MediaPipe's Wasm runtime, for FilesetResolver.forVisionTasks. */
export const MEDIAPIPE_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';

// onnxruntime-web fetches its Wasm runtime from here.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

export { ort };
