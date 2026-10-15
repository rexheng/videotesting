// SFace face embeddings: align a face to 112x112, run the ONNX model, L2-normalise.
// The pure helpers (toInputTensorData, l2normalize) run in Node; the rest needs a browser canvas.
// onnxruntime is imported lazily inside embedFace so this module never loads a CDN at import time.
import { SFACE_TEMPLATE, fivePoints, estimateSimilarity } from './geometry.js';

const SIZE = 112;
const PLANE = SIZE * SIZE;

/** RGBA pixels of a 112x112 crop -> SFace input: planar RGB (NCHW), raw 0-255 values, no normalisation. */
export function toInputTensorData(rgba) {
  const out = new Float32Array(3 * PLANE);
  for (let i = 0; i < PLANE; i++) {
    out[i] = rgba[i * 4];
    out[PLANE + i] = rgba[i * 4 + 1];
    out[2 * PLANE + i] = rgba[i * 4 + 2];
  }
  return out;
}

/** Copy of `vec` scaled to unit length; an all-zero vector stays all zeros. */
export function l2normalize(vec) {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
  const norm = Math.sqrt(sum) || 1;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

/**
 * Draw the face onto `canvas112` (112x112) warped so its eyes, nose and mouth corners sit on
 * SFACE_TEMPLATE, as OpenCV's FaceRecognizerSF.alignCrop does. `source` is the raw, unmirrored
 * frame (image, video or canvas) of size width x height; areas outside it are black.
 * Returns the canvas's 2D context. Throws if the landmarks are degenerate.
 */
export function alignFace(source, landmarks, width, height, canvas112) {
  const m = estimateSimilarity(fivePoints(landmarks, width, height), SFACE_TEMPLATE);
  if (![m.a, m.b, m.c, m.d, m.e, m.f].every(Number.isFinite)) {
    throw new Error('Cannot align face: degenerate landmarks');
  }
  if (canvas112.width !== SIZE) canvas112.width = SIZE;
  if (canvas112.height !== SIZE) canvas112.height = SIZE;
  const ctx = canvas112.getContext('2d', { willReadFrequently: true });
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
  ctx.drawImage(source, 0, 0, width, height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return ctx;
}

let scratchCanvas = null;

function getScratchCanvas() {
  scratchCanvas ??=
    typeof OffscreenCanvas === 'function' ? new OffscreenCanvas(SIZE, SIZE) : document.createElement('canvas');
  return scratchCanvas;
}

/** 128-value, L2-normalised SFace embedding of the face described by `landmarks` in `source`. */
export async function embedFace(session, source, landmarks, width, height) {
  const { ort } = await import('./libs.js');
  const ctx = alignFace(source, landmarks, width, height, getScratchCanvas());
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  const input = new ort.Tensor('float32', toInputTensorData(data), [1, 3, SIZE, SIZE]);
  const outputs = await session.run({ [session.inputNames[0]]: input });
  return l2normalize(outputs[session.outputNames[0]].data);
}
