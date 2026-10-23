// Shirt descriptor: find the torso from MediaPipe Pose landmarks and summarise the shirt as a
// 28-bin colour histogram that can be compared, averaged and blended.
// Pure module: no DOM, no CDN imports. Pose landmarks are normalised (0-1); torsoQuad converts
// them to pixels of the raw, unmirrored frame, and every pixel-space function works in that space.

import {
  SHOULDER_VISIBILITY,
  HIP_VISIBILITY,
  MIN_TORSO_PX,
  TORSO_SHRINK,
  MAX_SHIRT_SAMPLES,
} from '../config.js';
import { bboxOfPoints } from '../vision/geometry.js';

// Histogram layout: bins 0-3 are greys (by value), bins 4-27 are 12 hue sectors x {muted, vivid}.
const BIN_COUNT = 28;
const GREY_BINS = 4;
const HUE_SECTORS = 12;
const GREY_SATURATION = 0.2;
const GREY_VALUE = 0.15;
const VIVID_SATURATION = 0.5;

// MediaPipe Pose landmark indices.
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_HIP = 23;
const RIGHT_HIP = 24;

/**
 * Torso as 4 pixel corners [shoulder 11, shoulder 12, hip 24, hip 23], shrunk towards its
 * centroid so the sample stays on the shirt. Hips that are not visible (either one) are both
 * estimated one shoulder-width straight below their shoulder. Returns null when a shoulder is
 * not visible enough or the result is smaller than MIN_TORSO_PX in either direction.
 */
export function torsoQuad(poseLandmarks, width, height) {
  const leftShoulder = poseLandmarks[LEFT_SHOULDER];
  const rightShoulder = poseLandmarks[RIGHT_SHOULDER];
  const leftHip = poseLandmarks[LEFT_HIP];
  const rightHip = poseLandmarks[RIGHT_HIP];
  // A landmark with no visibility score counts as not visible.
  const isSeen = (landmark, threshold) => (landmark.visibility ?? 0) >= threshold;
  if (!isSeen(leftShoulder, SHOULDER_VISIBILITY) || !isSeen(rightShoulder, SHOULDER_VISIBILITY)) {
    return null;
  }

  const toPixels = (landmark) => [landmark.x * width, landmark.y * height];
  const shoulder11 = toPixels(leftShoulder);
  const shoulder12 = toPixels(rightShoulder);

  let hip24;
  let hip23;
  if (isSeen(leftHip, HIP_VISIBILITY) && isSeen(rightHip, HIP_VISIBILITY)) {
    hip24 = toPixels(rightHip);
    hip23 = toPixels(leftHip);
  } else {
    const shoulderWidth = Math.hypot(shoulder11[0] - shoulder12[0], shoulder11[1] - shoulder12[1]);
    hip24 = [shoulder12[0], shoulder12[1] + shoulderWidth];
    hip23 = [shoulder11[0], shoulder11[1] + shoulderWidth];
  }

  const corners = [shoulder11, shoulder12, hip24, hip23];
  const centroidX = corners.reduce((sum, [x]) => sum + x, 0) / 4;
  const centroidY = corners.reduce((sum, [, y]) => sum + y, 0) / 4;
  const keep = 1 - TORSO_SHRINK;
  const quad = corners.map(([x, y]) => [
    centroidX + (x - centroidX) * keep,
    centroidY + (y - centroidY) * keep,
  ]);

  const box = bboxOfPoints(quad);
  if (!(box.width >= MIN_TORSO_PX) || !(box.height >= MIN_TORSO_PX)) return null;
  return quad;
}

/** RGB (0-255 each) to [h 0-360, s 0-1, v 0-1]. Hue is always below 360; greys get hue 0. */
export function rgbToHsv(r, g, b) {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;

  let h = 0;
  if (delta !== 0) {
    if (max === rn) h = ((gn - bn) / delta) % 6;
    else if (max === gn) h = (bn - rn) / delta + 2;
    else h = (rn - gn) / delta + 4;
    h *= 60;
    if (h < 0) h += 360;
    // A tiny negative hue rounds up to exactly 360 on the line above; wrap it back to 0.
    if (h >= 360) h = 0;
  }
  return [h, max === 0 ? 0 : delta / max, max];
}

/** Histogram bin 0-27 for one pixel. */
function binOf(r, g, b) {
  const [h, s, v] = rgbToHsv(r, g, b);
  if (s < GREY_SATURATION || v < GREY_VALUE) {
    if (v < 0.25) return 0;
    if (v < 0.5) return 1;
    if (v < 0.75) return 2;
    return 3;
  }
  const sector = Math.min(HUE_SECTORS - 1, Math.floor(h / (360 / HUE_SECTORS)));
  return GREY_BINS + sector * 2 + (s < VIVID_SATURATION ? 0 : 1);
}

/** True when [px, py] is inside or on the edge of the convex quad, whichever way it is wound. */
export function insideConvexQuad(quad, px, py) {
  let sawPositive = false;
  let sawNegative = false;
  for (let i = 0; i < quad.length; i++) {
    const [ax, ay] = quad[i];
    const [bx, by] = quad[(i + 1) % quad.length];
    const cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    if (cross > 0) sawPositive = true;
    else if (cross < 0) sawNegative = true;
    if (sawPositive && sawNegative) return false;
  }
  return true;
}

/** Twice the signed area of the quad (shoelace); 0 for a degenerate quad. */
function quadArea2(quad) {
  let area2 = 0;
  for (let i = 0; i < quad.length; i++) {
    const [ax, ay] = quad[i];
    const [bx, by] = quad[(i + 1) % quad.length];
    area2 += ax * by - bx * ay;
  }
  return area2;
}

/** Scale so the values sum to 1 (an all-zero input stays all zero). Returns a new Float32Array. */
function normalised(values) {
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += values[i];
  const out = new Float32Array(values.length);
  if (sum > 0) for (let i = 0; i < values.length; i++) out[i] = values[i] / sum;
  return out;
}

/**
 * Colour histogram of the pixels inside the (convex) quad, sampled on a stride grid of at most
 * MAX_SHIRT_SAMPLES points. `rgba` is a width*height*4 buffer as from canvas getImageData.
 * Returns { histogram: Float32Array(28) summing to 1, swatch: '#rrggbb' } where the swatch is the
 * mean colour of the fullest bin, or null when no pixel was sampled.
 */
export function shirtHistogram(rgba, width, height, quad) {
  if (!(Math.abs(quadArea2(quad)) > 0)) return null;

  const box = bboxOfPoints(quad);
  const x0 = Math.max(0, Math.floor(box.x));
  const y0 = Math.max(0, Math.floor(box.y));
  const x1 = Math.min(width, Math.ceil(box.x + box.width));
  const y1 = Math.min(height, Math.ceil(box.y + box.height));
  if (!(x1 > x0) || !(y1 > y0)) return null;

  const spanX = x1 - x0;
  const spanY = y1 - y0;
  let stride = Math.max(1, Math.ceil(Math.sqrt((spanX * spanY) / MAX_SHIRT_SAMPLES)));
  while (Math.ceil(spanX / stride) * Math.ceil(spanY / stride) > MAX_SHIRT_SAMPLES) stride++;

  const counts = new Float64Array(BIN_COUNT);
  const sums = Array.from({ length: BIN_COUNT }, () => [0, 0, 0]);
  let total = 0;
  for (let y = y0; y < y1; y += stride) {
    for (let x = x0; x < x1; x += stride) {
      // Test the pixel's centre so a full-frame quad includes every edge pixel.
      if (!insideConvexQuad(quad, x + 0.5, y + 0.5)) continue;
      const i = (y * width + x) * 4;
      const r = rgba[i];
      const g = rgba[i + 1];
      const b = rgba[i + 2];
      const bin = binOf(r, g, b);
      counts[bin]++;
      sums[bin][0] += r;
      sums[bin][1] += g;
      sums[bin][2] += b;
      total++;
    }
  }
  if (total === 0) return null;

  let fullest = 0;
  for (let bin = 1; bin < BIN_COUNT; bin++) if (counts[bin] > counts[fullest]) fullest = bin;
  const hex = (sum) => Math.round(sum / counts[fullest]).toString(16).padStart(2, '0');
  const swatch = `#${hex(sums[fullest][0])}${hex(sums[fullest][1])}${hex(sums[fullest][2])}`;

  return { histogram: normalised(counts), swatch };
}

/** Bhattacharyya coefficient of two normalised histograms: 1 = identical, 0 = disjoint. */
export function histogramSimilarity(p, q) {
  let sum = 0;
  const length = Math.min(p.length, q.length);
  for (let i = 0; i < length; i++) sum += Math.sqrt(p[i] * q[i]);
  return sum;
}

/** Normalised mean of a list of histograms. */
export function averageHistograms(list) {
  const total = new Float64Array(list[0]?.length ?? BIN_COUNT);
  for (const histogram of list) {
    for (let i = 0; i < total.length; i++) total[i] += histogram[i];
  }
  return normalised(total);
}

/** normalise((1 - weight) * old + weight * fresh): nudge a stored histogram towards a new one. */
export function blendHistograms(old, fresh, weight) {
  const mixed = new Float64Array(old.length);
  for (let i = 0; i < mixed.length; i++) mixed[i] = (1 - weight) * old[i] + weight * fresh[i];
  return normalised(mixed);
}
