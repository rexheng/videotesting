// Decide whether a torso is bare skin or still clothed, from a selfie-multiclass category mask.
// Pure: no DOM and no CDN imports. The mask is one category id per pixel, row-major:
//   0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 other.
// Skin inside the torso quad counts as bare (body-skin, plus face-skin on the neck). Clothes do not.
// The quad is in the mask's own pixel space. torsoQuad() from the shirt module is in frame pixels;
// scaleQuad maps one onto the other when the mask size differs from the camera frame.

import {
  SHIRTCHECK_SHIRTLESS_AT,
  SHIRTCHECK_DRESSED_AT,
  SHIRTCHECK_MIN_SAMPLES,
  SHIRTCHECK_MIN_FACE,
  MAX_SHIRT_SAMPLES,
} from '../config.js';
import { bboxOfPoints } from '../vision/geometry.js';
import { insideConvexQuad } from '../recognition/shirt.js';

const BODY_SKIN = 2;
const FACE_SKIN = 3;
const CLOTHES = 4;

/** Map a quad from one pixel space onto another (the mask is not always the frame size). */
export function scaleQuad(quad, fromWidth, fromHeight, toWidth, toHeight) {
  const sx = toWidth / fromWidth;
  const sy = toHeight / fromHeight;
  return quad.map(([x, y]) => [x * sx, y * sy]);
}

/** A face box (normalised) is large enough to count as a person in front of the camera. */
export function isFacePresent(box, min = SHIRTCHECK_MIN_FACE) {
  return !!box && box.width >= min && box.height >= min;
}

/**
 * Category counts for pixels inside the convex quad, on a stride grid of at most MAX_SHIRT_SAMPLES.
 * Returns { counts: number[6], sampled }.
 */
export function countCategories(mask, width, height, quad, maxSamples = MAX_SHIRT_SAMPLES) {
  const counts = [0, 0, 0, 0, 0, 0];
  if (!quad || quad.length < 3 || !(width > 0) || !(height > 0)) return { counts, sampled: 0 };

  const box = bboxOfPoints(quad);
  const x0 = Math.max(0, Math.floor(box.x));
  const y0 = Math.max(0, Math.floor(box.y));
  const x1 = Math.min(width, Math.ceil(box.x + box.width));
  const y1 = Math.min(height, Math.ceil(box.y + box.height));
  if (!(x1 > x0) || !(y1 > y0)) return { counts, sampled: 0 };

  const spanX = x1 - x0;
  const spanY = y1 - y0;
  let stride = Math.max(1, Math.ceil(Math.sqrt((spanX * spanY) / maxSamples)));
  while (Math.ceil(spanX / stride) * Math.ceil(spanY / stride) > maxSamples) stride++;

  let sampled = 0;
  for (let y = y0; y < y1; y += stride) {
    for (let x = x0; x < x1; x += stride) {
      if (!insideConvexQuad(quad, x + 0.5, y + 0.5)) continue;
      const value = mask[y * width + x];
      if (value >= 0 && value < counts.length) counts[value]++;
      sampled++;
    }
  }
  return { counts, sampled };
}

/**
 * Turn category counts into a reading.
 * verdict is 'shirtless', 'dressed', 'close' (in between) or 'unclear' (not enough torso).
 * skinRatio is skin / (skin + clothes), or 0 when that is not meaningful.
 */
export function judgeBare(counts, sampled, {
  minSamples = SHIRTCHECK_MIN_SAMPLES,
  shirtlessAt = SHIRTCHECK_SHIRTLESS_AT,
  dressedAt = SHIRTCHECK_DRESSED_AT,
} = {}) {
  const skin = (counts[BODY_SKIN] || 0) + (counts[FACE_SKIN] || 0);
  const clothes = counts[CLOTHES] || 0;
  const relevant = skin + clothes;
  if (sampled < minSamples || relevant < minSamples * 0.5) {
    return { verdict: 'unclear', skinRatio: 0, relevant, sampled };
  }
  const skinRatio = skin / relevant;
  let verdict = 'close';
  if (skinRatio >= shirtlessAt) verdict = 'shirtless';
  else if (skinRatio <= dressedAt) verdict = 'dressed';
  return { verdict, skinRatio, relevant, sampled };
}

/** Count the mask inside the quad and judge it. */
export function judgeMask(mask, width, height, quad) {
  const { counts, sampled } = countCategories(mask, width, height, quad);
  return judgeBare(counts, sampled);
}

/** 'shirtless', 'dressed', or 'unclear' (the in-between 'close' band is not a final call). */
export function verdictFromRatio(ratio, {
  shirtlessAt = SHIRTCHECK_SHIRTLESS_AT,
  dressedAt = SHIRTCHECK_DRESSED_AT,
} = {}) {
  if (ratio == null || Number.isNaN(ratio)) return 'unclear';
  if (ratio >= shirtlessAt) return 'shirtless';
  if (ratio <= dressedAt) return 'dressed';
  return 'unclear';
}
