// Pure geometry helpers: no DOM, no imports. MediaPipe landmarks are normalised (0-1);
// functions that take width/height convert to pixels of the raw, unmirrored frame.

// SFace's reference landmarks in its 112x112 input.
// Order: subject's right eye, left eye, nose tip, right mouth corner, left mouth corner.
// The subject's right is image-left in the raw frame.
export const SFACE_TEMPLATE = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

/** Face-mesh landmarks as 5 pixel points in SFACE_TEMPLATE order. */
export function fivePoints(landmarks, width, height) {
  const at = (i) => [landmarks[i].x * width, landmarks[i].y * height];
  const mid = (i, j) => {
    const [x1, y1] = at(i);
    const [x2, y2] = at(j);
    return [(x1 + x2) / 2, (y1 + y2) / 2];
  };
  return [mid(33, 133), mid(362, 263), at(1), at(61), at(291)];
}

/**
 * Least-squares similarity transform (rotation, uniform scale, translation; no reflection)
 * taking src points onto dst points. Returned in canvas setTransform order:
 * x' = a*x + c*y + e, y' = b*x + d*y + f.
 */
export function estimateSimilarity(src, dst) {
  const count = src.length;
  let sMeanX = 0;
  let sMeanY = 0;
  let dMeanX = 0;
  let dMeanY = 0;
  for (let i = 0; i < count; i++) {
    sMeanX += src[i][0];
    sMeanY += src[i][1];
    dMeanX += dst[i][0];
    dMeanY += dst[i][1];
  }
  sMeanX /= count;
  sMeanY /= count;
  dMeanX /= count;
  dMeanY /= count;

  let norm = 0;
  let dot = 0;
  let cross = 0;
  for (let i = 0; i < count; i++) {
    const xs = src[i][0] - sMeanX;
    const ys = src[i][1] - sMeanY;
    const xd = dst[i][0] - dMeanX;
    const yd = dst[i][1] - dMeanY;
    norm += xs * xs + ys * ys;
    dot += xs * xd + ys * yd;
    cross += xs * yd - ys * xd;
  }
  const A = dot / norm;
  const B = cross / norm;

  return {
    a: A,
    b: B,
    c: -B,
    d: A,
    e: dMeanX - (A * sMeanX - B * sMeanY),
    f: dMeanY - (B * sMeanX + A * sMeanY),
  };
}

/** Head yaw in degrees from the face-mesh cheek landmarks; positive = user turned to their own left. */
export function yawDegrees(landmarks) {
  const left = landmarks[234];
  const right = landmarks[454];
  return (Math.atan2(right.z - left.z, right.x - left.x) * 180) / Math.PI;
}

/** Tightest axis-aligned box around [x, y] points. */
export function bboxOfPoints(points) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

/** Intersection over union of two {x, y, width, height} boxes; 0 when they do not overlap. */
export function iou(boxA, boxB) {
  const overlapW = Math.min(boxA.x + boxA.width, boxB.x + boxB.width) - Math.max(boxA.x, boxB.x);
  const overlapH = Math.min(boxA.y + boxA.height, boxB.y + boxB.height) - Math.max(boxA.y, boxB.y);
  if (overlapW <= 0 || overlapH <= 0) return 0;
  const intersection = overlapW * overlapH;
  const union = boxA.width * boxA.height + boxB.width * boxB.height - intersection;
  return union > 0 ? intersection / union : 0;
}

/** True when [x, y] lies inside the box (edges included). */
export function containsPoint(box, [x, y]) {
  return x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height;
}

/** Median of a list of numbers; averages the middle pair for even lengths. Does not mutate. */
export function median(values) {
  const sorted = [...values].sort((p, q) => p - q);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
