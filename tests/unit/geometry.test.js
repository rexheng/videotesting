import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SFACE_TEMPLATE,
  fivePoints,
  estimateSimilarity,
  yawDegrees,
  bboxOfPoints,
  iou,
  containsPoint,
  median,
} from '../../src/vision/geometry.js';

const near = (actual, expected, tolerance, message) =>
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${message ?? 'value'}: expected ${expected} +/- ${tolerance}, got ${actual}`,
  );

/** A full 478-landmark face mesh with every point at the centre, ready to be overridden. */
function blankLandmarks() {
  return Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
}

test('estimateSimilarity maps a scaled, rotated, translated template back onto the template', () => {
  const angle = (30 * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const src = SFACE_TEMPLATE.map(([x, y]) => [
    2 * (cos * x - sin * y) + 100,
    2 * (sin * x + cos * y) + 50,
  ]);

  const { a, b, c, d, e, f } = estimateSimilarity(src, SFACE_TEMPLATE);

  src.forEach(([x, y], i) => {
    const mappedX = a * x + c * y + e;
    const mappedY = b * x + d * y + f;
    near(mappedX, SFACE_TEMPLATE[i][0], 1e-6, `x of point ${i}`);
    near(mappedY, SFACE_TEMPLATE[i][1], 1e-6, `y of point ${i}`);
  });
});

test('estimateSimilarity returns the identity when source and destination match', () => {
  const m = estimateSimilarity(SFACE_TEMPLATE, SFACE_TEMPLATE);
  near(m.a, 1, 1e-9, 'a');
  near(m.b, 0, 1e-9, 'b');
  near(m.c, 0, 1e-9, 'c');
  near(m.d, 1, 1e-9, 'd');
  near(m.e, 0, 1e-9, 'e');
  near(m.f, 0, 1e-9, 'f');
});

test('yawDegrees is 0 for a face looking straight at the camera', () => {
  const landmarks = blankLandmarks();
  landmarks[234] = { x: 0.4, y: 0.5, z: 0 };
  landmarks[454] = { x: 0.6, y: 0.5, z: 0 };
  near(yawDegrees(landmarks), 0, 1e-9);
});

test('yawDegrees is positive when the user turns to their own left and negative to their right', () => {
  const faceAt = (degrees) => {
    const theta = (degrees * Math.PI) / 180;
    const landmarks = blankLandmarks();
    landmarks[234] = { x: 0.5 - 0.1 * Math.cos(theta), y: 0.5, z: -0.1 * Math.sin(theta) };
    landmarks[454] = { x: 0.5 + 0.1 * Math.cos(theta), y: 0.5, z: 0.1 * Math.sin(theta) };
    return landmarks;
  };
  near(yawDegrees(faceAt(30)), 30, 0.01, 'turned left');
  near(yawDegrees(faceAt(-30)), -30, 0.01, 'turned right');
});

test('fivePoints returns pixel points in template order', () => {
  const landmarks = blankLandmarks();
  landmarks[33] = { x: 0.2, y: 0.4, z: 0 };
  landmarks[133] = { x: 0.3, y: 0.4, z: 0 };
  landmarks[362] = { x: 0.7, y: 0.4, z: 0 };
  landmarks[263] = { x: 0.8, y: 0.6, z: 0 };
  landmarks[1] = { x: 0.5, y: 0.55, z: 0 };
  landmarks[61] = { x: 0.35, y: 0.75, z: 0 };
  landmarks[291] = { x: 0.65, y: 0.8, z: 0 };

  const points = fivePoints(landmarks, 1000, 500);

  assert.equal(points.length, 5);
  assert.deepEqual(points[0], [250, 200]);
  assert.deepEqual(points[1], [750, 250]);
  assert.deepEqual(points[2], [500, 275]);
  assert.deepEqual(points[3], [350, 375]);
  assert.deepEqual(points[4], [650, 400]);
});

test('iou is 1 for identical boxes, 0 for disjoint boxes and 1/3 for half-overlapping boxes', () => {
  const box = { x: 0, y: 0, width: 10, height: 10 };
  assert.equal(iou(box, { ...box }), 1);
  assert.equal(iou(box, { x: 50, y: 50, width: 10, height: 10 }), 0);
  near(iou(box, { x: 5, y: 0, width: 10, height: 10 }), 1 / 3, 1e-9);
});

test('containsPoint is true inside a box and false outside it', () => {
  const box = { x: 10, y: 20, width: 30, height: 40 };
  assert.equal(containsPoint(box, [25, 40]), true);
  assert.equal(containsPoint(box, [5, 40]), false);
  assert.equal(containsPoint(box, [25, 70]), false);
});

test('bboxOfPoints wraps the points in the tightest axis-aligned box', () => {
  assert.deepEqual(
    bboxOfPoints([
      [10, 20],
      [40, 5],
      [25, 60],
    ]),
    { x: 10, y: 5, width: 30, height: 55 },
  );
});

test('median handles odd and even length lists without mutating the input', () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([1, 2, 3, 4]), 2.5);
  const values = [3, 1, 2];
  median(values);
  assert.deepEqual(values, [3, 1, 2]);
});
