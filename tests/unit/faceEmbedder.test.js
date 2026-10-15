import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toInputTensorData, l2normalize } from '../../src/vision/faceEmbedder.js';

const near = (actual, expected, tolerance, message) =>
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${message ?? 'value'}: expected ${expected} +/- ${tolerance}, got ${actual}`,
  );

test('toInputTensorData lays out RGBA pixels as planar RGB (NCHW) with raw 0-255 values', () => {
  const rgba = new Uint8ClampedArray(112 * 112 * 4);
  rgba.set([10, 20, 30, 255], 0);

  const out = toInputTensorData(rgba);

  assert.ok(out instanceof Float32Array);
  assert.equal(out.length, 37632);
  assert.equal(out[0], 10);
  assert.equal(out[12544], 20);
  assert.equal(out[25088], 30);
});

test('toInputTensorData puts each pixel at the same index in every plane and drops alpha', () => {
  const rgba = new Uint8ClampedArray(112 * 112 * 4);
  const pixel = 112 * 5 + 7; // row 5, column 7
  rgba.set([200, 100, 50, 0], pixel * 4);

  const out = toInputTensorData(rgba);

  assert.equal(out[pixel], 200);
  assert.equal(out[12544 + pixel], 100);
  assert.equal(out[25088 + pixel], 50);
  assert.equal(out[pixel + 1], 0);
});

test('l2normalize scales a vector to unit length', () => {
  const out = l2normalize([3, 4]);

  assert.ok(out instanceof Float32Array);
  assert.equal(out.length, 2);
  near(out[0], 0.6, 1e-6, 'x');
  near(out[1], 0.8, 1e-6, 'y');
});

test('l2normalize returns zeros (not NaN) for an all-zero vector', () => {
  assert.deepEqual(Array.from(l2normalize([0, 0, 0])), [0, 0, 0]);
});
