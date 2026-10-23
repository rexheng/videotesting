import { test } from 'node:test';
import assert from 'node:assert/strict';
import { countCategories, judgeBare, judgeMask, scaleQuad, isFacePresent, verdictFromRatio } from '../../src/shirtcheck/bare.js';

// A width*height mask filled with one category, then a full-frame quad.
function solid(width, height, category) {
  const mask = new Uint8Array(width * height);
  mask.fill(category);
  const quad = [
    [0, 0],
    [width, 0],
    [width, height],
    [0, height],
  ];
  return { mask, quad };
}

test('a torso of clothes reads as dressed, and bare skin as shirtless', () => {
  const clothed = solid(40, 40, 4);
  const dressed = judgeMask(clothed.mask, 40, 40, clothed.quad);
  assert.equal(dressed.verdict, 'dressed');
  assert.equal(dressed.skinRatio, 0);

  const bare = solid(40, 40, 2);
  const shirtless = judgeMask(bare.mask, 40, 40, bare.quad);
  assert.equal(shirtless.verdict, 'shirtless');
  assert.equal(shirtless.skinRatio, 1);
});

test('neck pixels labelled face-skin still count as skin, hair does not', () => {
  const { mask, quad } = solid(20, 20, 3);
  assert.equal(judgeMask(mask, 20, 20, quad).verdict, 'shirtless');

  const hair = solid(20, 20, 1);
  assert.equal(judgeMask(hair.mask, 20, 20, hair.quad).verdict, 'unclear');
});

test('half skin and half cloth is too close to call', () => {
  const width = 40;
  const height = 40;
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < mask.length; i++) mask[i] = i < mask.length / 2 ? 2 : 4;
  const quad = [
    [0, 0],
    [width, 0],
    [width, height],
    [0, height],
  ];
  const reading = judgeMask(mask, width, height, quad);
  assert.equal(reading.verdict, 'close');
  assert.ok(Math.abs(reading.skinRatio - 0.5) < 0.02);
  assert.equal(verdictFromRatio(reading.skinRatio), 'unclear');
});

test('pixels outside the quad are ignored', () => {
  const width = 40;
  const height = 20;
  const mask = new Uint8Array(width * height);
  mask.fill(4); // clothes everywhere
  // Only the left half is the torso, and it is skin.
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < 20; x++) mask[y * width + x] = 2;
  }
  const quad = [
    [0, 0],
    [20, 0],
    [20, height],
    [0, height],
  ];
  const reading = judgeMask(mask, width, height, quad);
  assert.equal(reading.verdict, 'shirtless');
  assert.equal(reading.skinRatio, 1);
});

test('too small a region is unclear', () => {
  const reading = judgeBare([0, 0, 3, 0, 3, 0], 6);
  assert.equal(reading.verdict, 'unclear');
  assert.equal(reading.skinRatio, 0);
});

test('scaleQuad stretches a frame quad onto a smaller mask', () => {
  const scaled = scaleQuad(
    [
      [0, 0],
      [100, 50],
    ],
    200,
    100,
    20,
    10,
  );
  assert.deepEqual(scaled, [
    [0, 0],
    [10, 5],
  ]);
});

test('a face has to be large enough, and countCategories skips an empty quad', () => {
  assert.equal(isFacePresent({ x: 0, y: 0, width: 0.2, height: 0.2 }), true);
  assert.equal(isFacePresent({ x: 0, y: 0, width: 0.02, height: 0.2 }), false);
  assert.equal(isFacePresent(null), false);
  const empty = countCategories(new Uint8Array(0), 0, 0, null);
  assert.equal(empty.sampled, 0);
});
