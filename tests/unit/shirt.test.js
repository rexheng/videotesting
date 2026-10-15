import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  torsoQuad,
  rgbToHsv,
  shirtHistogram,
  histogramSimilarity,
  averageHistograms,
  blendHistograms,
} from '../../src/recognition/shirt.js';
import { MAX_SHIRT_SAMPLES } from '../../src/config.js';

const near = (actual, expected, tolerance, message) =>
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${message ?? 'value'}: expected ${expected} +/- ${tolerance}, got ${actual}`,
  );

/** RGBA buffer where pixel (x, y) is colourAt(x, y) = [r, g, b]. */
function makeImage(width, height, colourAt) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b] = colourAt(x, y);
      const i = (y * width + x) * 4;
      rgba[i] = r;
      rgba[i + 1] = g;
      rgba[i + 2] = b;
      rgba[i + 3] = 255;
    }
  }
  return rgba;
}

const solidImage = (width, height, colour) => makeImage(width, height, () => colour);

const fullQuad = (width, height) => [
  [0, 0],
  [width, 0],
  [width, height],
  [0, height],
];

/** A 33-landmark pose, everything invisible at the centre unless overridden by index. */
function blankPose(overrides = {}) {
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0 }));
  for (const [index, landmark] of Object.entries(overrides)) {
    pose[index] = { ...pose[index], ...landmark };
  }
  return pose;
}

// ---------------------------------------------------------------------------------------------
// rgbToHsv

test('rgbToHsv converts the primaries and greys', () => {
  assert.deepEqual(rgbToHsv(255, 0, 0), [0, 1, 1]);
  assert.deepEqual(rgbToHsv(0, 255, 0), [120, 1, 1]);
  assert.deepEqual(rgbToHsv(0, 0, 255), [240, 1, 1]);
  assert.deepEqual(rgbToHsv(0, 0, 0), [0, 0, 0]);
  assert.deepEqual(rgbToHsv(255, 255, 255), [0, 0, 1]);
  const [h, s, v] = rgbToHsv(128, 128, 128);
  assert.equal(h, 0, 'achromatic hue is 0');
  assert.equal(s, 0);
  near(v, 128 / 255, 1e-12, 'grey value');
});

test('rgbToHsv returns hue in [0, 360), never 360 (ruling R3)', () => {
  const [hNear] = rgbToHsv(255, 0, 1);
  assert.ok(hNear > 359 && hNear < 360, `expected just under 360, got ${hNear}`);
  // A vanishingly small negative raw hue rounds up to exactly 360 if not guarded.
  const [hTiny] = rgbToHsv(255, 0, 1e-14);
  assert.ok(hTiny >= 0 && hTiny < 360, `hue must stay below 360, got ${hTiny}`);
});

// ---------------------------------------------------------------------------------------------
// shirtHistogram

test('solid red lands entirely in bin 5 with a #ff0000 swatch', () => {
  const result = shirtHistogram(solidImage(8, 8, [255, 0, 0]), 8, 8, fullQuad(8, 8));
  assert.ok(result.histogram instanceof Float32Array);
  assert.equal(result.histogram.length, 28);
  assert.equal(result.histogram[5], 1);
  assert.equal(result.swatch, '#ff0000');
});

test('solid white lands in grey bin 3, solid black in grey bin 0', () => {
  const white = shirtHistogram(solidImage(8, 8, [255, 255, 255]), 8, 8, fullQuad(8, 8));
  assert.equal(white.histogram[3], 1);
  assert.equal(white.swatch, '#ffffff');
  const black = shirtHistogram(solidImage(8, 8, [0, 0, 0]), 8, 8, fullQuad(8, 8));
  assert.equal(black.histogram[0], 1);
  assert.equal(black.swatch, '#000000');
});

test('grey bins are split by value; low saturation counts as grey', () => {
  const binOf = (colour) => {
    const { histogram } = shirtHistogram(solidImage(2, 2, colour), 2, 2, fullQuad(2, 2));
    return histogram.findIndex((value) => value === 1);
  };
  assert.equal(binOf([50, 50, 50]), 0); // v ~0.20
  assert.equal(binOf([100, 100, 100]), 1); // v ~0.39
  assert.equal(binOf([150, 150, 150]), 2); // v ~0.59
  assert.equal(binOf([200, 200, 200]), 3); // v ~0.78
  assert.equal(binOf([30, 0, 0]), 0); // dark red: v < 0.15 is grey
  assert.equal(binOf([200, 170, 170]), 3); // pale pink: s ~0.15 is grey
});

test('coloured pixels use bin 4 + floor(h/30)*2 + (s < 0.5 ? 0 : 1)', () => {
  const binOf = (colour) => {
    const { histogram } = shirtHistogram(solidImage(2, 2, colour), 2, 2, fullQuad(2, 2));
    return histogram.findIndex((value) => value === 1);
  };
  assert.equal(binOf([0, 255, 0]), 4 + 4 * 2 + 1); // green, saturated
  assert.equal(binOf([0, 0, 255]), 4 + 8 * 2 + 1); // blue, saturated
  assert.equal(binOf([200, 120, 120]), 4 + 0 * 2 + 0); // muted red, s ~0.4
  assert.equal(binOf([255, 0, 1]), 4 + 11 * 2 + 1); // hue just under 360 stays in the last bin
});

test('histogram is normalised to sum 1', () => {
  const rgba = makeImage(6, 6, (x, y) => ((x + y) % 3 === 0 ? [255, 0, 0] : (x + y) % 3 === 1 ? [0, 255, 0] : [20, 20, 20]));
  const { histogram } = shirtHistogram(rgba, 6, 6, fullQuad(6, 6));
  near(histogram.reduce((sum, value) => sum + value, 0), 1, 1e-6, 'sum');
});

test('swatch is the mean colour of the fullest bin', () => {
  const colours = [[250, 0, 0], [254, 0, 0], [252, 0, 0], [0, 0, 255]];
  const rgba = makeImage(4, 1, (x) => colours[x]);
  const result = shirtHistogram(rgba, 4, 1, fullQuad(4, 1));
  near(result.histogram[5], 0.75, 1e-6, 'red share');
  near(result.histogram[21], 0.25, 1e-6, 'blue share');
  assert.equal(result.swatch, '#fc0000');
});

test('only pixels inside the quad are sampled', () => {
  const rgba = makeImage(10, 10, (x) => (x < 5 ? [255, 0, 0] : [0, 0, 255]));
  const leftHalf = [[0, 0], [5, 0], [5, 10], [0, 10]];
  assert.equal(shirtHistogram(rgba, 10, 10, leftHalf).histogram[5], 1);
});

test('a rotated quad of either winding samples exactly its interior', () => {
  const inside = (x, y) => Math.abs(x + 0.5 - 50) + Math.abs(y + 0.5 - 50) <= 50;
  const rgba = makeImage(100, 100, (x, y) => (inside(x, y) ? [255, 0, 0] : [0, 0, 255]));
  const diamond = [[50, 0], [100, 50], [50, 100], [0, 50]];
  assert.equal(shirtHistogram(rgba, 100, 100, diamond).histogram[5], 1);
  assert.equal(shirtHistogram(rgba, 100, 100, [...diamond].reverse()).histogram[5], 1);
});

test('a quad hanging off the frame is clipped to it', () => {
  const quad = [[-50, -50], [20, -50], [20, 20], [-50, 20]];
  const result = shirtHistogram(solidImage(10, 10, [255, 0, 0]), 10, 10, quad);
  assert.equal(result.histogram[5], 1);
});

test('shirtHistogram returns null when no pixels are sampled', () => {
  const rgba = solidImage(10, 10, [255, 0, 0]);
  assert.equal(shirtHistogram(rgba, 10, 10, [[200, 200], [300, 200], [300, 300], [200, 300]]), null);
  assert.equal(shirtHistogram(rgba, 10, 10, [[1, 1], [1, 1], [1, 1], [1, 1]]), null);
});

test('shirtHistogram never reads more than MAX_SHIRT_SAMPLES pixels', () => {
  const width = 400;
  const height = 300;
  const data = solidImage(width, height, [255, 0, 0]);
  let pixelReads = 0;
  const counting = new Proxy(data, {
    get(target, key) {
      if (typeof key === 'string' && /^\d+$/.test(key) && Number(key) % 4 === 0) pixelReads++;
      return target[key];
    },
  });
  const result = shirtHistogram(counting, width, height, fullQuad(width, height));
  assert.equal(result.histogram[5], 1);
  assert.ok(pixelReads > 0 && pixelReads <= MAX_SHIRT_SAMPLES, `read ${pixelReads} pixels`);
});

// ---------------------------------------------------------------------------------------------
// histogramSimilarity / averageHistograms / blendHistograms

test('histogramSimilarity is the Bhattacharyya coefficient', () => {
  const histogramOf = (colour) => shirtHistogram(solidImage(8, 8, colour), 8, 8, fullQuad(8, 8)).histogram;
  const red = histogramOf([255, 0, 0]);
  const blue = histogramOf([0, 0, 255]);
  assert.ok(histogramSimilarity(red, blue) < 0.1);
  near(histogramSimilarity(red, red), 1, 1e-9, 'red vs red');

  const halfAndHalf = makeImage(8, 8, (x) => (x < 4 ? [255, 0, 0] : [0, 0, 255]));
  const mixed = shirtHistogram(halfAndHalf, 8, 8, fullQuad(8, 8)).histogram;
  near(histogramSimilarity(mixed, red), Math.SQRT1_2, 0.02, 'half red vs red');
});

test('averageHistograms averages and normalises', () => {
  const average = averageHistograms([Float32Array.of(1, 0, 0), Float32Array.of(0, 1, 0), Float32Array.of(0, 1, 0)]);
  assert.ok(average instanceof Float32Array);
  near(average[0], 1 / 3, 1e-6);
  near(average[1], 2 / 3, 1e-6);
  near(average[2], 0, 1e-6);
  near(average.reduce((sum, value) => sum + value, 0), 1, 1e-6, 'sum');
});

test('blendHistograms mixes old and fresh by weight', () => {
  const blended = blendHistograms([1, 0], [0, 1], 0.3);
  assert.ok(blended instanceof Float32Array);
  near(blended[0], 0.7, 1e-6);
  near(blended[1], 0.3, 1e-6);
  const kept = blendHistograms(Float32Array.of(0.5, 0.5), Float32Array.of(0.5, 0.5), 0.3);
  near(kept[0], 0.5, 1e-6);
  near(kept[1], 0.5, 1e-6);
});

// ---------------------------------------------------------------------------------------------
// torsoQuad

test('torsoQuad returns null when a shoulder is not visible enough', () => {
  const pose = blankPose({
    11: { x: 0.1, y: 0.1, visibility: 0.3 },
    12: { x: 0.2, y: 0.1, visibility: 0.9 },
    23: { x: 0.1, y: 0.4, visibility: 0.9 },
    24: { x: 0.2, y: 0.4, visibility: 0.9 },
  });
  assert.equal(torsoQuad(pose, 1000, 1000), null);
  const otherSide = blankPose({
    11: { x: 0.1, y: 0.1, visibility: 0.9 },
    12: { x: 0.2, y: 0.1, visibility: 0.3 },
    23: { x: 0.1, y: 0.4, visibility: 0.9 },
    24: { x: 0.2, y: 0.4, visibility: 0.9 },
  });
  assert.equal(torsoQuad(otherSide, 1000, 1000), null);
});

test('torsoQuad estimates invisible hips below the shoulders, then shrinks 15% toward the centroid', () => {
  const pose = blankPose({
    11: { x: 0.1, y: 0.1, visibility: 0.9 },
    12: { x: 0.2, y: 0.1, visibility: 0.9 },
  });
  const quad = torsoQuad(pose, 1000, 1000);
  assert.equal(quad.length, 4);
  // Unshrunk: (100,100) (200,100) (200,200) (100,200); centroid (150,150).
  const expected = [[107.5, 107.5], [192.5, 107.5], [192.5, 192.5], [107.5, 192.5]];
  quad.forEach(([x, y], i) => {
    near(x, expected[i][0], 1e-9, `corner ${i} x`);
    near(y, expected[i][1], 1e-9, `corner ${i} y`);
  });
});

test('torsoQuad estimates both hips when only one is hidden', () => {
  const pose = blankPose({
    11: { x: 0.1, y: 0.1, visibility: 0.9 },
    12: { x: 0.2, y: 0.1, visibility: 0.9 },
    23: { x: 0.1, y: 0.5, visibility: 0.9 },
    24: { x: 0.2, y: 0.5, visibility: 0.2 },
  });
  const quad = torsoQuad(pose, 1000, 1000);
  near(quad[2][1], 192.5, 1e-9, 'hip 24 estimated, not the reported y=500');
  near(quad[3][1], 192.5, 1e-9, 'hip 23 estimated too');
});

test('torsoQuad uses visible hips and orders corners [11, 12, 24, 23]', () => {
  const pose = blankPose({
    11: { x: 0.1, y: 0.1, visibility: 0.9 },
    12: { x: 0.3, y: 0.1, visibility: 0.9 },
    23: { x: 0.12, y: 0.5, visibility: 0.9 },
    24: { x: 0.28, y: 0.5, visibility: 0.9 },
  });
  const quad = torsoQuad(pose, 1000, 1000);
  const raw = [[100, 100], [300, 100], [280, 500], [120, 500]];
  const cx = raw.reduce((sum, [x]) => sum + x, 0) / 4;
  const cy = raw.reduce((sum, [, y]) => sum + y, 0) / 4;
  raw.forEach(([x, y], i) => {
    near(quad[i][0], cx + (x - cx) * 0.85, 1e-9, `corner ${i} x`);
    near(quad[i][1], cy + (y - cy) * 0.85, 1e-9, `corner ${i} y`);
  });
});

test('torsoQuad converts normalised landmarks using width and height separately', () => {
  const pose = blankPose({
    11: { x: 0.25, y: 0.25, visibility: 0.9 },
    12: { x: 0.75, y: 0.25, visibility: 0.9 },
    23: { x: 0.25, y: 0.75, visibility: 0.9 },
    24: { x: 0.75, y: 0.75, visibility: 0.9 },
  });
  const quad = torsoQuad(pose, 800, 400);
  // Unshrunk box is x 200..600, y 100..300; shrunk by 15% about (400, 200).
  near(quad[0][0], 400 - 200 * 0.85, 1e-9);
  near(quad[0][1], 200 - 100 * 0.85, 1e-9);
  near(quad[2][0], 400 + 200 * 0.85, 1e-9);
  near(quad[2][1], 200 + 100 * 0.85, 1e-9);
});

test('torsoQuad returns null when the torso is too small', () => {
  const tiny = blankPose({
    11: { x: 0.1, y: 0.1, visibility: 0.9 },
    12: { x: 0.12, y: 0.1, visibility: 0.9 },
  });
  assert.equal(torsoQuad(tiny, 1000, 1000), null);

  // Wide enough but too short: visible hips sit right under the shoulders.
  const flat = blankPose({
    11: { x: 0.1, y: 0.1, visibility: 0.9 },
    12: { x: 0.5, y: 0.1, visibility: 0.9 },
    23: { x: 0.1, y: 0.11, visibility: 0.9 },
    24: { x: 0.5, y: 0.11, visibility: 0.9 },
  });
  assert.equal(torsoQuad(flat, 1000, 1000), null);
});
