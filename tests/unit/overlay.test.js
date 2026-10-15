import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mirrorBox, drawLabels, UNKNOWN_COLOUR } from '../../src/ui/overlay.js';

test('mirrorBox flips x across the frame width and keeps the rest', () => {
  assert.deepEqual(mirrorBox({ x: 10, y: 5, width: 20, height: 30 }, 100), { x: 70, y: 5, width: 20, height: 30 });
});

/** A 2D-context stand-in that records drawing calls along with the style in force at the time. */
function recordingContext(width, height) {
  const calls = [];
  const ctx = {
    canvas: { width, height },
    strokeStyle: '',
    fillStyle: '',
    clearRect: (...args) => calls.push(['clearRect', ...args]),
    strokeRect: (...args) => calls.push(['strokeRect', ...args, ctx.strokeStyle]),
    fillRect: (...args) => calls.push(['fillRect', ...args, ctx.fillStyle]),
    fillText: (...args) => calls.push(['fillText', ...args, ctx.fillStyle]),
    measureText: (text) => ({ width: text.length * 10 }),
  };
  return { ctx, calls };
}

test('drawLabels clears the canvas, then draws mirrored boxes with the text above them', () => {
  const { ctx, calls } = recordingContext(640, 480);
  const colours = { alice: '#4363d8' };
  drawLabels(
    ctx,
    [
      { trackId: 1, box: { x: 100, y: 200, width: 50, height: 60 }, personId: 'alice', text: 'Alice' },
      { trackId: 2, box: { x: 400, y: 100, width: 80, height: 90 }, personId: null, text: 'Unknown' },
    ],
    { width: 640, colourFor: (id) => colours[id] },
  );

  assert.deepEqual(calls[0], ['clearRect', 0, 0, 640, 480]);
  const strokes = calls.filter(([name]) => name === 'strokeRect');
  assert.deepEqual(strokes, [
    ['strokeRect', 490, 200, 50, 60, '#4363d8'],
    ['strokeRect', 160, 100, 80, 90, UNKNOWN_COLOUR],
  ]);

  const texts = calls.filter(([name]) => name === 'fillText');
  assert.deepEqual(
    texts.map(([, text]) => text),
    ['Alice', 'Unknown'],
  );
  // The text sits above its box.
  assert.ok(texts[0][3] < 200);
  assert.ok(texts[1][3] < 100);
});

test('drawLabels keeps a tag on screen when its box touches the top or the right edge', () => {
  const { ctx, calls } = recordingContext(640, 480);
  drawLabels(ctx, [{ trackId: 1, box: { x: 0, y: 0, width: 40, height: 40 }, personId: null, text: 'Unknown' }], {
    width: 640,
    colourFor: () => undefined,
  });
  const [, x, y, w, h] = calls.find(([name]) => name === 'fillRect');
  assert.ok(y >= 0);
  assert.ok(x >= 0 && x + w <= 640, `tag spans ${x}..${x + w}`);
  assert.ok(h > 0);
});
