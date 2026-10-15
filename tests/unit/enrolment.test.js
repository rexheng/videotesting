import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  createEnrolment,
  enrolmentWantsEmbedding,
  stepEnrolment,
} from '../../src/enrol/enrolment.js';
import {
  SAMPLES_PER_STEP,
  SAMPLE_GAP_MS,
  NO_FACE_TIP_MS,
  STUCK_TIP_MS,
} from '../../src/config.js';

const FRAME_MS = 200;

/** A fresh 128-number embedding whose first value identifies it. */
function embedding(tag = 0) {
  const vector = new Float32Array(128);
  vector[0] = tag;
  return vector;
}

/** A shirt reading: a normalised 28-bin histogram with all the weight on `bin`. */
function shirtReading(bin = 0, swatch = '#112233') {
  const histogram = new Float32Array(28);
  histogram[bin] = 1;
  return { histogram, swatch };
}

const faceFrame = (yaw, tag = 0) => ({ face: { yaw, embedding: embedding(tag) }, shirt: null });
const shirtFrame = (bin = 0, swatch = '#112233') => ({ face: null, shirt: shirtReading(bin, swatch) });
const emptyFrame = { face: null, shirt: null };

/** Feeds `count` identical frames, FRAME_MS apart from `start`. Returns the state and the next time. */
function feed(state, frame, count, start) {
  let now = start;
  let current = state;
  for (let i = 0; i < count; i++) {
    current = stepEnrolment(current, frame, now);
    now += FRAME_MS;
  }
  return { state: current, now };
}

/**
 * Walks a full-mode enrolment as far as the start of the given step. `now` is the time of the next
 * frame; `begun` is when the step started (the moment its predecessor took its last sample).
 */
function advanceTo(step, start = 0) {
  let { state, now } = feed(createEnrolment('Alice', start), faceFrame(5), SAMPLES_PER_STEP, start);
  if (step !== 'left') {
    ({ state, now } = feed(state, faceFrame(31), SAMPLES_PER_STEP, now));
  }
  if (step === 'shirt') {
    ({ state, now } = feed(state, faceFrame(-21), SAMPLES_PER_STEP, now));
  }
  return { state, now, begun: now - FRAME_MS };
}

test('createEnrolment starts a full enrolment on the straight step', () => {
  const state = createEnrolment('Alice', 0);
  assert.equal(state.name, 'Alice');
  assert.equal(state.mode, 'full');
  assert.equal(state.step, 'straight');
  assert.equal(state.progress, 0);
  assert.equal(state.prompt, 'Look straight at the camera');
  assert.equal(state.tip, null);
  assert.equal(state.arrow, null);
  assert.equal(state.yawOffset, 0);
  assert.equal(state.result, null);
});

test('full walk-through: straight, left, right, shirt, done', () => {
  let state = createEnrolment('Alice', 0);
  let now = 0;

  ({ state, now } = feed(state, faceFrame(5), SAMPLES_PER_STEP, now));
  assert.equal(state.step, 'left');
  assert.equal(state.yawOffset, 5);
  assert.equal(state.prompt, 'Slowly turn your head LEFT');

  // 28 - 5 = 23 is short of the 25 degree turn, so these add nothing.
  ({ state, now } = feed(state, faceFrame(28), SAMPLES_PER_STEP, now));
  assert.equal(state.step, 'left');
  assert.equal(state.progress, 0);

  ({ state, now } = feed(state, faceFrame(31), SAMPLES_PER_STEP, now));
  assert.equal(state.step, 'right');
  assert.equal(state.prompt, 'Now turn RIGHT');

  // -21 - 5 = -26, past the -25 threshold.
  ({ state, now } = feed(state, faceFrame(-21), SAMPLES_PER_STEP, now));
  assert.equal(state.step, 'shirt');
  assert.equal(state.prompt, 'Step back so we can see your shirt');
  assert.equal(state.result, null);

  ({ state, now } = feed(state, shirtFrame(), SAMPLES_PER_STEP, now));
  assert.equal(state.step, 'done');
  assert.equal(state.prompt, 'Done! Hi Alice 👋');
  assert.equal(state.progress, 1);
  assert.equal(state.tip, null);
  assert.equal(state.arrow, null);
  assert.equal(state.result.faceEmbeddings.length, 15);
  assert.equal(state.result.shirtHistogram.length, 28);
  assert.equal(state.result.shirtSwatch, '#112233');
});

test('straight uses the median of its yaws as the offset', () => {
  let state = createEnrolment('Alice', 0);
  let now = 0;
  for (const yaw of [2, -4, 8, 6, 0]) {
    state = stepEnrolment(state, faceFrame(yaw), now);
    now += FRAME_MS;
  }
  assert.equal(state.step, 'left');
  assert.equal(state.yawOffset, 2);
});

test('straight rejects a yaw outside the straight range', () => {
  const { state } = feed(createEnrolment('Alice', 0), faceFrame(13), SAMPLES_PER_STEP, 0);
  assert.equal(state.step, 'straight');
  assert.equal(state.progress, 0);
});

test('left and right measure the turn from the straight offset', () => {
  const { state: atLeft } = advanceTo('left');
  // Raw yaw -26 would count as a right turn from zero, but from offset 5 it is nowhere near left.
  assert.equal(enrolmentWantsEmbedding(atLeft, 30, 10_000), true);
  assert.equal(enrolmentWantsEmbedding(atLeft, 29, 10_000), false);
  assert.equal(enrolmentWantsEmbedding(atLeft, -26, 10_000), false);

  const { state: atRight } = advanceTo('right');
  assert.equal(enrolmentWantsEmbedding(atRight, -20, 10_000), true);
  assert.equal(enrolmentWantsEmbedding(atRight, -19, 10_000), false);
  assert.equal(enrolmentWantsEmbedding(atRight, 31, 10_000), false);
});

test('samples closer together than the gap are not accepted', () => {
  let state = createEnrolment('Alice', 0);
  state = stepEnrolment(state, faceFrame(5), 0);
  state = stepEnrolment(state, faceFrame(5), 100);
  assert.equal(state.progress, 1 / SAMPLES_PER_STEP);
  state = stepEnrolment(state, faceFrame(5), SAMPLE_GAP_MS);
  assert.equal(state.progress, 2 / SAMPLES_PER_STEP);
});

test('a null embedding is not accepted, even with the yaw in range', () => {
  const state = stepEnrolment(
    createEnrolment('Alice', 0),
    { face: { yaw: 5, embedding: null }, shirt: null },
    1000,
  );
  assert.equal(state.progress, 0);
  assert.equal(state.step, 'straight');
});

test('progress is the samples taken over the samples needed', () => {
  let state = createEnrolment('Alice', 0);
  state = stepEnrolment(state, faceFrame(5), 0);
  state = stepEnrolment(state, faceFrame(5), FRAME_MS);
  assert.equal(state.progress, 2 / SAMPLES_PER_STEP);
});

test('progress restarts at zero on the next step', () => {
  const { state } = advanceTo('left');
  assert.equal(state.step, 'left');
  assert.equal(state.progress, 0);
});

test('the no-face tip appears after NO_FACE_TIP_MS without a face', () => {
  let state = createEnrolment('Alice', 0);
  state = stepEnrolment(state, emptyFrame, NO_FACE_TIP_MS - 1);
  assert.equal(state.tip, null);
  state = stepEnrolment(state, emptyFrame, NO_FACE_TIP_MS);
  assert.equal(state.tip, 'Move closer, or try more light');
});

test('the no-face timer restarts when a face was last seen', () => {
  let state = createEnrolment('Alice', 0);
  // A face with a null embedding is seen but gives no sample.
  state = stepEnrolment(state, { face: { yaw: 5, embedding: null }, shirt: null }, 2500);
  state = stepEnrolment(state, emptyFrame, 2500 + NO_FACE_TIP_MS - 1);
  assert.equal(state.tip, null);
  state = stepEnrolment(state, emptyFrame, 2500 + NO_FACE_TIP_MS);
  assert.equal(state.tip, 'Move closer, or try more light');
});

test('the stuck tip on left says to turn further', () => {
  let { state, begun } = advanceTo('left');
  // A face is there but turned too little to count.
  state = stepEnrolment(state, faceFrame(10), begun + STUCK_TIP_MS - 1);
  assert.equal(state.tip, null);
  state = stepEnrolment(state, faceFrame(10), begun + STUCK_TIP_MS);
  assert.equal(state.step, 'left');
  assert.equal(state.tip, 'Turn a bit further');
});

test('the stuck tip on right says to turn further', () => {
  let { state, begun } = advanceTo('right');
  state = stepEnrolment(state, faceFrame(10), begun + STUCK_TIP_MS);
  assert.equal(state.step, 'right');
  assert.equal(state.tip, 'Turn a bit further');
});

test('the stuck tip on straight says to face the camera', () => {
  let state = createEnrolment('Alice', 0);
  state = stepEnrolment(state, faceFrame(40), STUCK_TIP_MS);
  assert.equal(state.tip, 'Face the camera and hold still');
});

test('the no-face tip takes priority over the stuck tip when there is no face', () => {
  let state = createEnrolment('Alice', 0);
  state = stepEnrolment(state, emptyFrame, STUCK_TIP_MS);
  assert.equal(state.tip, 'Move closer, or try more light');
});

test('taking a sample clears the tip and restarts the stuck timer', () => {
  let state = createEnrolment('Alice', 0);
  state = stepEnrolment(state, faceFrame(40), STUCK_TIP_MS);
  assert.equal(state.tip, 'Face the camera and hold still');
  state = stepEnrolment(state, faceFrame(5), STUCK_TIP_MS + FRAME_MS);
  assert.equal(state.tip, null);
  state = stepEnrolment(state, faceFrame(40), STUCK_TIP_MS + FRAME_MS + STUCK_TIP_MS - 1);
  assert.equal(state.tip, null);
  state = stepEnrolment(state, faceFrame(40), STUCK_TIP_MS + FRAME_MS + STUCK_TIP_MS);
  assert.equal(state.tip, 'Face the camera and hold still');
});

test('the arrow points left on the left step and right on the right step', () => {
  assert.equal(createEnrolment('Alice', 0).arrow, null);
  assert.equal(advanceTo('left').state.arrow, 'left');
  assert.equal(advanceTo('right').state.arrow, 'right');
  assert.equal(advanceTo('shirt').state.arrow, null);
});

test('the shirt step tips when no shirt is seen for NO_FACE_TIP_MS', () => {
  let { state, begun } = advanceTo('shirt');
  assert.equal(state.step, 'shirt');
  state = stepEnrolment(state, emptyFrame, begun + NO_FACE_TIP_MS - 1);
  assert.equal(state.tip, null);
  state = stepEnrolment(state, emptyFrame, begun + NO_FACE_TIP_MS);
  assert.equal(state.tip, 'Step back so your shoulders are in view');
  state = stepEnrolment(state, shirtFrame(), begun + NO_FACE_TIP_MS + FRAME_MS);
  assert.equal(state.tip, null);
});

test('the shirt step averages its histograms and keeps the last swatch', () => {
  let { state, now } = advanceTo('shirt');
  for (let i = 0; i < SAMPLES_PER_STEP; i++) {
    // Alternate two bins so the average is split between them.
    state = stepEnrolment(state, shirtFrame(i % 2, `#00000${i}`), now);
    now += FRAME_MS;
  }
  assert.equal(state.step, 'done');
  assert.equal(state.result.shirtSwatch, '#000004');
  const histogram = state.result.shirtHistogram;
  assert.ok(Math.abs(histogram[0] - 0.6) < 1e-6);
  assert.ok(Math.abs(histogram[1] - 0.4) < 1e-6);
});

test('the shirt step is gap-limited', () => {
  let { state, now } = advanceTo('shirt');
  state = stepEnrolment(state, shirtFrame(), now);
  state = stepEnrolment(state, shirtFrame(), now + 100);
  assert.equal(state.progress, 1 / SAMPLES_PER_STEP);
});

test('result keeps the embeddings as given, in step order', () => {
  let state = createEnrolment('Alice', 0);
  let now = 0;
  const given = [];
  const frames = [
    [5, 0],
    [31, 1],
    [-21, 2],
  ];
  for (const [yaw, step] of frames) {
    for (let i = 0; i < SAMPLES_PER_STEP; i++) {
      const frame = faceFrame(yaw, step * 10 + i);
      given.push(frame.face.embedding);
      state = stepEnrolment(state, frame, now);
      now += FRAME_MS;
    }
  }
  ({ state } = feed(state, shirtFrame(), SAMPLES_PER_STEP, now));
  assert.equal(state.step, 'done');
  state.result.faceEmbeddings.forEach((vector, i) => assert.equal(vector, given[i]));
});

test('shirt mode: five shirt frames finish with no face embeddings', () => {
  const start = createEnrolment('Alice', 0, { startAt: 'shirt' });
  assert.equal(start.mode, 'shirt');
  assert.equal(start.step, 'shirt');
  assert.equal(start.yawOffset, 0);
  assert.equal(start.prompt, 'Step back so we can see your shirt');

  const { state } = feed(start, shirtFrame(), SAMPLES_PER_STEP, 0);
  assert.equal(state.step, 'done');
  assert.equal(state.prompt, 'Shirt updated for Alice 👕');
  assert.deepEqual(state.result.faceEmbeddings, []);
  assert.equal(state.result.shirtHistogram.length, 28);
});

test('shirt mode ignores face frames', () => {
  const start = createEnrolment('Alice', 0, { startAt: 'shirt' });
  const { state } = feed(start, faceFrame(5), SAMPLES_PER_STEP, 0);
  assert.equal(state.step, 'shirt');
  assert.equal(state.progress, 0);
});

test('a done state is returned unchanged', () => {
  const { state: done, now } = advanceTo('shirt');
  const finished = feed(done, shirtFrame(), SAMPLES_PER_STEP, now).state;
  assert.equal(finished.step, 'done');
  const after = stepEnrolment(finished, faceFrame(5), now + 10_000);
  assert.deepEqual(after, finished);
  assert.equal(after.tip, null);
  assert.equal(after.arrow, null);
});

test('stepEnrolment does not mutate the state it was given', () => {
  const before = createEnrolment('Alice', 0);
  const snapshot = structuredClone(before);
  const after = stepEnrolment(before, faceFrame(5), 0);
  assert.notEqual(after, before);
  assert.deepEqual(before, snapshot);
});

test('enrolmentWantsEmbedding is false out of range and true in range once the gap has elapsed', () => {
  let state = createEnrolment('Alice', 0);
  assert.equal(enrolmentWantsEmbedding(state, 30, 1000), false);
  assert.equal(enrolmentWantsEmbedding(state, 5, 1000), true);

  state = stepEnrolment(state, faceFrame(5), 1000);
  assert.equal(enrolmentWantsEmbedding(state, 5, 1000 + SAMPLE_GAP_MS - 1), false);
  assert.equal(enrolmentWantsEmbedding(state, 5, 1000 + SAMPLE_GAP_MS), true);
});

test('enrolmentWantsEmbedding is false on the shirt step and when done', () => {
  const { state: shirt, now } = advanceTo('shirt');
  assert.equal(enrolmentWantsEmbedding(shirt, 0, now + 10_000), false);
  const done = feed(shirt, shirtFrame(), SAMPLES_PER_STEP, now);
  assert.equal(enrolmentWantsEmbedding(done.state, 0, done.now + 10_000), false);
});
