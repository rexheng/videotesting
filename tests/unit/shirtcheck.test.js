import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  createShirtCheck,
  beginShirtCheck,
  tickShirtCheck,
  requestCheck,
  shirtCheckView,
} from '../../src/shirtcheck/game.js';
import {
  SHIRTCHECK_FACE_HOLD_MS,
  SHIRTCHECK_DARE_MS,
  SHIRTCHECK_HOLD_MS,
  SHIRTCHECK_SAMPLE_GAP_MS,
  SHIRTCHECK_FACE_GRACE_MS,
} from '../../src/config.js';

const bare = { verdict: 'shirtless', skinRatio: 0.86, relevant: 100, sampled: 100 };
const clothed = { verdict: 'dressed', skinRatio: 0.12, relevant: 100, sampled: 100 };

function holdFace(state, reading = null) {
  let now = 0;
  state = tickShirtCheck(state, now, { face: true, reading });
  assert.equal(state.phase, 'face');
  now = SHIRTCHECK_FACE_HOLD_MS;
  state = tickShirtCheck(state, now, { face: true, reading });
  assert.equal(state.phase, 'dare');
  return { state, now };
}

test('the idle screen states the dare, and a face has to hold before it starts', () => {
  const idle = shirtCheckView(createShirtCheck(), 0);
  assert.equal(idle.prompt, 'Look at me. Then lose the shirt.');
  assert.equal(idle.primaryLabel, 'Start');

  let state = beginShirtCheck(createShirtCheck());
  assert.equal(shirtCheckView(state, 0).prompt, 'Look at the camera.');
  assert.equal(shirtCheckView(state, 0).primaryLabel, 'Wait');
  assert.equal(shirtCheckView(state, 0).primaryDisabled, true);

  state = tickShirtCheck(state, 0, { face: false, reading: null });
  state = tickShirtCheck(state, 5000, { face: false, reading: null });
  assert.equal(state.phase, 'face');

  state = tickShirtCheck(state, 5000, { face: true, reading: null });
  assert.equal(shirtCheckView(state, 5000).prompt, 'Hold still.');
  assert.equal(shirtCheckView(state, 5000).primaryLabel, 'Wait');
  assert.equal(shirtCheckView(state, 5000).primaryDisabled, true);
  state = tickShirtCheck(state, 5000 + SHIRTCHECK_FACE_HOLD_MS - 1, { face: true, reading: null });
  assert.equal(state.phase, 'face');
  const dareAt = 5000 + SHIRTCHECK_FACE_HOLD_MS;
  state = tickShirtCheck(state, dareAt, { face: true, reading: null });
  assert.equal(state.phase, 'dare');
  assert.equal(shirtCheckView(state, dareAt).prompt, 'Shirt off.');
  assert.equal(shirtCheckView(state, dareAt).primaryLabel, 'Call it');
  assert.equal(shirtCheckView(state, dareAt).primaryDisabled, true);
  assert.equal(shirtCheckView(state, dareAt).readout, '');
  state = tickShirtCheck(state, dareAt + 1, {
    face: true,
    reading: { verdict: 'unclear', skinRatio: 0, relevant: 0, sampled: 0 },
  });
  assert.equal(shirtCheckView(state, dareAt + 1).prompt, 'Step back. Shoulders in frame.');
  assert.equal(shirtCheckView(state, dareAt + 1).readout, '');
});

test('looking away resets the face lock', () => {
  let state = beginShirtCheck(createShirtCheck());
  state = tickShirtCheck(state, 0, { face: true, reading: null });
  state = tickShirtCheck(state, 400, { face: false, reading: null });
  assert.equal(state.faceSince, null);
  state = tickShirtCheck(state, 400 + SHIRTCHECK_FACE_HOLD_MS, { face: false, reading: null });
  assert.equal(state.phase, 'face');
});

test('a shirtless reading that holds is confirmed, and one dressed frame breaks the hold', () => {
  let { state, now } = holdFace(beginShirtCheck(createShirtCheck()), bare);
  const dareAt = now;
  let t = dareAt + SHIRTCHECK_SAMPLE_GAP_MS;
  state = tickShirtCheck(state, t, { face: true, reading: bare });
  assert.equal(shirtCheckView(state, t).prompt, "That's skin. Hold.");
  assert.equal(shirtCheckView(state, t).readout, '');
  while (state.phase === 'dare' && t <= dareAt + SHIRTCHECK_SAMPLE_GAP_MS + SHIRTCHECK_HOLD_MS) {
    t += SHIRTCHECK_SAMPLE_GAP_MS;
    state = tickShirtCheck(state, t, { face: true, reading: bare });
  }
  assert.equal(state.phase, 'verdict');
  assert.equal(state.result.verdict, 'shirtless');
  assert.equal(state.result.reason, 'held');
  assert.equal(shirtCheckView(state, t).prompt, "Shirt's off.");
  assert.equal(shirtCheckView(state, t).tip, 'The chest is bare.');
  assert.equal(shirtCheckView(state, t).readout, '');
  assert.equal(shirtCheckView(state, t).stamp, 'OFF');

  ({ state, now } = holdFace(beginShirtCheck(createShirtCheck()), clothed));
  now += SHIRTCHECK_SAMPLE_GAP_MS;
  state = tickShirtCheck(state, now, { face: true, reading: bare });
  const since = state.shirtlessSince;
  now += SHIRTCHECK_SAMPLE_GAP_MS;
  state = tickShirtCheck(state, now, { face: true, reading: clothed });
  assert.equal(state.shirtlessSince, null);
  assert.equal(shirtCheckView(state, now).prompt, 'Take it off.');
  assert.equal(shirtCheckView(state, now).tip, 'Get it off your chest.');
  assert.equal(shirtCheckView(state, now).readout, '');
  now = since + SHIRTCHECK_HOLD_MS;
  state = tickShirtCheck(state, now, { face: true, reading: bare });
  assert.equal(state.phase, 'dare');
});

test('Check calls a shirt that stays on, and does nothing without enough readings', () => {
  let { state, now } = holdFace(beginShirtCheck(createShirtCheck()));
  state = requestCheck(state, now);
  assert.equal(state.phase, 'dare');

  for (let i = 0; i < 3; i++) {
    now += SHIRTCHECK_SAMPLE_GAP_MS;
    state = tickShirtCheck(state, now, { face: true, reading: clothed });
  }
  assert.equal(shirtCheckView(state, now).primaryLabel, 'Call it');
  assert.equal(shirtCheckView(state, now).primaryDisabled, false);
  assert.equal(shirtCheckView(state, now).tip, 'Get it off your chest. Call it.');
  assert.equal(shirtCheckView(state, now).readout, '');
  state = requestCheck(state, now);
  assert.equal(state.result.verdict, 'dressed');
  assert.equal(state.result.reason, 'check');
  assert.equal(shirtCheckView(state, now).prompt, 'Shirt stayed on.');
  assert.equal(shirtCheckView(state, now).tip, 'I still see a shirt.');
  assert.equal(shirtCheckView(state, now).readout, '');
  assert.equal(shirtCheckView(state, now).stamp, 'ON');
});

test('the clock calls unclear when it never saw a torso, and dressed when the readings say so', () => {
  let { state, now } = holdFace(beginShirtCheck(createShirtCheck()));
  const dareStarted = now;
  now = dareStarted + SHIRTCHECK_DARE_MS;
  state = tickShirtCheck(state, now, { face: true, reading: null });
  assert.equal(state.result.verdict, 'unclear');
  assert.equal(state.result.reason, 'timeout');
  assert.equal(shirtCheckView(state, now).prompt, "Couldn't see.");
  assert.equal(shirtCheckView(state, now).tip, 'Try again.');
  assert.equal(shirtCheckView(state, now).readout, '');
  assert.equal(shirtCheckView(state, now).stamp, '?');

  ({ state, now } = holdFace(beginShirtCheck(createShirtCheck())));
  for (let i = 0; i < 4; i++) {
    now += SHIRTCHECK_SAMPLE_GAP_MS;
    state = tickShirtCheck(state, now, { face: true, reading: clothed });
  }
  const dareAt = state.dareStartedAt;
  now = dareAt + SHIRTCHECK_DARE_MS;
  state = tickShirtCheck(state, now, { face: true, reading: clothed });
  assert.equal(state.result.verdict, 'dressed');
  assert.equal(state.result.reason, 'timeout');
  assert.equal(shirtCheckView(state, now).prompt, 'Shirt stayed on.');
});

test('leaving the frame during the dare asks for the face back and drops a shirtless streak', () => {
  let { state, now } = holdFace(beginShirtCheck(createShirtCheck()));
  now += SHIRTCHECK_SAMPLE_GAP_MS;
  state = tickShirtCheck(state, now, { face: true, reading: bare });
  assert.ok(state.shirtlessSince != null);
  now += SHIRTCHECK_FACE_GRACE_MS + 1;
  state = tickShirtCheck(state, now, { face: false, reading: bare });
  assert.equal(state.shirtlessSince, null);
  assert.equal(shirtCheckView(state, now).prompt, 'Come back into frame.');
});

test('a close reading is not a win, and a final call counts it as still dressed', () => {
  const halfway = { verdict: 'close', skinRatio: 0.5, relevant: 100, sampled: 100 };
  let { state, now } = holdFace(beginShirtCheck(createShirtCheck()));
  now += SHIRTCHECK_SAMPLE_GAP_MS;
  state = tickShirtCheck(state, now, { face: true, reading: halfway });
  assert.equal(shirtCheckView(state, now).prompt, 'Pull it off.');
  assert.equal(shirtCheckView(state, now).tip, 'Keep pulling until the chest is bare.');
  assert.equal(shirtCheckView(state, now).readout, '');
  assert.equal(shirtCheckView(state, now).primaryDisabled, true);
  assert.equal(state.phase, 'dare');

  const heldFrom = now;
  while (now < heldFrom + SHIRTCHECK_HOLD_MS) {
    now += SHIRTCHECK_SAMPLE_GAP_MS;
    state = tickShirtCheck(state, now, { face: true, reading: halfway });
    assert.equal(state.phase, 'dare');
  }
  assert.equal(shirtCheckView(state, now).prompt, 'Pull it off.');
  assert.equal(shirtCheckView(state, now).primaryLabel, 'Call it');
  assert.equal(shirtCheckView(state, now).primaryDisabled, false);
  assert.equal(shirtCheckView(state, now).tip, 'Keep pulling until the chest is bare. Call it.');
  assert.equal(shirtCheckView(state, now).readout, '');
  state = requestCheck(state, now);
  assert.equal(state.result.verdict, 'dressed');
  assert.equal(state.result.reason, 'check');
  assert.equal(shirtCheckView(state, now).prompt, 'Shirt stayed on.');
  assert.equal(shirtCheckView(state, now).readout, '');

  ({ state, now } = holdFace(beginShirtCheck(createShirtCheck())));
  for (let i = 0; i < 3; i++) {
    now += SHIRTCHECK_SAMPLE_GAP_MS;
    state = tickShirtCheck(state, now, { face: true, reading: halfway });
  }
  assert.equal(state.phase, 'dare');
  assert.equal(shirtCheckView(state, now).prompt, 'Pull it off.');
  now = state.dareStartedAt + SHIRTCHECK_DARE_MS;
  state = tickShirtCheck(state, now, { face: true, reading: halfway });
  assert.equal(state.result.verdict, 'dressed');
  assert.equal(state.result.reason, 'timeout');
  assert.equal(shirtCheckView(state, now).prompt, 'Shirt stayed on.');
  assert.equal(shirtCheckView(state, now).readout, '');

  ({ state, now } = holdFace(beginShirtCheck(createShirtCheck())));
  now = state.dareStartedAt + SHIRTCHECK_DARE_MS;
  state = tickShirtCheck(state, now, { face: true, reading: halfway });
  assert.equal(state.samples.length < 3, true);
  assert.equal(state.result.verdict, 'unclear');
  assert.equal(shirtCheckView(state, now).prompt, "Couldn't see.");
  assert.equal(shirtCheckView(state, now).tip, 'Try again.');

  const missing = { verdict: 'close', skinRatio: Number.NaN, relevant: 100, sampled: 100 };
  ({ state, now } = holdFace(beginShirtCheck(createShirtCheck())));
  for (let i = 0; i < 3; i++) {
    now += SHIRTCHECK_SAMPLE_GAP_MS;
    state = tickShirtCheck(state, now, { face: true, reading: missing });
  }
  assert.equal(shirtCheckView(state, now).primaryDisabled, true);
  assert.equal(requestCheck(state, now).phase, 'dare');
  now = state.dareStartedAt + SHIRTCHECK_DARE_MS;
  state = tickShirtCheck(state, now, { face: true, reading: missing });
  assert.equal(state.result.verdict, 'unclear');

  state = beginShirtCheck(state);
  assert.equal(state.phase, 'face');
  assert.equal(state.result, null);
  assert.equal(state.samples.length, 0);
});
