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
  assert.equal(idle.prompt, 'Show your face. Then take the shirt off.');
  assert.equal(idle.primaryLabel, 'Start');

  let state = beginShirtCheck(createShirtCheck());
  assert.equal(shirtCheckView(state, 0).prompt, 'Look straight at the camera.');

  state = tickShirtCheck(state, 0, { face: false, reading: null });
  state = tickShirtCheck(state, 5000, { face: false, reading: null });
  assert.equal(state.phase, 'face');

  state = tickShirtCheck(state, 5000, { face: true, reading: null });
  assert.equal(shirtCheckView(state, 5000).prompt, 'Hold still.');
  state = tickShirtCheck(state, 5000 + SHIRTCHECK_FACE_HOLD_MS - 1, { face: true, reading: null });
  assert.equal(state.phase, 'face');
  const dareAt = 5000 + SHIRTCHECK_FACE_HOLD_MS;
  state = tickShirtCheck(state, dareAt, { face: true, reading: null });
  assert.equal(state.phase, 'dare');
  assert.equal(shirtCheckView(state, dareAt).prompt, 'Take the shirt off.');
  state = tickShirtCheck(state, dareAt + 1, {
    face: true,
    reading: { verdict: 'unclear', skinRatio: 0, relevant: 0, sampled: 0 },
  });
  assert.equal(shirtCheckView(state, dareAt + 1).prompt, 'Step back so your shoulders are in view.');
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
  assert.equal(shirtCheckView(state, t).prompt, 'That looks like skin.');
  while (state.phase === 'dare' && t <= dareAt + SHIRTCHECK_SAMPLE_GAP_MS + SHIRTCHECK_HOLD_MS) {
    t += SHIRTCHECK_SAMPLE_GAP_MS;
    state = tickShirtCheck(state, t, { face: true, reading: bare });
  }
  assert.equal(state.phase, 'verdict');
  assert.equal(state.result.verdict, 'shirtless');
  assert.equal(state.result.reason, 'held');
  assert.equal(shirtCheckView(state, t).prompt, 'Shirt off. Confirmed.');
  assert.equal(shirtCheckView(state, t).stamp, 'OFF');

  ({ state, now } = holdFace(beginShirtCheck(createShirtCheck()), clothed));
  now += SHIRTCHECK_SAMPLE_GAP_MS;
  state = tickShirtCheck(state, now, { face: true, reading: bare });
  const since = state.shirtlessSince;
  now += SHIRTCHECK_SAMPLE_GAP_MS;
  state = tickShirtCheck(state, now, { face: true, reading: clothed });
  assert.equal(state.shirtlessSince, null);
  assert.equal(shirtCheckView(state, now).prompt, 'That is still a shirt.');
  now = since + SHIRTCHECK_HOLD_MS;
  state = tickShirtCheck(state, now, { face: true, reading: bare });
  assert.equal(state.phase, 'dare');
});

test('Check me calls a shirt that stays on, and does nothing without enough readings', () => {
  let { state, now } = holdFace(beginShirtCheck(createShirtCheck()));
  state = requestCheck(state, now);
  assert.equal(state.phase, 'dare');

  for (let i = 0; i < 3; i++) {
    now += SHIRTCHECK_SAMPLE_GAP_MS;
    state = tickShirtCheck(state, now, { face: true, reading: clothed });
  }
  assert.equal(shirtCheckView(state, now).primaryDisabled, false);
  state = requestCheck(state, now);
  assert.equal(state.result.verdict, 'dressed');
  assert.equal(state.result.reason, 'check');
  assert.equal(shirtCheckView(state, now).prompt, 'Still wearing it.');
  assert.equal(shirtCheckView(state, now).stamp, 'ON');
});

test('the clock calls unclear when it never saw a torso, and dressed when the readings say so', () => {
  let { state, now } = holdFace(beginShirtCheck(createShirtCheck()));
  const dareStarted = now;
  now = dareStarted + SHIRTCHECK_DARE_MS;
  state = tickShirtCheck(state, now, { face: true, reading: null });
  assert.equal(state.result.verdict, 'unclear');
  assert.equal(state.result.reason, 'timeout');
  assert.equal(shirtCheckView(state, now).prompt, 'No call.');

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
});

test('leaving the frame during the dare asks for the face back and drops a shirtless streak', () => {
  let { state, now } = holdFace(beginShirtCheck(createShirtCheck()));
  now += SHIRTCHECK_SAMPLE_GAP_MS;
  state = tickShirtCheck(state, now, { face: true, reading: bare });
  assert.ok(state.shirtlessSince != null);
  now += SHIRTCHECK_FACE_GRACE_MS + 1;
  state = tickShirtCheck(state, now, { face: false, reading: bare });
  assert.equal(state.shirtlessSince, null);
  assert.equal(shirtCheckView(state, now).prompt, 'Where did you go?');
});

test('a close reading is not a win, and Try again starts a fresh round', () => {
  let { state, now } = holdFace(beginShirtCheck(createShirtCheck()));
  now += SHIRTCHECK_SAMPLE_GAP_MS;
  state = tickShirtCheck(state, now, {
    face: true,
    reading: { verdict: 'close', skinRatio: 0.5, relevant: 100, sampled: 100 },
  });
  assert.equal(shirtCheckView(state, now).prompt, 'Not enough yet.');
  assert.equal(state.phase, 'dare');

  now += SHIRTCHECK_DARE_MS;
  state = tickShirtCheck(state, now, {
    face: true,
    reading: { verdict: 'close', skinRatio: 0.5, relevant: 100, sampled: 100 },
  });
  assert.equal(state.result.verdict, 'unclear');

  state = beginShirtCheck(state);
  assert.equal(state.phase, 'face');
  assert.equal(state.result, null);
  assert.equal(state.samples.length, 0);
});
