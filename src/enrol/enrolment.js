// Sign-up state machine: pure and DOM-free. Each frame the app passes in the head yaw, an optional
// face embedding and an optional shirt histogram, and gets back a new state to render. Steps run
// straight -> left -> right -> shirt -> done; shirt-only mode ("Re-scan shirt") starts at shirt.
//
// Yaw sign: positive means the user turned to their own left. The display is mirrored, so that is
// screen-left, and the left step's arrow is 'left'.

import {
  YAW_STRAIGHT,
  YAW_TURNED,
  SAMPLES_PER_STEP,
  SAMPLE_GAP_MS,
  NO_FACE_TIP_MS,
  STUCK_TIP_MS,
} from '../config.js';
import { median } from '../vision/geometry.js';
import { averageHistograms } from '../recognition/shirt.js';

const FACE_STEPS = ['straight', 'left', 'right'];
const NEXT_STEP = { straight: 'left', left: 'right', right: 'shirt', shirt: 'done' };

const PROMPTS = {
  straight: 'Look straight at the camera',
  left: 'Slowly turn your head LEFT',
  right: 'Now turn RIGHT',
  shirt: 'Step back so we can see your shirt',
};
const ARROWS = { left: 'left', right: 'right' };

const NO_FACE_TIP = 'Move closer, or try more light';
const NO_SHIRT_TIP = 'Step back so your shoulders are in view';
const STUCK_TIPS = {
  straight: 'Face the camera and hold still',
  left: 'Turn a bit further',
  right: 'Turn a bit further',
};

/** Whether this yaw counts for a face step, given the straight-ahead offset. */
function yawInRange(step, yaw, yawOffset) {
  if (step === 'straight') return Math.abs(yaw) <= YAW_STRAIGHT;
  if (step === 'left') return yaw - yawOffset >= YAW_TURNED;
  if (step === 'right') return yaw - yawOffset <= -YAW_TURNED;
  return false;
}

function donePrompt(mode, name) {
  return mode === 'shirt' ? `Shirt updated for ${name} 👕` : `Done! Hi ${name} 👋`;
}

/** The fields the UI renders for a step, plus the timers that restart whenever a step begins. */
function stepView(state, step, now) {
  const done = step === 'done';
  return {
    step,
    progress: done ? 1 : 0,
    prompt: done ? donePrompt(state.mode, state.name) : PROMPTS[step],
    tip: null,
    arrow: ARROWS[step] ?? null,
    _samples: 0,
    _seenAt: now,
    _stuckSince: now,
  };
}

/**
 * A new enrolment for `name`. `startAt: 'shirt'` is for Re-scan shirt: it takes only the shirt step
 * and leaves the stored face alone.
 */
export function createEnrolment(name, now, { startAt = 'straight' } = {}) {
  const shirtOnly = startAt === 'shirt';
  const base = {
    name,
    mode: shirtOnly ? 'shirt' : 'full',
    yawOffset: 0,
    result: null,
    _yaws: [],
    _embeddings: [],
    _histograms: [],
    _swatch: null,
    _lastSampleAt: -Infinity,
  };
  return { ...base, ...stepView(base, shirtOnly ? 'shirt' : 'straight', now) };
}

/** True when this frame's face is worth embedding: a face step, yaw in range, gap elapsed. */
export function enrolmentWantsEmbedding(state, yaw, now) {
  return (
    FACE_STEPS.includes(state.step) &&
    yawInRange(state.step, yaw, state.yawOffset) &&
    now - state._lastSampleAt >= SAMPLE_GAP_MS
  );
}

/** The tip for a step that took no sample this frame, or null. `present` = input seen this frame. */
function tipFor(state, present, now) {
  if (!present && now - state._seenAt >= NO_FACE_TIP_MS) {
    return state.step === 'shirt' ? NO_SHIRT_TIP : NO_FACE_TIP;
  }
  if (present && state.step !== 'shirt' && now - state._stuckSince >= STUCK_TIP_MS) {
    return STUCK_TIPS[state.step];
  }
  return null;
}

/** Moves a state whose current step just filled up on to the next step (or to done). */
function advance(state, now) {
  const step = NEXT_STEP[state.step];
  const next = { ...state };
  if (state.step === 'straight') next.yawOffset = median(state._yaws);
  if (step === 'done') {
    next.result = {
      faceEmbeddings: state._embeddings,
      shirtHistogram: averageHistograms(state._histograms),
      shirtSwatch: state._swatch,
    };
  }
  return { ...next, ...stepView(next, step, now) };
}

/** One frame of input. Returns a new state; the one passed in is never changed. */
export function stepEnrolment(state, frame, now) {
  if (state.step === 'done') return state;

  const onShirt = state.step === 'shirt';
  const present = Boolean(onShirt ? frame.shirt : frame.face);
  const gapElapsed = now - state._lastSampleAt >= SAMPLE_GAP_MS;
  const sampled = onShirt
    ? present && gapElapsed
    : present &&
      gapElapsed &&
      frame.face.embedding != null &&
      yawInRange(state.step, frame.face.yaw, state.yawOffset);

  if (!sampled) {
    const seen = present ? { ...state, _seenAt: now } : state;
    return { ...seen, tip: tipFor(state, present, now) };
  }

  const samples = state._samples + 1;
  const next = {
    ...state,
    _samples: samples,
    _seenAt: now,
    _stuckSince: now,
    _lastSampleAt: now,
    progress: samples / SAMPLES_PER_STEP,
    tip: null,
  };
  if (onShirt) {
    next._histograms = [...state._histograms, frame.shirt.histogram];
    next._swatch = frame.shirt.swatch;
  } else {
    next._embeddings = [...state._embeddings, frame.face.embedding];
    if (state.step === 'straight') next._yaws = [...state._yaws, frame.face.yaw];
  }
  return samples >= SAMPLES_PER_STEP ? advance(next, now) : next;
}
