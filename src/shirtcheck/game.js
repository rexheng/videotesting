// Shirt Check rounds. Pure state machine: the page feeds it "is there a face?" and a torso reading
// each frame, and this decides the prompt, the meter and when to call the result.
//
// A round: look at the camera until the face holds, then take the shirt off. A shirtless reading
// that holds still is a win. Call it forces a call from the recent readings. The clock calls
// whatever the recent readings say when time runs out. A half-off shirt stays in the dare; a
// finished call counts it as still dressed.

import { median } from '../vision/geometry.js';
import { verdictFromRatio } from './bare.js';
import {
  SHIRTCHECK_FACE_HOLD_MS,
  SHIRTCHECK_DARE_MS,
  SHIRTCHECK_HOLD_MS,
  SHIRTCHECK_FACE_GRACE_MS,
  SHIRTCHECK_SAMPLE_GAP_MS,
} from '../config.js';

function blank() {
  return {
    phase: 'idle',
    faceSince: null,
    dareStartedAt: null,
    lastFaceAt: null,
    lastReading: null,
    lastSampleAt: null,
    lastShirtlessAt: null,
    shirtlessSince: null,
    samples: [],
    result: null,
  };
}

/** Idle round, before Start. */
export function createShirtCheck() {
  return blank();
}

/** Leave idle or a verdict and start looking for a face. */
export function beginShirtCheck(state) {
  if (state.phase !== 'idle' && state.phase !== 'verdict') return state;
  return { ...blank(), phase: 'face' };
}

function faceIsCurrent(state, now) {
  return state.lastFaceAt != null && now - state.lastFaceAt <= SHIRTCHECK_FACE_GRACE_MS;
}

function finish(state, verdict, skinRatio, reason) {
  return {
    ...state,
    phase: 'verdict',
    shirtlessSince: null,
    result: { verdict, skinRatio, reason },
  };
}

/**
 * A finished call from a measured ratio.
 * Shirtless only at the shirtless line. The live close band, and anything more clothed, is
 * dressed: the shirt was in frame and did not come off. Null or NaN stays unclear.
 */
function finalVerdict(ratio) {
  const verdict = verdictFromRatio(ratio);
  if (verdict === 'unclear' && ratio != null && !Number.isNaN(ratio)) return 'dressed';
  return verdict;
}

/**
 * Recent samples that agree enough for the Call it button.
 * At least three samples in the last 1.2s, with a ratio that is not null or NaN.
 * Returns { verdict, skinRatio } or null.
 */
export function checkDecision(samples, now) {
  const recent = samples.filter((sample) => now - sample.t <= 1200);
  if (recent.length < 3) return null;
  const skinRatio = median(recent.map((sample) => sample.skinRatio));
  const verdict = finalVerdict(skinRatio);
  if (verdict === 'unclear') return null;
  return { verdict, skinRatio };
}

/**
 * One frame. `obs` is { face: boolean, reading: { verdict, skinRatio } | null }.
 * `reading.verdict` is shirtless, dressed, close or unclear.
 */
export function tickShirtCheck(state, now, obs) {
  if (state.phase === 'face') {
    if (!obs.face) return { ...state, faceSince: null };
    const faceSince = state.faceSince ?? now;
    if (now - faceSince >= SHIRTCHECK_FACE_HOLD_MS) {
      return {
        ...state,
        phase: 'dare',
        faceSince,
        dareStartedAt: now,
        lastFaceAt: now,
        lastReading: null,
        samples: [],
        shirtlessSince: null,
      };
    }
    return { ...state, faceSince, lastFaceAt: now };
  }

  if (state.phase !== 'dare') return state;

  let next = { ...state, lastFaceAt: obs.face ? now : state.lastFaceAt };
  const faceOk = obs.face || faceIsCurrent(state, now);
  if (!faceOk) {
    next = { ...next, shirtlessSince: null };
  } else {
    const reading = obs.reading;
    const gapOpen = now - (state.lastSampleAt ?? -Infinity) >= SHIRTCHECK_SAMPLE_GAP_MS;
    if (reading && reading.verdict !== 'unclear' && gapOpen) {
      const samples = next.samples.concat({
        t: now,
        skinRatio: reading.skinRatio,
        verdict: reading.verdict,
      });
      next = { ...next, samples, lastSampleAt: now, lastReading: reading };
      if (reading.verdict === 'shirtless') {
        const gap = state.lastShirtlessAt == null ? Infinity : now - state.lastShirtlessAt;
        const shirtlessSince = state.shirtlessSince == null || gap > SHIRTCHECK_SAMPLE_GAP_MS * 3
          ? now
          : state.shirtlessSince;
        next = { ...next, shirtlessSince, lastShirtlessAt: now };
        if (now - shirtlessSince >= SHIRTCHECK_HOLD_MS) {
          const ratios = samples
            .filter((sample) => sample.t >= shirtlessSince && sample.verdict === 'shirtless')
            .map((sample) => sample.skinRatio);
          return finish(next, 'shirtless', median(ratios), 'held');
        }
      } else {
        next = { ...next, shirtlessSince: null };
      }
    } else {
      next = { ...next, lastReading: reading ?? null };
    }
  }

  if (now - next.dareStartedAt >= SHIRTCHECK_DARE_MS) {
    const ratio = next.samples.length ? median(next.samples.map((sample) => sample.skinRatio)) : null;
    const verdict = next.samples.length >= 3 ? finalVerdict(ratio) : 'unclear';
    return finish(next, verdict, ratio, 'timeout');
  }
  return next;
}

/** Force a call from the recent readings. No change when there are not enough, or outside the dare. */
export function requestCheck(state, now) {
  if (state.phase !== 'dare') return state;
  const decision = checkDecision(state.samples, now);
  if (!decision) return state;
  return finish(state, decision.verdict, decision.skinRatio, 'check');
}

function percent(ratio) {
  return ratio == null || Number.isNaN(ratio) ? null : Math.round(ratio * 100);
}

/**
 * Everything the page needs to draw one moment. `meter` is 0–1 (cloth at 0, skin at 1).
 */
export function shirtCheckView(state, now) {
  if (state.phase === 'idle') {
    return {
      prompt: 'Look at me. Then lose the shirt.',
      tip: 'Upper body only. The picture stays in this browser and nothing is recorded.',
      primaryLabel: 'Start',
      primaryDisabled: false,
      showQuit: false,
      showMeter: false,
      meter: 0,
      readout: '',
      clock: '',
      stamp: '',
      stampKind: '',
      faceProgress: 0,
    };
  }

  if (state.phase === 'face') {
    const locking = state.faceSince != null;
    return {
      prompt: locking ? 'Hold still.' : 'Look at the camera.',
      tip: locking ? 'Eyes on the camera.' : "I'll start once I see your face.",
      primaryLabel: 'Wait',
      primaryDisabled: true,
      showQuit: true,
      showMeter: false,
      meter: 0,
      readout: '',
      clock: '',
      stamp: '',
      stampKind: '',
      faceProgress: locking ? Math.min(1, (now - state.faceSince) / SHIRTCHECK_FACE_HOLD_MS) : 0,
    };
  }

  if (state.phase === 'dare') {
    const here = faceIsCurrent(state, now);
    const reading = state.lastReading;
    const ratio = reading && reading.verdict !== 'unclear' ? reading.skinRatio : 0;
    const decision = checkDecision(state.samples, now);
    let prompt = 'Shirt off.';
    let tip = "Shoulders in frame. I'll call it when I see skin.";
    if (!here) {
      prompt = 'Come back into frame.';
      tip = 'Look at the camera.';
    } else if (reading?.verdict === 'unclear') {
      prompt = 'Step back. Shoulders in frame.';
      tip = 'I need your chest in frame.';
    } else if (reading?.verdict === 'shirtless') {
      prompt = "That's skin. Hold.";
      tip = 'Stay there.';
    } else if (reading?.verdict === 'close') {
      prompt = 'Pull it off.';
      tip = 'Keep pulling until the chest is bare.';
    } else if (reading?.verdict === 'dressed') {
      prompt = 'Take it off.';
      tip = 'Get it off your chest.';
    }
    // The press ends the round. Name the button once it can be pressed.
    if (decision && !tip.includes('Call it')) tip = `${tip} Call it.`;
    const remain = Math.max(0, SHIRTCHECK_DARE_MS - (now - state.dareStartedAt));
    return {
      prompt,
      tip,
      primaryLabel: 'Call it',
      primaryDisabled: !decision,
      showQuit: true,
      showMeter: true,
      meter: ratio,
      readout: '',
      clock: `${Math.ceil(remain / 1000)}s left`,
      stamp: '',
      stampKind: '',
      faceProgress: 1,
    };
  }

  const result = state.result;
  const pct = percent(result?.skinRatio);
  if (result?.verdict === 'shirtless') {
    return {
      prompt: "Shirt's off.",
      tip: pct == null ? '' : 'The chest is bare.',
      primaryLabel: 'Again',
      primaryDisabled: false,
      showQuit: false,
      showMeter: pct != null,
      meter: result.skinRatio ?? 0,
      readout: '',
      clock: '',
      stamp: 'OFF',
      stampKind: 'shirtless',
      faceProgress: 0,
    };
  }
  if (result?.verdict === 'dressed') {
    return {
      prompt: 'Shirt stayed on.',
      tip: 'I still see a shirt.',
      primaryLabel: 'Again',
      primaryDisabled: false,
      showQuit: false,
      showMeter: pct != null,
      meter: result.skinRatio ?? 0,
      readout: '',
      clock: '',
      stamp: 'ON',
      stampKind: 'dressed',
      faceProgress: 0,
    };
  }
  return {
    prompt: "Couldn't see.",
    tip: 'Try again.',
    primaryLabel: 'Again',
    primaryDisabled: false,
    showQuit: false,
    showMeter: false,
    meter: 0,
    readout: '',
    clock: '',
    stamp: '?',
    stampKind: 'unclear',
    faceProgress: 0,
  };
}
