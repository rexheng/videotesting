// Shirt Check page: camera, then the face / pose / skin models, then the round.
// ?scripted=1 skips the models and reads window.__shirtFrame each frame, for the browser tests.
import { startCamera, classifyCameraError, cameraErrorMessage } from '../camera.js';
import { createShirtCheck, beginShirtCheck, tickShirtCheck, requestCheck, shirtCheckView } from './game.js';
import { drawShirtCheck } from './draw.js';
import { loadShirtCheckModels, createShirtDetector } from './vision.js';

const MODELS_FAILED = "Couldn't download the shirt checker. Check your internet connection.";

const video = document.getElementById('camera');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');
const promptEl = document.getElementById('prompt');
const tipEl = document.getElementById('tip');
const primary = document.getElementById('primary');
const secondary = document.getElementById('secondary');
const meter = document.getElementById('meter');
const meterFill = document.getElementById('meter-fill');
const readout = document.getElementById('readout');
const clock = document.getElementById('clock');
const stamp = document.getElementById('stamp');
const faceLock = document.getElementById('face-lock');
const faceLockFill = document.getElementById('face-lock-fill');
const statusEl = document.getElementById('status');

const scripted = new URLSearchParams(location.search).get('scripted') === '1';

let state = createShirtCheck();
let cameraReady = false;
let models = null;
let detector = null;
let starting = false;
let generation = 0;
let looping = false;

function showStatus(text) {
  statusEl.textContent = text;
}

function showModelsFailed() {
  const retry = document.createElement('button');
  retry.id = 'retry';
  retry.type = 'button';
  retry.textContent = 'Try again';
  retry.addEventListener('click', () => location.reload());
  statusEl.replaceChildren(document.createTextNode(`${MODELS_FAILED} `), retry);
}

function readScripted() {
  const frame = window.__shirtFrame || { face: false, reading: null };
  return {
    face: !!frame.face,
    faceBox: frame.faceBox || null,
    polygon: frame.polygon || null,
    reading: frame.reading || null,
  };
}

function render(now) {
  const view = shirtCheckView(state, now);
  promptEl.textContent = view.prompt;
  tipEl.textContent = view.tip;
  primary.textContent = view.primaryLabel;
  primary.disabled = view.primaryDisabled || starting;
  secondary.hidden = !view.showQuit;
  meter.hidden = !view.showMeter;
  meter.setAttribute('aria-valuenow', String(Math.round(view.meter * 100)));
  meterFill.style.width = `${Math.round(view.meter * 100)}%`;
  meter.dataset.kind = view.stampKind || state.lastReading?.verdict || '';
  readout.textContent = view.readout;
  clock.textContent = view.clock;
  stamp.hidden = !view.stamp;
  stamp.textContent = view.stamp;
  stamp.dataset.kind = view.stampKind;
  faceLock.hidden = state.phase !== 'face';
  faceLockFill.style.width = `${Math.round(view.faceProgress * 100)}%`;
}

function stopLoop() {
  generation += 1;
  looping = false;
}

function ensureLoop() {
  if (looping) return;
  looping = true;
  const gen = generation;
  const step = () => {
    if (gen !== generation) return;
    const now = performance.now();
    if ((state.phase === 'face' || state.phase === 'dare') && video.readyState >= 2) {
      const width = video.videoWidth;
      const height = video.videoHeight;
      if (width > 0 && overlay.width !== width) {
        overlay.width = width;
        overlay.height = height;
      }
      try {
        const obs = scripted ? readScripted() : detector.detect(video, now);
        state = tickShirtCheck(state, now, obs);
        if (overlay.width > 0) drawShirtCheck(overlayCtx, overlay.width, overlay.height, obs);
      } catch (err) {
        console.warn('Shirt Check skipped a frame:', err);
      }
    }
    render(now);
    if (gen !== generation || state.phase === 'idle' || state.phase === 'verdict') {
      looping = false;
      return;
    }
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

async function startRound() {
  if (starting) return;
  if (state.phase !== 'idle' && state.phase !== 'verdict') return;
  starting = true;
  render(performance.now());
  try {
    if (!cameraReady) {
      const { width, height } = await startCamera(video);
      overlay.width = width;
      overlay.height = height;
      cameraReady = true;
    }
    if (!scripted && !models) {
      showStatus('Loading the shirt checker…');
      models = await loadShirtCheckModels({ runningMode: 'VIDEO' });
      detector = createShirtDetector(models);
      showStatus('');
    }
    state = beginShirtCheck(state);
    ensureLoop();
  } catch (err) {
    console.error(err);
    if (!cameraReady) showStatus(cameraErrorMessage(classifyCameraError(err)));
    else showModelsFailed();
  } finally {
    starting = false;
    render(performance.now());
  }
}

primary.addEventListener('click', () => {
  if (state.phase === 'dare') {
    state = requestCheck(state, performance.now());
    render(performance.now());
    if (state.phase === 'verdict') stopLoop();
    return;
  }
  startRound();
});

secondary.addEventListener('click', () => {
  stopLoop();
  state = createShirtCheck();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  render(performance.now());
});

render(performance.now());
