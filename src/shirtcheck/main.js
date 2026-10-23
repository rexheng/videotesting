// Shirt Check page: camera, then the face / pose / skin models, then the round.
// ?scripted=1 skips the models and reads window.__shirtFrame each frame, for the browser tests.
import { startCamera, classifyCameraError, cameraErrorMessage } from '../camera.js';
import { createShirtCheck, beginShirtCheck, tickShirtCheck, requestCheck, shirtCheckView } from './game.js';
import { drawShirtCheck } from './draw.js';
import { loadShirtCheckModels, createShirtDetector } from './vision.js';

const MODELS_FAILED = "Couldn't download the shirt checker. Check your internet connection.";
const CAMERA_STOPPED = 'The camera stopped.';
const START_TIMEOUT_MS = 12000;
const EMPTY_OBS = { face: false, reading: null, faceBox: null, polygon: null };

const video = document.getElementById('camera');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d', { willReadFrequently: true });
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
let sawCameraFrame = false;
let startSeq = 0;
let keptStream = null;
let notReadySince = null;
const watchedTracks = new WeakSet();

function showStatus(text) {
  statusEl.textContent = text;
}

function showCameraStatus(err) {
  showStatus(cameraErrorMessage(classifyCameraError(err)));
}

function videoTrackEnded() {
  const stream = video.srcObject;
  if (!stream || typeof stream.getVideoTracks !== 'function') return false;
  const tracks = stream.getVideoTracks();
  return tracks.length > 0 && tracks.every((track) => track.readyState === 'ended');
}

function stopStream(stream) {
  if (!stream || stream === keptStream || typeof stream.getTracks !== 'function') return;
  stream.getTracks().forEach((track) => {
    try {
      track.stop();
    } catch {
      // The track has already ended.
    }
  });
}

// Stop the stream this attempt created. A later attempt's stream stays on the video.
function discardStaleCamera(stream) {
  if (!stream || stream === keptStream) return;
  if (video.srcObject === stream) {
    video.srcObject = keptStream;
    if (keptStream) void video.play?.().catch(() => {});
  }
  stopStream(stream);
}

// startCamera creates the MediaStream and assigns it without returning it. Remember the
// stream so a late resolve can stop that one only. Once this attempt's seq is no longer
// startSeq, leave the live video alone — a srcObject write aborts the newer play().
async function openRoundCamera(owned, seq) {
  const camera = new Proxy(video, {
    get(target, prop) {
      // whenMetadataLoaded returns at HAVE_METADATA, so a stale attempt never attaches
      // listeners or calls play() on the live element. The rejected play() stops this stream.
      if (seq !== startSeq && prop === 'readyState') return HTMLMediaElement.HAVE_METADATA;
      const value = target[prop];
      if (typeof value !== 'function') return value;
      const bound = value.bind(target);
      return (...args) => {
        if (seq !== startSeq) return abandonedCameraCall(prop);
        return bound(...args);
      };
    },
    set(target, prop, value) {
      if (prop === 'srcObject') owned.stream = value;
      if (seq !== startSeq) return true;
      target[prop] = value;
      return true;
    },
  });
  return startCamera(camera);
}

function abandonedCameraCall(prop) {
  if (prop === 'play') return Promise.reject(new Error('abandoned camera attempt'));
  return undefined;
}

// The camera was producing frames, then the track ended or the element lost the picture.
function reportCameraStopped() {
  if (!cameraReady) return;
  cameraReady = false;
  sawCameraFrame = false;
  notReadySince = null;
  showStatus(CAMERA_STOPPED);
  if (!starting) return;
  // Drop the hung start so its later resolve cannot clear a newer attempt.
  startSeq += 1;
  starting = false;
  primary.disabled = false;
  render(performance.now());
}

function watchCamera() {
  const stream = video.srcObject;
  const tracks = stream && typeof stream.getVideoTracks === 'function' ? stream.getVideoTracks() : [];
  for (const track of tracks) {
    if (watchedTracks.has(track)) continue;
    watchedTracks.add(track);
    track.onended = reportCameraStopped;
  }
  if (videoTrackEnded()) reportCameraStopped();
}

function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

// Stylesheet reads these on every frame. reading is empty on idle and face,
// the live verdict during the dare, and the called result on a verdict.
function readingAttr(state) {
  if (state.phase === 'verdict') return state.result ? state.result.verdict : '';
  if (state.phase === 'dare') return state.lastReading?.verdict || '';
  return '';
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

function clearOverlay() {
  const width = overlay.width;
  const height = overlay.height;
  if (!(width > 0) || !(height > 0)) return;
  // Reset the bitmap so a verdict cannot keep the previous guide.
  overlay.width = width;
  overlay.height = height;
}

function render(now) {
  if (state.phase === 'verdict') clearOverlay();
  const view = shirtCheckView(state, now);
  document.body.dataset.phase = state.phase;
  document.body.dataset.reading = readingAttr(state);
  setText(promptEl, view.prompt);
  setText(tipEl, view.tip);
  setText(primary, view.primaryLabel);
  primary.disabled = view.primaryDisabled || starting;
  secondary.hidden = !view.showQuit;
  meter.hidden = !view.showMeter;
  meter.setAttribute('aria-valuenow', String(Math.round(view.meter * 100)));
  meterFill.style.width = `${Math.round(view.meter * 100)}%`;
  meter.dataset.kind = view.stampKind || state.lastReading?.verdict || '';
  setText(readout, view.readout);
  setText(clock, view.clock);
  stamp.hidden = !view.stamp;
  setText(stamp, view.stamp);
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
    let now = 0;
    try {
      now = performance.now();
      const frameReady = video.readyState >= 2;
      const showingKept = keptStream != null && video.srcObject === keptStream;
      // Scripted rounds have no camera. A missing element frame is not a stopped camera.
      if (!scripted && cameraReady && showingKept && videoTrackEnded()) reportCameraStopped();
      if (frameReady) {
        sawCameraFrame = true;
        notReadySince = null;
      } else if (!scripted && cameraReady && showingKept && sawCameraFrame) {
        // A srcObject swap dips readyState for a frame. Wait before calling it stopped.
        notReadySince = notReadySince ?? now;
        if (now - notReadySince >= 500) reportCameraStopped();
      } else {
        notReadySince = null;
      }
      if (state.phase === 'face' || state.phase === 'dare') {
        // A stopped camera keeps a frozen picture with readyState >= 2. Do not score it.
        const cameraStopped = !scripted && !cameraReady;
        let obs = null;
        if (!cameraStopped && (scripted || frameReady)) {
          try {
            if (frameReady) {
              const width = video.videoWidth;
              const height = video.videoHeight;
              if (width > 0 && overlay.width !== width) {
                overlay.width = width;
                overlay.height = height;
              }
            }
            obs = scripted ? readScripted() : detector.detect(video, now);
            if (!scripted && !cameraReady) obs = null;
          } catch (err) {
            // A missing or thrown frame must not replay the previous reading.
            console.warn('Shirt Check skipped a frame:', err);
          }
        }
        // Empty frames let the dare clock finish without confirming a frozen shirtless hold.
        state = tickShirtCheck(state, now, obs || EMPTY_OBS);
        if (state.phase === 'verdict') clearOverlay();
        else if (obs && overlay.width > 0) drawShirtCheck(overlayCtx, overlay.width, overlay.height, obs);
      }
    } catch (err) {
      console.warn('Shirt Check skipped a frame:', err);
    }
    try {
      render(now);
    } catch (err) {
      console.warn('Shirt Check skipped a frame:', err);
    }
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
  const seq = ++startSeq;
  render(performance.now());
  let abandoned = false;
  const owned = { stream: null };
  const timer = setTimeout(() => {
    if (seq !== startSeq || abandoned) return;
    abandoned = true;
    startSeq += 1;
    starting = false;
    primary.disabled = false;
    if (statusEl.textContent !== CAMERA_STOPPED) {
      if (cameraReady) showModelsFailed();
      else showCameraStatus({ reason: 'other' });
    }
    render(performance.now());
  }, START_TIMEOUT_MS);
  try {
    if (!cameraReady) {
      const { width, height } = await openRoundCamera(owned, seq);
      if (seq !== startSeq || abandoned) {
        discardStaleCamera(owned.stream);
        return;
      }
      overlay.width = width;
      overlay.height = height;
      cameraReady = true;
      keptStream = owned.stream ?? video.srcObject;
      // A late abandoned startCamera can overwrite the element before this attempt finishes.
      if (owned.stream && video.srcObject !== owned.stream) {
        video.srcObject = owned.stream;
        void video.play?.().catch(() => {});
      }
      watchCamera();
      if (seq !== startSeq || abandoned || !cameraReady) return;
    }
    if (!scripted && !models) {
      showStatus('Loading the shirt checker…');
      const loaded = await loadShirtCheckModels({ runningMode: 'VIDEO' });
      if (seq !== startSeq || abandoned || !cameraReady) return;
      models = loaded;
      detector = createShirtDetector(models);
      showStatus('');
    }
    if (seq !== startSeq || abandoned || !cameraReady) return;
    if (statusEl.textContent) showStatus('');
    state = beginShirtCheck(state);
    ensureLoop();
  } catch (err) {
    if (seq !== startSeq || abandoned) {
      discardStaleCamera(owned.stream);
      return;
    }
    console.error(err);
    if (!cameraReady) showCameraStatus(err);
    else showModelsFailed();
  } finally {
    clearTimeout(timer);
    if (seq !== startSeq || abandoned) return;
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
  clearOverlay();
  render(performance.now());
});

render(performance.now());
