// Camera start-up and friendly error handling.
// The pure helpers at the top import nothing and touch no browser globals at import time,
// so they run under plain Node for unit tests.

const REASONS = ['denied', 'not-found', 'unsupported', 'other'];

const MESSAGES = {
  denied:
    "Camera access was blocked. Click the camera icon in your browser's address bar, allow the camera, then reload the page.",
  'not-found': 'No camera found. Plug in a webcam and reload the page.',
  unsupported: "This browser can't use the camera. Try the latest Chrome, Edge, Safari or Firefox.",
  other: "The camera couldn't start. Close other apps using it, then reload the page.",
};

/** Map any error from starting the camera to one of: 'denied' | 'not-found' | 'unsupported' | 'other'. */
export function classifyCameraError(err) {
  if (REASONS.includes(err?.reason)) return err.reason;
  switch (err?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'denied';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'not-found';
    default:
      return 'other';
  }
}

/** The friendly sentence to show for a reason from classifyCameraError. */
export function cameraErrorMessage(reason) {
  return MESSAGES[reason] ?? MESSAGES.other;
}

function cameraError(reason, cause) {
  return Object.assign(new Error(cameraErrorMessage(reason), { cause }), { reason });
}

function whenMetadataLoaded(videoEl) {
  if (videoEl.readyState >= 1) return Promise.resolve();
  return new Promise((resolve, reject) => {
    videoEl.addEventListener('loadedmetadata', () => resolve(), { once: true });
    videoEl.addEventListener('error', () => reject(videoEl.error ?? new Error('video error')), { once: true });
  });
}

/**
 * Start the user-facing webcam in `videoEl`.
 * Resolves with the video's intrinsic size once it is playing.
 * Throws an Error whose `.reason` is one of the four reasons above.
 */
export async function startCamera(videoEl) {
  const mediaDevices = navigator.mediaDevices;
  if (typeof mediaDevices?.getUserMedia !== 'function') throw cameraError('unsupported');

  let stream;
  try {
    stream = await mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    throw cameraError(classifyCameraError(err), err);
  }

  try {
    videoEl.muted = true;
    videoEl.playsInline = true;
    videoEl.srcObject = stream;
    await whenMetadataLoaded(videoEl);
    await videoEl.play();
  } catch (err) {
    stream.getTracks().forEach((track) => track.stop());
    throw cameraError(classifyCameraError(err), err);
  }

  return { width: videoEl.videoWidth, height: videoEl.videoHeight };
}
