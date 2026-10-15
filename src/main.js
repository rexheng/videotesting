// Start-up and wiring: camera, then the face models, then the live tagging app.
import { startCamera, classifyCameraError, cameraErrorMessage } from './camera.js';
import { loadModels } from './vision/models.js';
import { createStore, NOT_SAVED_MESSAGE } from './storage.js';
import { createApp } from './app.js';
import { createDebugPanel } from './ui/debugPanel.js';

const LOADING_MESSAGE = 'Loading face models…';
const MODELS_FAILED_MESSAGE = "Couldn't download the face models. Check your internet connection.";

const video = document.getElementById('camera');
const overlay = document.getElementById('overlay');
const status = document.getElementById('status');
const liveTags = document.getElementById('live-tags');
const debugPre = document.getElementById('debug');

function showStatus(text) {
  status.textContent = text;
}

/** Tell the user people will not be kept: storage is unavailable, or a save failed. */
function showNotSaved() {
  showStatus(NOT_SAVED_MESSAGE);
}

function showModelsFailed() {
  const retry = document.createElement('button');
  retry.id = 'retry';
  retry.type = 'button';
  retry.textContent = 'Try again';
  // Reload rather than retry in place: the browser caches a failed module import from the CDN.
  retry.addEventListener('click', () => location.reload());
  status.replaceChildren(MODELS_FAILED_MESSAGE, ' ', retry);
}

let lastTagsKey = null;

/** Mirror the labels into the visually hidden #live-tags list, only when the texts change. */
function showLiveTags(labels) {
  const texts = labels.map((label) => label.text).sort();
  const key = JSON.stringify(texts);
  if (key === lastTagsKey) return;
  lastTagsKey = key;
  liveTags.replaceChildren(
    ...texts.map((text) => {
      const item = document.createElement('li');
      item.textContent = text;
      return item;
    }),
  );
}

async function main() {
  const store = createStore();
  const debugPanel = new URLSearchParams(location.search).get('debug') === '1' ? createDebugPanel(debugPre) : null;

  try {
    const { width, height } = await startCamera(video);
    // Match the overlay's drawing surface to the video so overlay pixels line up with video pixels.
    overlay.width = width;
    overlay.height = height;
  } catch (err) {
    showStatus(cameraErrorMessage(classifyCameraError(err)));
    return;
  }

  showStatus(LOADING_MESSAGE);
  let models;
  try {
    models = await loadModels({ runningMode: 'VIDEO' });
  } catch (err) {
    console.error('Face models failed to load:', err);
    showModelsFailed();
    return;
  }

  showStatus('');
  if (!store.available) showNotSaved();

  const app = createApp({ video, overlay, models, store, onLabels: showLiveTags, onDebug: debugPanel?.update });
  app.start();
}

main();
