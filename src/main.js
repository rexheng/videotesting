import { startCamera, classifyCameraError, cameraErrorMessage } from './camera.js';

const video = document.getElementById('camera');
const overlay = document.getElementById('overlay');
const status = document.getElementById('status');

async function main() {
  try {
    const { width, height } = await startCamera(video);
    // Match the overlay's drawing surface to the video so overlay pixels line up with video pixels.
    overlay.width = width;
    overlay.height = height;
  } catch (err) {
    status.textContent = cameraErrorMessage(classifyCameraError(err));
  }
}

main();
