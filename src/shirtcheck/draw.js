// Draw the face box and the torso quad on the overlay.
// The video element is mirrored with CSS; this canvas is not, so x is flipped here, the same way
// the tagger's labels are. Coordinates on `obs` are normalised (0–1) on the raw frame.

import { mirrorBox } from '../ui/overlay.js';

const STROKE = {
  shirtless: '#e6194b',
  dressed: '#4363d8',
  close: '#f58231',
  unclear: '#f4f1ea',
};

/** Clear the canvas and draw this frame's face box and torso, mirrored to match the video. */
export function drawShirtCheck(ctx, width, height, obs) {
  ctx.clearRect(0, 0, width, height);
  if (!obs) return;
  ctx.lineWidth = Math.max(3, Math.round(width / 180));

  const verdict = obs.reading?.verdict;
  if (obs.polygon?.length) {
    ctx.beginPath();
    obs.polygon.forEach(([nx, ny], index) => {
      const x = (1 - nx) * width;
      const y = ny * height;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = verdict === 'shirtless' ? 'rgba(230, 25, 75, 0.28)' : 'rgba(244, 241, 234, 0.14)';
    ctx.strokeStyle = STROKE[verdict] || STROKE.unclear;
    ctx.fill();
    ctx.stroke();
  }

  if (obs.faceBox) {
    const box = mirrorBox(
      {
        x: obs.faceBox.x * width,
        y: obs.faceBox.y * height,
        width: obs.faceBox.width * width,
        height: obs.faceBox.height * height,
      },
      width,
    );
    ctx.strokeStyle = '#f4f1ea';
    ctx.strokeRect(box.x, box.y, box.width, box.height);
  }
}
