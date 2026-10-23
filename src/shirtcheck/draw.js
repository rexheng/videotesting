// Shirt Check camera guide: four corner brackets on the face, a hairline on the torso.
// The video element is mirrored with CSS; this canvas is not, so x is flipped here
// (mirrorBox for the face, 1 - nx for each polygon point), the same way the tagger's
// labels are. Coordinates on `obs` are normalised (0–1) on the raw frame.
// No text — it would read backwards on this unmirrored canvas.

import { mirrorBox } from '../ui/overlay.js';

const HAIRLINE = '#f4f1ea';
const ACCENT = '#ff4b3a';
const CLOSE = 'rgba(255, 75, 58, 0.55)';

/** Hairline in frame pixels: 2px at 1280, clamped to about 1.5–2.5. */
function guideWidth(frameWidth) {
  return Math.min(2.5, Math.max(1.5, frameWidth / 640));
}

/** Neutral hairline when dressed or unclear; page accent when shirtless; same accent at 0.55 when close. */
function torsoColour(verdict) {
  if (verdict === 'shirtless') return ACCENT;
  if (verdict === 'close') return CLOSE;
  return HAIRLINE;
}

function strokeTorso(ctx, polygon, width, height) {
  ctx.beginPath();
  let count = 0;
  for (const point of polygon) {
    const nx = point?.[0];
    const ny = point?.[1];
    if (!Number.isFinite(nx) || !Number.isFinite(ny)) continue;
    const x = (1 - nx) * width;
    const y = ny * height;
    if (count === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    count += 1;
  }
  if (count < 2) return;
  ctx.closePath();
  ctx.stroke();
}

/** One viewfinder corner. (ax, ay) are the arm offsets leaving the corner. */
function traceCorner(ctx, cx, cy, ax, ay, radius) {
  ctx.moveTo(cx, cy + ay);
  ctx.arcTo(cx, cy, cx + ax, cy, radius);
  ctx.lineTo(cx + ax, cy);
}

function strokeFaceBrackets(ctx, box, lineWidth) {
  const { x, y, width: w, height: h } = box;
  if (!(w > 2) || !(h > 2)) return;
  const reach = Math.min(w, h) * 0.25;
  const ax = Math.min(reach, w * 0.46);
  const ay = Math.min(reach, h * 0.46);
  const radius = Math.min(lineWidth * 2.5, ax * 0.35, ay * 0.35);
  ctx.beginPath();
  traceCorner(ctx, x, y, ax, ay, radius);
  traceCorner(ctx, x + w, y, -ax, ay, radius);
  traceCorner(ctx, x + w, y + h, -ax, -ay, radius);
  traceCorner(ctx, x, y + h, ax, -ay, radius);
  ctx.stroke();
}

/** Clear the canvas and draw this frame's face brackets and torso outline, mirrored to the video. */
export function drawShirtCheck(ctx, width, height, obs) {
  ctx.clearRect(0, 0, width, height);
  if (!obs || !(width > 0) || !(height > 0)) return;

  const lineWidth = guideWidth(width);
  ctx.save();
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.setLineDash([]);
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;

  if (obs.polygon?.length >= 2) {
    ctx.strokeStyle = torsoColour(obs.reading?.verdict);
    strokeTorso(ctx, obs.polygon, width, height);
  }

  const face = obs.faceBox;
  if (face && face.width > 0 && face.height > 0) {
    const box = mirrorBox(
      {
        x: face.x * width,
        y: face.y * height,
        width: face.width * width,
        height: face.height * height,
      },
      width,
    );
    ctx.strokeStyle = HAIRLINE;
    strokeFaceBrackets(ctx, box, lineWidth);
  }

  ctx.restore();
}
