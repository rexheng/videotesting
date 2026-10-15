// Overlay drawing: name labels on the canvas that sits over the mirrored video.
// The canvas itself is NOT mirrored (so text reads normally); instead each box's x is flipped here.
// Boxes arrive in pixels of the raw, unmirrored frame, and the canvas is sized to that frame.
// No DOM or CDN imports: drawLabels only uses the 2D context it is given.

/** Label colour for a face nobody signed up with. */
export const UNKNOWN_COLOUR = '#9e9e9e';

/** The box as it appears on the mirrored video: x' = width - x - w. Pure. */
export function mirrorBox(box, width) {
  return { x: width - box.x - box.width, y: box.y, width: box.width, height: box.height };
}

/** Black or white, whichever reads better on a '#rrggbb' background (WCAG relative luminance). */
function textColourOn(hex) {
  const match = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (!match) return '#fff';
  const [r, g, b] = match.slice(1).map((part) => {
    const c = parseInt(part, 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  // Above ~0.179 black text has more contrast than white.
  return luminance > 0.179 ? '#000' : '#fff';
}

/**
 * Clear the canvas, then draw each label ({ box, personId, text }) as its mirrored box with the
 * text on a coloured tag just above it (or just inside the top edge when there is no room above).
 * `width` is the frame width; `colourFor(personId)` gives a person's colour. Unknown labels
 * (personId null) are grey.
 */
export function drawLabels(ctx, labels, { width, colourFor }) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Scale with the frame so labels look the same size whatever the camera resolution.
  const fontPx = Math.max(14, Math.round(width / 32));
  const pad = Math.round(fontPx / 4);
  ctx.lineWidth = Math.max(2, Math.round(width / 320));
  ctx.font = `600 ${fontPx}px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif`;
  ctx.textBaseline = 'top';

  for (const label of labels) {
    const colour = (label.personId === null ? null : colourFor(label.personId)) ?? UNKNOWN_COLOUR;
    const box = mirrorBox(label.box, width);
    ctx.strokeStyle = colour;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    const tagWidth = ctx.measureText(label.text).width + pad * 2;
    const tagHeight = fontPx + pad * 2;
    const tagX = Math.min(Math.max(box.x, 0), Math.max(0, width - tagWidth));
    const tagY = box.y >= tagHeight ? box.y - tagHeight : box.y;
    ctx.fillStyle = colour;
    ctx.fillRect(tagX, tagY, tagWidth, tagHeight);
    ctx.fillStyle = textColourOn(colour);
    ctx.fillText(label.text, tagX + pad, tagY + pad);
  }
}
