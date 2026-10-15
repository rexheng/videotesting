// The ?debug=1 panel: frame rate, per-model timings and the first face's head turn, for checking
// speed on phones. Per-frame stats are averaged and the text is refreshed at most every intervalMs.

const fixed = (value, digits = 1) => (value === null || value === undefined ? '–' : value.toFixed(digits));

/**
 * Unhide `pre` and return { update(stats) }, where stats is the app's per-frame onDebug payload:
 * { faceMs, poseMs, embedMs, embeds, faces, poses, yaw }.
 */
export function createDebugPanel(pre, { intervalMs = 500 } = {}) {
  pre.hidden = false;
  pre.textContent = 'FPS –\nStarting…';

  let windowStart = null; // set by the first frame, so model loading does not count against FPS
  let frames = 0;
  let faceMs = 0;
  let poseMs = 0;
  let embedMs = 0;
  let embeds = 0;
  let lastEmbedMs = null; // per embedding; kept across windows, since most frames embed nothing
  let latest = null;

  function update(stats) {
    const now = performance.now();
    if (windowStart === null) {
      windowStart = now;
      return;
    }
    frames += 1;
    faceMs += stats.faceMs;
    poseMs += stats.poseMs;
    embedMs += stats.embedMs;
    embeds += stats.embeds;
    latest = stats;

    const elapsed = now - windowStart;
    if (elapsed < intervalMs) return;

    if (embeds > 0) lastEmbedMs = embedMs / embeds;
    pre.textContent = [
      `FPS ${fixed((frames * 1000) / elapsed)}`,
      `face ${fixed(faceMs / frames)} ms · pose ${fixed(poseMs / frames)} ms · embed ${fixed(lastEmbedMs)} ms`,
      `faces ${latest.faces} · bodies ${latest.poses}`,
      `yaw ${latest.yaw === null ? '–' : `${fixed(latest.yaw)}°`}`,
    ].join('\n');

    windowStart = now;
    frames = 0;
    faceMs = 0;
    poseMs = 0;
    embedMs = 0;
    embeds = 0;
  }

  return { update };
}
