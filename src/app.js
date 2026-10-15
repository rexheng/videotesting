// The live tagging pipeline and app state. Each video frame: detect faces and bodies, follow them
// with the Tracker, recognise faces (throttled) and, for people whose face is hidden, shirts; then
// draw name labels over the mirrored video. While someone signs up (or re-scans their shirt), the
// same frame also feeds the sign-up state machine with the largest face and torso in view; live
// tagging keeps running. The app owns the list of people and saves it whenever it changes.
//
// Coordinates: MediaPipe landmarks are normalised (0-1) and are converted to pixels of the raw,
// unmirrored frame. Only drawLabels mirrors, to match the mirrored video.

import { MAX_EMBEDS_PER_FRAME, SHIRT_BLEND, LABEL_COLOURS } from './config.js';
import { embedFace } from './vision/faceEmbedder.js';
import { yawDegrees, bboxOfPoints } from './vision/geometry.js';
import { buildObservations, Tracker } from './recognition/tracker.js';
import { assignFaces, matchShirt } from './recognition/matcher.js';
import { torsoQuad, shirtHistogram, blendHistograms } from './recognition/shirt.js';
import { createEnrolment, enrolmentWantsEmbedding, stepEnrolment } from './enrol/enrolment.js';
import { newPerson, withNewShirt } from './storage.js';
import { drawLabels } from './ui/overlay.js';

const hasFace = (track) => track.observation.faceIndex !== null;
const isFaceMatch = (track) => track.status === 'match' && track.source === 'face';
const areaOf = (box) => box.width * box.height;

/** The item with the largest area (by areaOfItem), or null for an empty list. */
function largest(items, areaOfItem) {
  let best = null;
  let bestArea = -Infinity;
  for (const item of items) {
    const area = areaOfItem(item);
    if (area > bestArea) {
      best = item;
      bestArea = area;
    }
  }
  return best;
}

function createCanvas() {
  return typeof OffscreenCanvas === 'function' ? new OffscreenCanvas(1, 1) : document.createElement('canvas');
}

/**
 * Create the live tagging app.
 *   video     the playing <video> showing the camera (raw, unmirrored pixels)
 *   overlay   the <canvas> drawn over the video
 *   models    { face, pose, sface } from loadModels({ runningMode: 'VIDEO' })
 *   store     from createStore(); people are loaded from it now, and saved to it on every change
 *   onLabels  called with each frame's labels ({ trackId, box, personId, text }[])
 *   onDebug   optional; called each frame with { faceMs, poseMs, embedMs, embeds, faces, poses, yaw }
 *   onEnrolUpdate   called with the sign-up state when it starts and after each frame it runs
 *                   (the last one has step 'done'), or with null when it is cancelled
 *   onPeopleChange  called with (people, saved) after the people change and are saved; saved is
 *                   false when store.save failed
 */
export function createApp({
  video,
  overlay,
  models,
  store,
  onLabels = () => {},
  onDebug = null,
  onEnrolUpdate = () => {},
  onPeopleChange = () => {},
}) {
  const { face, pose, sface } = models;
  const overlayCtx = overlay.getContext('2d');
  const tracker = new Tracker();

  let people = [];
  let peopleById = new Map();
  // personId -> histogram refreshed from confident face matches (handles lighting changes).
  // In memory only: never saved automatically.
  const liveShirts = new Map();
  // The sign-up in progress: { state, personId } (personId is set for Re-scan shirt), or null.
  let enrolment = null;

  // A copy of the current frame, drawn at most once per frame and only when pixels are needed
  // (face embeddings and shirt histograms), so every read in a frame sees the same pixels.
  const frameCanvas = createCanvas();
  const frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true });

  let lastTimestamp = -Infinity;
  let busy = false;
  let running = false;
  let loopId = 0; // bumped on start/stop, so a callback from an old loop does nothing
  let cancelScheduled = null;
  const reportedErrors = new Set();

  setPeople(store.load());

  /** Log each distinct per-frame error once, so a recurring one does not flood the console. */
  function reportError(err) {
    const key = String(err?.message ?? err);
    if (reportedErrors.has(key)) return;
    reportedErrors.add(key);
    console.warn('Live tagging skipped a face or frame:', err);
  }

  /** MediaPipe needs strictly increasing timestamps (ms) for detectForVideo. */
  function nextTimestamp(now) {
    lastTimestamp = Math.max(lastTimestamp + 1, Math.round(now));
    return lastTimestamp;
  }

  function fitOverlay(width, height) {
    if (overlay.width !== width) overlay.width = width;
    if (overlay.height !== height) overlay.height = height;
  }

  /** Remove all labels: clear the overlay and report an empty label list. */
  function clearLabels() {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    onLabels([]);
  }

  function snapshotFrame(width, height) {
    if (frameCanvas.width !== width) frameCanvas.width = width;
    if (frameCanvas.height !== height) frameCanvas.height = height;
    frameCtx.drawImage(video, 0, 0, width, height);
  }

  /** Shirt { histogram, swatch } inside a torso quad of the snapshot, or null. Reads only the quad's box. */
  function shirtInQuad(quad, width, height) {
    const box = bboxOfPoints(quad);
    const x0 = Math.min(width, Math.max(0, Math.floor(box.x)));
    const y0 = Math.min(height, Math.max(0, Math.floor(box.y)));
    const x1 = Math.min(width, Math.max(0, Math.ceil(box.x + box.width)));
    const y1 = Math.min(height, Math.max(0, Math.ceil(box.y + box.height)));
    if (x1 <= x0 || y1 <= y0) return null;
    const { data } = frameCtx.getImageData(x0, y0, x1 - x0, y1 - y0);
    const localQuad = quad.map(([x, y]) => [x - x0, y - y0]);
    return shirtHistogram(data, x1 - x0, y1 - y0, localQuad);
  }

  /** Embedding of a face in the snapshot, or null if it cannot be embedded (e.g. degenerate landmarks). */
  async function embedInSnapshot(landmarks, width, height) {
    try {
      return await embedFace(sface, frameCanvas, landmarks, width, height);
    } catch (err) {
      reportError(err);
      return null;
    }
  }

  /** Tracks due an embedding, unnamed ones first, at most MAX_EMBEDS_PER_FRAME. */
  function tracksToEmbed(tracks, now) {
    // With nobody signed up every face is "Unknown" anyway, so skip the work.
    if (people.length === 0) return [];
    const due = tracks.filter((track) => tracker.needsEmbedding(track, now));
    const unnamed = due.filter((track) => track.personId === null);
    const named = due.filter((track) => track.personId !== null);
    return [...unnamed, ...named].slice(0, MAX_EMBEDS_PER_FRAME);
  }

  /** Steps 3-4: embed the chosen faces, pair all faces with people, apply the results. */
  async function recogniseFaces(frame, toEmbed) {
    if (toEmbed.length === 0) return;
    const { tracks, faceLandmarks, width, height, now, stats } = frame;
    const embeddingByTrack = new Map();
    for (const track of toEmbed) {
      const start = performance.now();
      const embedding = await embedInSnapshot(faceLandmarks[track.observation.faceIndex], width, height);
      stats.embedMs += performance.now() - start;
      if (embedding) {
        embeddingByTrack.set(track.id, embedding);
        stats.embeds += 1;
      }
    }
    // Faces not embedded this frame are null ('pending'), which the tracker ignores.
    const faced = tracks.filter(hasFace);
    const results = assignFaces(
      faced.map((track) => embeddingByTrack.get(track.id) ?? null),
      people,
    );
    faced.forEach((track, i) => tracker.applyFace(track.id, results[i], now));
  }

  /**
   * Step 5: an unnamed track whose face is hidden this frame but whose torso is visible is compared
   * with each person's live shirt (or stored shirt), leaving out people already on a track.
   */
  function recogniseShirts(frame) {
    const { tracks, torsos, width, height } = frame;
    const taken = new Set(tracks.filter((track) => track.personId !== null).map((track) => track.personId));
    for (const track of tracks) {
      const quad = torsos.get(track.id);
      if (track.personId !== null || hasFace(track) || !quad) continue;
      const candidates = people
        .filter((person) => !taken.has(person.id))
        .map((person) => ({ personId: person.id, histogram: liveShirts.get(person.id) ?? person.shirtHistogram }));
      if (candidates.length === 0) return; // everyone is already on a track
      const shirt = shirtInQuad(quad, width, height);
      if (!shirt) continue;
      const result = matchShirt(shirt.histogram, candidates);
      tracker.applyShirt(track.id, result);
      if (result.personId !== null) taken.add(result.personId);
    }
  }

  /** Step 6: nudge the live shirt of each confidently face-matched person towards this frame's. */
  function refreshLiveShirts(frame) {
    const { tracks, torsos, width, height } = frame;
    const holders = new Map();
    for (const track of tracks) {
      if (track.personId !== null) holders.set(track.personId, (holders.get(track.personId) ?? 0) + 1);
    }
    for (const track of tracks) {
      const quad = torsos.get(track.id);
      if (!quad || !hasFace(track) || !isFaceMatch(track)) continue;
      // Two tracks claiming one person: labels() keeps only one, so do not guess which shirt is theirs.
      if (holders.get(track.personId) !== 1) continue;
      const person = peopleById.get(track.personId);
      if (!person) continue;
      const shirt = shirtInQuad(quad, width, height);
      if (!shirt) continue;
      const live = liveShirts.get(person.id) ?? person.shirtHistogram;
      liveShirts.set(person.id, blendHistograms(live, shirt.histogram, SHIRT_BLEND));
    }
  }

  /**
   * Sign-up input for this frame, chosen before anything awaits: the largest face (its yaw, and
   * whether the sign-up wants it embedded) and the largest torso. Null when nobody is signing up.
   */
  function planSignUp(frame) {
    if (!enrolment) return null;
    const { faceLandmarks, torsos, width, height, now } = frame;
    const landmarks = largest(faceLandmarks, (points) =>
      areaOf(bboxOfPoints(points.map((p) => [p.x * width, p.y * height]))),
    );
    let face = null;
    if (landmarks) {
      const yaw = yawDegrees(landmarks);
      face = { landmarks, yaw, embed: enrolmentWantsEmbedding(enrolment.state, yaw, now) };
    }
    let quad = null;
    if (enrolment.state.step === 'shirt') {
      const quads = [...torsos.values()].filter(Boolean);
      quad = largest(quads, (points) => areaOf(bboxOfPoints(points)));
    }
    return { session: enrolment, face, quad };
  }

  /** Step 8: feed the sign-up this frame's face and shirt; when it is done, save the person. */
  async function stepSignUp(frame, plan) {
    const { width, height, now, stats } = frame;
    let embedding = null;
    if (plan.face?.embed) {
      const start = performance.now();
      embedding = await embedInSnapshot(plan.face.landmarks, width, height);
      stats.embedMs += performance.now() - start;
      if (embedding) stats.embeds += 1;
    }
    // Cancelled, or a new sign-up started, while the face was being embedded.
    if (enrolment !== plan.session) return;
    // The person whose shirt is being re-scanned has gone (deletePerson cancels this already).
    if (plan.session.personId !== null && !peopleById.has(plan.session.personId)) {
      cancelEnrolment();
      return;
    }

    const shirt = plan.quad ? shirtInQuad(plan.quad, width, height) : null;
    const face = plan.face ? { yaw: plan.face.yaw, embedding } : null;
    const state = stepEnrolment(plan.session.state, { face, shirt }, now);
    plan.session.state = state;
    if (state.step === 'done') {
      enrolment = null;
      finishSignUp(plan.session.personId, state);
    }
    onEnrolUpdate(state);
  }

  /** Add the new person, or give the re-scanned person their new shirt, and save everyone. */
  function finishSignUp(personId, { name, result }) {
    if (personId === null) {
      savePeople([...people, newPerson({ name, ...result }, people, Date.now())]);
      return;
    }
    // A new person object (and histogram), so setPeople drops their old live shirt.
    savePeople(people.map((person) => (person.id === personId ? withNewShirt(person, result) : person)));
  }

  const colourFor = (personId) => peopleById.get(personId)?.colour ?? LABEL_COLOURS[0];

  /**
   * Run the pipeline on the video's current frame. `now` is a performance.now()-style time in ms.
   * Resolves with the labels drawn, or null when the frame was skipped (video not ready, or the
   * previous frame is still being processed).
   */
  async function processFrame(now = performance.now()) {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (busy || !width || !height || video.readyState < 2) return null;
    busy = true;
    try {
      fitOverlay(width, height);
      const stats = { faceMs: 0, poseMs: 0, embedMs: 0, embeds: 0 };

      // 1. Detect faces and bodies.
      const timestamp = nextTimestamp(now);
      let start = performance.now();
      const faceLandmarks = face.detectForVideo(video, timestamp).faceLandmarks ?? [];
      stats.faceMs = performance.now() - start;
      start = performance.now();
      const poseLandmarks = pose.detectForVideo(video, timestamp).landmarks ?? [];
      stats.poseMs = performance.now() - start;

      // 2. Turn them into observations and follow them from frame to frame.
      const observations = buildObservations(
        faceLandmarks.map((landmarks) => ({ landmarks })),
        poseLandmarks.map((landmarks) => ({ landmarks })),
        width,
        height,
      );
      const tracks = tracker.associate(observations, now);
      const torsos = new Map();
      for (const track of tracks) {
        const { poseIndex } = track.observation;
        torsos.set(track.id, poseIndex === null ? null : torsoQuad(poseLandmarks[poseIndex], width, height));
      }
      const frame = { now, width, height, faceLandmarks, poseLandmarks, tracks, torsos, stats };
      const signUp = planSignUp(frame);

      // Copy the frame now, before anything awaits, if any step below will read its pixels.
      const toEmbed = tracksToEmbed(tracks, now);
      const needsPixels =
        toEmbed.length > 0 ||
        Boolean(signUp && (signUp.face?.embed || signUp.quad)) ||
        tracks.some(
          (track) =>
            torsos.get(track.id) &&
            (hasFace(track) ? isFaceMatch(track) : track.personId === null && people.length > 0),
        );
      if (needsPixels) snapshotFrame(width, height);

      // 3-4. Faces.  5. Shirts for hidden faces.  6. Live shirt refresh.
      await recogniseFaces(frame, toEmbed);
      recogniseShirts(frame);
      refreshLiveShirts(frame);

      // 7. Labels.
      const labels = tracker.labels(peopleById);
      drawLabels(overlayCtx, labels, { width, colourFor });
      onLabels(labels);

      // 8. Sign-up, while someone is signing up.
      if (signUp) await stepSignUp(frame, signUp);

      onDebug?.({
        ...stats,
        faces: faceLandmarks.length,
        poses: poseLandmarks.length,
        yaw: faceLandmarks.length ? yawDegrees(faceLandmarks[0]) : null,
      });
      return labels;
    } finally {
      busy = false;
    }
  }

  function scheduleFrame(id) {
    const onFrame = (now) => runFrame(now, id);
    if (typeof video.requestVideoFrameCallback === 'function') {
      const handle = video.requestVideoFrameCallback(onFrame);
      cancelScheduled = () => video.cancelVideoFrameCallback(handle);
    } else {
      const handle = requestAnimationFrame(onFrame);
      cancelScheduled = () => cancelAnimationFrame(handle);
    }
  }

  async function runFrame(now, id) {
    if (id !== loopId) return;
    cancelScheduled = null;
    try {
      await processFrame(now);
    } catch (err) {
      // One bad frame must not stop the loop.
      reportError(err);
    }
    if (id === loopId) scheduleFrame(id);
    else if (!running) clearLabels(); // stopped while this frame was in flight
  }

  /** Start processing every new video frame. */
  function start() {
    if (running) return;
    running = true;
    loopId += 1;
    scheduleFrame(loopId);
  }

  /** Stop processing frames and clear the labels. */
  function stop() {
    if (!running) return;
    running = false;
    loopId += 1;
    cancelScheduled?.();
    cancelScheduled = null;
    clearLabels();
  }

  /** The signed-up people (a copy of the list). */
  function getPeople() {
    return [...people];
  }

  /**
   * Replace the signed-up people (does not save). Live shirts are dropped for people who are gone
   * or whose stored shirt changed (e.g. re-scanned), so the new shirt is used straight away.
   */
  function setPeople(next) {
    const previous = peopleById;
    people = [...next];
    peopleById = new Map(people.map((person) => [person.id, person]));
    for (const personId of liveShirts.keys()) {
      const person = peopleById.get(personId);
      if (!person || person.shirtHistogram !== previous.get(personId)?.shirtHistogram) {
        liveShirts.delete(personId);
      }
    }
  }

  /** Replace the people, save them and report the change. Returns false when the save failed. */
  function savePeople(next) {
    setPeople(next);
    const saved = store.save(people);
    onPeopleChange(getPeople(), saved);
    return saved;
  }

  /**
   * Start signing up `name`, replacing any sign-up in progress. With `personId` (an existing
   * person) only the shirt is scanned: Re-scan shirt. `now` is a performance.now()-style time.
   * Reports the first state through onEnrolUpdate and returns it.
   */
  function startEnrolment(name, { personId = null, now = performance.now() } = {}) {
    if (personId !== null && !peopleById.has(personId)) throw new Error(`No person with id ${personId}`);
    const state = createEnrolment(name, now, personId === null ? {} : { startAt: 'shirt' });
    enrolment = { state, personId };
    onEnrolUpdate(state);
    return state;
  }

  /** Discard the sign-up in progress, if any (reports null through onEnrolUpdate). */
  function cancelEnrolment() {
    if (!enrolment) return;
    enrolment = null;
    onEnrolUpdate(null);
  }

  /** Remove a person and save. A re-scan of that person's shirt is cancelled first. */
  function deletePerson(personId) {
    if (!peopleById.has(personId)) return;
    if (enrolment?.personId === personId) cancelEnrolment();
    savePeople(people.filter((person) => person.id !== personId));
  }

  return {
    start,
    stop,
    getPeople,
    setPeople,
    processFrame,
    startEnrolment,
    cancelEnrolment,
    deletePerson,
  };
}
