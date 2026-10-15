// Observations and identity tracking. Each frame's detected faces and poses become
// "observations" (one per person-ish thing); the Tracker follows them from frame to frame by
// box overlap so a name stays on a person when they turn away, applies face and shirt match
// results, keeps one name from showing on two people, and produces the label texts to draw.
// Pure module: no DOM, no CDN imports. Boxes are { x, y, width, height } in pixels of the raw,
// unmirrored frame; landmarks arrive normalised (0-1) and are converted with width/height.
//
// Per-frame order (driven by the live pipeline):
//   associate -> embed faces where needsEmbedding -> applyFace -> shirt-match face-hidden
//   tracks -> applyShirt -> labels.

import { TRACK_IOU, TRACK_TTL_MS, EMBED_RECHECK_MS } from '../config.js';
import { bboxOfPoints, iou, containsPoint } from '../vision/geometry.js';

/** A pose landmark counts toward the body box only at or above this visibility. */
const BODY_VISIBILITY = 0.5;
/** MediaPipe pose landmark 0 is the nose. */
const NOSE = 0;

const toPixels = (landmarks, width, height) => landmarks.map((p) => [p.x * width, p.y * height]);

function unionBox(a, b) {
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  return {
    x,
    y,
    width: Math.max(a.x + a.width, b.x + b.width) - x,
    height: Math.max(a.y + a.height, b.y + b.height) - y,
  };
}

/** Body box of a pose from its visible landmarks, or null when none is visible enough. */
function bodyBoxOf(pose, width, height) {
  const visible = pose.landmarks.filter((p) => p.visibility >= BODY_VISIBILITY);
  return visible.length ? bboxOfPoints(toPixels(visible, width, height)) : null;
}

/**
 * Turn one frame's faces and poses into observations:
 *   { box, faceBox: box|null, faceIndex: number|null, poseIndex: number|null }
 * Face i links to the first unclaimed pose whose nose lies inside the face box; a linked
 * observation's box is the union of body and face. Unlinked faces become face-only observations,
 * unlinked poses pose-only ones. Faces come first (in face order), then pose-only (in pose order).
 * A pose with no landmark visible enough has no body box and is ignored altogether.
 */
export function buildObservations(faces, poses, width, height) {
  const bodies = poses.map((pose) => bodyBoxOf(pose, width, height));
  const claimed = new Set();
  const observations = [];

  faces.forEach((face, faceIndex) => {
    const faceBox = bboxOfPoints(toPixels(face.landmarks, width, height));
    let poseIndex = null;
    for (let i = 0; i < poses.length; i++) {
      if (claimed.has(i) || bodies[i] === null) continue;
      const nose = poses[i].landmarks[NOSE];
      if (nose && containsPoint(faceBox, [nose.x * width, nose.y * height])) {
        poseIndex = i;
        claimed.add(i);
        break;
      }
    }
    const box = poseIndex === null ? faceBox : unionBox(bodies[poseIndex], faceBox);
    observations.push({ box, faceBox, faceIndex, poseIndex });
  });

  bodies.forEach((body, poseIndex) => {
    if (body === null || claimed.has(poseIndex)) return;
    observations.push({ box: body, faceBox: null, faceIndex: null, poseIndex });
  });

  return observations;
}

const hasFace = (track) => track.observation.faceIndex !== null;

/** A track that currently has a name it earned from a face and can see that face. */
const faceConfirmed = (track) => track.source === 'face' && hasFace(track);

function clearIdentity(track, status) {
  track.personId = null;
  track.source = null;
  track.status = status;
  track.score = 0;
}

export class Tracker {
  constructor({ ttlMs = TRACK_TTL_MS, iouThreshold = TRACK_IOU } = {}) {
    this.ttlMs = ttlMs;
    this.iouThreshold = iouThreshold;
    this.tracks = [];
    this.current = []; // tracks seen in the latest associate() call
    this.nextId = 1;
  }

  /**
   * Match this frame's observations to existing tracks (greedy, highest IoU first, at or above
   * the threshold). Unmatched observations start new tracks; tracks unseen for more than ttlMs
   * are dropped first. Returns the tracks seen this frame, in observation order.
   */
  associate(observations, now) {
    this.tracks = this.tracks.filter((track) => now - track.lastSeen <= this.ttlMs);

    const pairs = [];
    observations.forEach((observation, obsIndex) => {
      for (const track of this.tracks) {
        const overlap = iou(track.box, observation.box);
        if (overlap >= this.iouThreshold) pairs.push({ overlap, obsIndex, track });
      }
    });
    pairs.sort((p, q) => q.overlap - p.overlap || p.track.id - q.track.id || p.obsIndex - q.obsIndex);

    const trackForObs = new Array(observations.length).fill(null);
    const usedTracks = new Set();
    for (const { obsIndex, track } of pairs) {
      if (trackForObs[obsIndex] || usedTracks.has(track)) continue;
      trackForObs[obsIndex] = track;
      usedTracks.add(track);
    }

    this.current = observations.map((observation, obsIndex) => {
      let track = trackForObs[obsIndex];
      if (!track) {
        track = {
          id: this.nextId++,
          box: observation.box,
          observation,
          personId: null,
          source: null,
          status: 'none',
          score: 0,
          lastEmbedAt: -Infinity,
          lastSeen: now,
        };
        this.tracks.push(track);
      }
      track.box = observation.box;
      track.observation = observation;
      track.lastSeen = now;
      return track;
    });
    return this.current;
  }

  /**
   * True when the track has a face this frame and its name is not a recent, confirmed face match:
   * not yet 'match', named by shirt rather than face, or last embedded EMBED_RECHECK_MS ago.
   */
  needsEmbedding(track, now) {
    if (!hasFace(track)) return false;
    return (
      track.status !== 'match' ||
      track.source !== 'face' ||
      now - track.lastEmbedAt >= EMBED_RECHECK_MS
    );
  }

  /**
   * Apply a face-match result ({ personId, score, status } from assignFaces) to a track.
   *  - 'pending': no embedding was computed; ignored entirely (the embed clock is not touched).
   *  - anything else records lastEmbedAt = now, then:
   *  - 'match': the track is that person, sourced from the face. A face match always overrides a
   *    shirt match.
   *  - 'unsure': applied only if the track has no confident ('match') identity.
   *  - 'unknown': a face was visible and embedded and matched nobody, so it is not the tagged
   *    person (people may have swapped places): any name, from face or shirt, is cleared.
   */
  applyFace(trackId, { personId, score, status }, now) {
    const track = this.#find(trackId);
    if (!track || status === 'pending') return;
    track.lastEmbedAt = now;

    if (status === 'match') {
      track.personId = personId;
      track.score = score;
      track.source = 'face';
      track.status = 'match';
    } else if (status === 'unsure') {
      if (track.personId !== null && track.status === 'match') return;
      track.personId = personId;
      track.score = score;
      track.source = 'face';
      track.status = 'unsure';
    } else if (status === 'unknown') {
      clearIdentity(track, 'unknown');
    }
  }

  /**
   * Apply a shirt-match result ({ personId, score }). Only for a track that has no name yet and
   * no face this frame: a visible face is judged by the face alone.
   */
  applyShirt(trackId, { personId, score }) {
    const track = this.#find(trackId);
    if (!track || personId === null || personId === undefined) return;
    if (track.personId !== null || hasFace(track)) return;
    track.personId = personId;
    track.score = score;
    track.source = 'shirt';
    track.status = 'match';
  }

  /**
   * Labels for the tracks seen this frame: { trackId, box, personId, text }.
   * First, identities are made consistent: a person no longer in `peopleById` is dropped, and when
   * several tracks share a person only one keeps them (a face-sourced track with a face this frame
   * wins, otherwise the highest score; ties go to the older track). Losers lose their identity
   * (personId, source -> null, status -> 'none'). Then:
   *   face-sourced match with a face this frame -> "Name"
   *   unsure                                    -> "Name?"
   *   any other track with a person             -> "Name 👕"
   *   a face this frame, no person              -> "Unknown"
   *   otherwise                                 -> no label
   */
  labels(peopleById) {
    this.#resolveConflicts(peopleById);

    const labels = [];
    for (const track of this.current) {
      const name = track.personId === null ? undefined : peopleById.get(track.personId)?.name;
      let text = null;
      if (name !== undefined) {
        if (track.source === 'face' && track.status === 'match' && hasFace(track)) text = name;
        else if (track.status === 'unsure') text = `${name}?`;
        else text = `${name} 👕`;
      } else if (hasFace(track)) {
        text = 'Unknown';
      }
      if (text !== null) {
        labels.push({ trackId: track.id, box: track.box, personId: track.personId, text });
      }
    }
    return labels;
  }

  #find(trackId) {
    return this.tracks.find((track) => track.id === trackId);
  }

  #resolveConflicts(peopleById) {
    const holders = new Map(); // personId -> tracks seen this frame carrying that person
    for (const track of this.current) {
      if (track.personId === null) continue;
      if (!peopleById.has(track.personId)) {
        clearIdentity(track, 'none');
        continue;
      }
      if (!holders.has(track.personId)) holders.set(track.personId, []);
      holders.get(track.personId).push(track);
    }

    for (const claimants of holders.values()) {
      if (claimants.length < 2) continue;
      const [winner] = [...claimants].sort(
        (p, q) =>
          Number(faceConfirmed(q)) - Number(faceConfirmed(p)) || q.score - p.score || p.id - q.id,
      );
      for (const track of claimants) {
        if (track !== winner) clearIdentity(track, 'none');
      }
    }
  }
}
