import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildObservations, Tracker } from '../../src/recognition/tracker.js';
import { TRACK_IOU, TRACK_TTL_MS, EMBED_RECHECK_MS } from '../../src/config.js';

const people = new Map([
  ['alice', { name: 'Alice' }],
  ['bob', { name: 'Bob' }],
]);

const BOX = { x: 100, y: 100, width: 100, height: 200 };
const FAR = { x: 600, y: 100, width: 100, height: 200 };

/** A hand-made observation. `withFace` gives it a face this frame (faceIndex 0). */
function observation(box, withFace = false) {
  return {
    box,
    faceBox: withFace ? { x: box.x + 20, y: box.y, width: 60, height: 60 } : null,
    faceIndex: withFace ? 0 : null,
    poseIndex: withFace ? null : 0,
  };
}

/** One tracker with one track, seen at `now` (default 0). */
function singleTrack({ withFace = true, now = 0, box = BOX } = {}) {
  const tracker = new Tracker();
  const [track] = tracker.associate([observation(box, withFace)], now);
  return { tracker, track };
}

const textsOf = (labels) => labels.map((label) => label.text);

test('config thresholds are the values the brief assumes', () => {
  assert.equal(TRACK_IOU, 0.3);
  assert.equal(TRACK_TTL_MS, 1000);
  assert.equal(EMBED_RECHECK_MS, 1000);
});

// ---------------------------------------------------------------- associate

test('associate: a new track starts with no identity', () => {
  const { track } = singleTrack({ withFace: true, now: 50 });
  assert.equal(typeof track.id, 'number');
  assert.deepEqual(track.box, BOX);
  assert.equal(track.personId, null);
  assert.equal(track.source, null);
  assert.equal(track.status, 'none');
  assert.equal(track.score, 0);
  assert.equal(track.lastEmbedAt, -Infinity);
  assert.equal(track.lastSeen, 50);
  assert.equal(track.observation.faceIndex, 0);
});

test('associate: the same box next frame keeps the same id', () => {
  const tracker = new Tracker();
  const [first] = tracker.associate([observation(BOX, true)], 0);
  const [second] = tracker.associate([observation(BOX, true)], 100);
  assert.equal(second.id, first.id);
  assert.equal(second.lastSeen, 100);
});

test('associate: a track follows a small move and takes the new box and observation', () => {
  const tracker = new Tracker();
  const [first] = tracker.associate([observation(BOX, true)], 0);
  const moved = { ...BOX, x: BOX.x + 10 };
  const [second] = tracker.associate([observation(moved, false)], 100);
  assert.equal(second.id, first.id);
  assert.deepEqual(second.box, moved);
  assert.equal(second.observation.faceIndex, null);
});

test('associate: a far box gets a new id and both tracks are alive', () => {
  const tracker = new Tracker();
  const [first] = tracker.associate([observation(BOX)], 0);
  const seen = tracker.associate([observation(BOX), observation(FAR)], 100);
  assert.equal(seen.length, 2);
  assert.equal(seen[0].id, first.id);
  assert.notEqual(seen[1].id, first.id);
});

test('associate: only tracks seen this frame are returned', () => {
  const tracker = new Tracker();
  tracker.associate([observation(BOX), observation(FAR)], 0);
  const seen = tracker.associate([observation(BOX)], 100);
  assert.equal(seen.length, 1);
});

test('associate: a track unseen for more than ttlMs is gone, but not at exactly ttlMs', () => {
  const tracker = new Tracker();
  const [first] = tracker.associate([observation(BOX)], 0);
  const [still] = tracker.associate([observation(BOX)], TRACK_TTL_MS);
  assert.equal(still.id, first.id);

  const [gone] = tracker.associate([observation(BOX)], TRACK_TTL_MS + TRACK_TTL_MS + 1);
  assert.notEqual(gone.id, first.id);
});

test('associate: a track unseen at last + 1001 is gone', () => {
  const tracker = new Tracker();
  const [first] = tracker.associate([observation(BOX)], 0);
  tracker.associate([], 500);
  const [fresh] = tracker.associate([observation(BOX)], 1001);
  assert.notEqual(fresh.id, first.id);
});

test('associate: a track unseen briefly is picked up again with its identity', () => {
  const tracker = new Tracker();
  const [first] = tracker.associate([observation(BOX, true)], 0);
  tracker.applyFace(first.id, { personId: 'alice', score: 0.8, status: 'match' }, 0);
  tracker.associate([], 400); // person occluded for a moment
  const [back] = tracker.associate([observation(BOX, false)], 800);
  assert.equal(back.id, first.id);
  assert.equal(back.personId, 'alice');
});

test('associate: the highest overlap wins when two tracks compete for one observation', () => {
  const tracker = new Tracker();
  const left = { x: 0, y: 0, width: 100, height: 100 };
  const right = { x: 60, y: 0, width: 100, height: 100 };
  // Seen together once so both tracks exist (their IoU is 0.25, below the threshold).
  const [leftTrack, rightTrack] = tracker.associate([observation(left), observation(right)], 0);
  assert.notEqual(leftTrack.id, rightTrack.id);

  // One observation overlapping both, but clearly more the right one.
  const candidate = { x: 45, y: 0, width: 100, height: 100 };
  const [matched] = tracker.associate([observation(candidate)], 100);
  assert.equal(matched.id, rightTrack.id);
});

test('associate: observations below the IoU threshold do not match', () => {
  const tracker = new Tracker();
  const [first] = tracker.associate([observation({ x: 0, y: 0, width: 100, height: 100 })], 0);
  // Overlap 40x100 = 4000 over union 16000 = 0.25 < 0.3.
  const [second] = tracker.associate([observation({ x: 60, y: 0, width: 100, height: 100 })], 100);
  assert.notEqual(second.id, first.id);
});

test('associate: the threshold is configurable', () => {
  const strict = new Tracker({ iouThreshold: 0.9 });
  const [first] = strict.associate([observation(BOX)], 0);
  const [second] = strict.associate([observation({ ...BOX, x: BOX.x + 20 })], 100);
  assert.notEqual(second.id, first.id);

  const brief = new Tracker({ ttlMs: 50 });
  const [one] = brief.associate([observation(BOX)], 0);
  const [two] = brief.associate([observation(BOX)], 60);
  assert.notEqual(two.id, one.id);
});

// ---------------------------------------------------------------- labels: follow and shirt

test('follow: a face match gives the name, then the same body without a face gives the shirt label', () => {
  const tracker = new Tracker();
  const [track] = tracker.associate([observation(BOX, true)], 0);
  tracker.applyFace(track.id, { personId: 'alice', score: 0.8, status: 'match' }, 0);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Alice']);

  tracker.associate([observation(BOX, false)], 100);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Alice 👕']);
});

test('labels: carry the track id, box and person id', () => {
  const { tracker, track } = singleTrack();
  tracker.applyFace(track.id, { personId: 'alice', score: 0.8, status: 'match' }, 0);
  const [label] = tracker.labels(people);
  assert.deepEqual(label, { trackId: track.id, box: BOX, personId: 'alice', text: 'Alice' });
});

test('shirt: a new track without a face plus a shirt match gives the shirt label', () => {
  const { tracker, track } = singleTrack({ withFace: false });
  tracker.applyShirt(track.id, { personId: 'bob', score: 0.9 });
  assert.equal(track.source, 'shirt');
  assert.equal(track.status, 'match');
  assert.equal(track.score, 0.9);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Bob 👕']);
});

test('shirt: ignores a result with no person', () => {
  const { tracker, track } = singleTrack({ withFace: false });
  tracker.applyShirt(track.id, { personId: null, score: 0 });
  assert.equal(track.personId, null);
  assert.equal(track.status, 'none');
  assert.deepEqual(tracker.labels(people), []);
});

test('shirt: does not replace a person the track already has', () => {
  const { tracker, track } = singleTrack({ withFace: false });
  tracker.applyShirt(track.id, { personId: 'bob', score: 0.9 });
  tracker.applyShirt(track.id, { personId: 'alice', score: 0.95 });
  assert.equal(track.personId, 'bob');
  assert.equal(track.score, 0.9);
});

test('shirt: an unknown id is ignored', () => {
  const { tracker } = singleTrack({ withFace: false });
  assert.doesNotThrow(() => tracker.applyShirt(9999, { personId: 'bob', score: 0.9 }));
  assert.doesNotThrow(() => tracker.applyFace(9999, { personId: 'bob', score: 0.9, status: 'match' }, 0));
});

test('face overrides shirt: a Bob-by-shirt track becomes Alice on a face match', () => {
  const tracker = new Tracker();
  const [hidden] = tracker.associate([observation(BOX, false)], 0);
  tracker.applyShirt(hidden.id, { personId: 'bob', score: 0.9 });
  assert.deepEqual(textsOf(tracker.labels(people)), ['Bob 👕']);

  const [turned] = tracker.associate([observation(BOX, true)], 100);
  assert.equal(turned.id, hidden.id);
  tracker.applyFace(turned.id, { personId: 'alice', score: 0.7, status: 'match' }, 100);
  assert.equal(turned.personId, 'alice');
  assert.equal(turned.source, 'face');
  assert.deepEqual(textsOf(tracker.labels(people)), ['Alice']);
});

// ---------------------------------------------------------------- labels: one name per person

test('one name per person: a face track beats a shirt track carrying the same person', () => {
  const tracker = new Tracker();
  const [faceTrack, shirtTrack] = tracker.associate(
    [observation(BOX, true), observation(FAR, false)],
    0,
  );
  tracker.applyFace(faceTrack.id, { personId: 'alice', score: 0.5, status: 'match' }, 0);
  tracker.applyShirt(shirtTrack.id, { personId: 'alice', score: 0.99 });

  const labels = tracker.labels(people);
  assert.equal(labels.length, 1);
  assert.equal(labels[0].trackId, faceTrack.id);
  assert.equal(labels[0].text, 'Alice');
  // The loser has lost its identity, not merely its label.
  assert.equal(shirtTrack.personId, null);
  assert.equal(shirtTrack.source, null);
  assert.equal(shirtTrack.status, 'none');
});

test('one name per person: the face track wins even when listed second', () => {
  const tracker = new Tracker();
  const [shirtTrack, faceTrack] = tracker.associate(
    [observation(FAR, false), observation(BOX, true)],
    0,
  );
  tracker.applyShirt(shirtTrack.id, { personId: 'alice', score: 0.99 });
  tracker.applyFace(faceTrack.id, { personId: 'alice', score: 0.4, status: 'match' }, 0);
  const labels = tracker.labels(people);
  assert.deepEqual(labels.map((label) => label.trackId), [faceTrack.id]);
});

test('one name per person: a face-sourced track whose face is hidden loses to one showing its face', () => {
  const tracker = new Tracker();
  const [first, second] = tracker.associate([observation(BOX, true), observation(FAR, true)], 0);
  tracker.applyFace(first.id, { personId: 'alice', score: 0.9, status: 'match' }, 0);
  tracker.applyFace(second.id, { personId: 'alice', score: 0.5, status: 'match' }, 0);
  // Next frame the first person turns away (still face-sourced Alice, but no face now).
  tracker.associate([observation(BOX, false), observation(FAR, true)], 100);
  const labels = tracker.labels(people);
  assert.deepEqual(labels.map((label) => [label.trackId, label.text]), [[second.id, 'Alice']]);
});

test('one name per person: with no face-sourced contender the higher score wins', () => {
  const tracker = new Tracker();
  const [low, high] = tracker.associate([observation(BOX, false), observation(FAR, false)], 0);
  tracker.applyShirt(low.id, { personId: 'bob', score: 0.8 });
  tracker.applyShirt(high.id, { personId: 'bob', score: 0.95 });
  const labels = tracker.labels(people);
  assert.deepEqual(labels.map((label) => [label.trackId, label.text]), [[high.id, 'Bob 👕']]);
});

test('one name per person: different people on different tracks are all labelled', () => {
  const tracker = new Tracker();
  const [first, second] = tracker.associate([observation(BOX, true), observation(FAR, true)], 0);
  tracker.applyFace(first.id, { personId: 'alice', score: 0.8, status: 'match' }, 0);
  tracker.applyFace(second.id, { personId: 'bob', score: 0.8, status: 'match' }, 0);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Alice', 'Bob']);
});

test('labels: a person missing from the people map is not labelled and the track is freed', () => {
  const { tracker, track } = singleTrack();
  tracker.applyFace(track.id, { personId: 'ghost', score: 0.8, status: 'match' }, 0);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Unknown']);
  assert.equal(track.personId, null);
});

// ---------------------------------------------------------------- needsEmbedding

test('needsEmbedding: false just after a face match, true once the recheck interval has passed', () => {
  const tracker = new Tracker();
  const [track] = tracker.associate([observation(BOX, true)], 0);
  assert.equal(tracker.needsEmbedding(track, 0), true); // brand new track
  tracker.applyFace(track.id, { personId: 'alice', score: 0.8, status: 'match' }, 0);
  assert.equal(tracker.needsEmbedding(track, 0), false);
  assert.equal(tracker.needsEmbedding(track, 500), false);
  assert.equal(tracker.needsEmbedding(track, EMBED_RECHECK_MS - 1), false);
  assert.equal(tracker.needsEmbedding(track, EMBED_RECHECK_MS), true);
});

test('needsEmbedding: false when the track has no face this frame', () => {
  const { tracker, track } = singleTrack({ withFace: false });
  assert.equal(tracker.needsEmbedding(track, 0), false);
  assert.equal(tracker.needsEmbedding(track, 5000), false);
});

test('needsEmbedding: true straight away unless the track is a confident face match', () => {
  const { tracker, track } = singleTrack({ withFace: true });
  tracker.applyFace(track.id, { personId: 'alice', score: 0.33, status: 'unsure' }, 0);
  assert.equal(tracker.needsEmbedding(track, 1), true);

  const other = new Tracker();
  const [unknown] = other.associate([observation(BOX, true)], 0);
  other.applyFace(unknown.id, { personId: null, score: 0.1, status: 'unknown' }, 0);
  assert.equal(other.needsEmbedding(unknown, 1), true);

  const shirted = new Tracker();
  const [hidden] = shirted.associate([observation(BOX, false)], 0);
  shirted.applyShirt(hidden.id, { personId: 'bob', score: 0.9 });
  const [shown] = shirted.associate([observation(BOX, true)], 10);
  assert.equal(shirted.needsEmbedding(shown, 10), true); // a shirt-sourced name is never "face confirmed"
});

// ---------------------------------------------------------------- applyFace rules

test('an unknown face gives Unknown', () => {
  const { tracker, track } = singleTrack();
  tracker.applyFace(track.id, { personId: null, score: 0.1, status: 'unknown' }, 0);
  assert.equal(track.status, 'unknown');
  assert.equal(track.lastEmbedAt, 0);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Unknown']);
});

test('an unsure face gives the name with a question mark', () => {
  const { tracker, track } = singleTrack();
  tracker.applyFace(track.id, { personId: 'alice', score: 0.33, status: 'unsure' }, 40);
  assert.equal(track.personId, 'alice');
  assert.equal(track.source, 'face');
  assert.equal(track.status, 'unsure');
  assert.equal(track.score, 0.33);
  assert.equal(track.lastEmbedAt, 40);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Alice?']);
});

test('no face and no identity gives no label', () => {
  const { tracker } = singleTrack({ withFace: false });
  assert.deepEqual(tracker.labels(people), []);
});

test('a face track whose first embedding is still pending reads Unknown', () => {
  const { tracker, track } = singleTrack({ withFace: true });
  tracker.applyFace(track.id, { personId: null, score: -1, status: 'pending' }, 0);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Unknown']);
});

test('pending is ignored: identity and embed time are untouched', () => {
  const { tracker, track } = singleTrack();
  tracker.applyFace(track.id, { personId: 'alice', score: 0.8, status: 'match' }, 0);
  tracker.applyFace(track.id, { personId: null, score: -1, status: 'pending' }, 700);
  assert.equal(track.personId, 'alice');
  assert.equal(track.status, 'match');
  assert.equal(track.score, 0.8);
  assert.equal(track.lastEmbedAt, 0);
});

test('an unsure face does not downgrade a confident face match', () => {
  const { tracker, track } = singleTrack();
  tracker.applyFace(track.id, { personId: 'alice', score: 0.8, status: 'match' }, 0);
  tracker.applyFace(track.id, { personId: 'bob', score: 0.32, status: 'unsure' }, 1000);
  assert.equal(track.personId, 'alice');
  assert.equal(track.status, 'match');
  assert.equal(track.lastEmbedAt, 1000);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Alice']);
});

test('a later face match replaces an unsure one and a different match replaces the person', () => {
  const { tracker, track } = singleTrack();
  tracker.applyFace(track.id, { personId: 'alice', score: 0.33, status: 'unsure' }, 0);
  tracker.applyFace(track.id, { personId: 'alice', score: 0.7, status: 'match' }, 100);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Alice']);
  tracker.applyFace(track.id, { personId: 'bob', score: 0.6, status: 'match' }, 1100);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Bob']);
});

test('R4: an unknown face clears a name the track got from a face match', () => {
  const { tracker, track } = singleTrack();
  tracker.applyFace(track.id, { personId: 'alice', score: 0.8, status: 'match' }, 0);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Alice']);

  // Two people swapped places: a face is visible, embedded, and matches nobody.
  tracker.applyFace(track.id, { personId: null, score: 0.1, status: 'unknown' }, 1000);
  assert.equal(track.personId, null);
  assert.equal(track.source, null);
  assert.equal(track.status, 'unknown');
  assert.equal(track.lastEmbedAt, 1000);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Unknown']);
});

test('R4: an unknown face also clears a name the track got from its shirt', () => {
  const tracker = new Tracker();
  const [hidden] = tracker.associate([observation(BOX, false)], 0);
  tracker.applyShirt(hidden.id, { personId: 'bob', score: 0.9 });
  const [shown] = tracker.associate([observation(BOX, true)], 100);
  tracker.applyFace(shown.id, { personId: null, score: 0.1, status: 'unknown' }, 100);
  assert.equal(shown.personId, null);
  assert.equal(shown.source, null);
  assert.equal(shown.status, 'unknown');
  assert.deepEqual(textsOf(tracker.labels(people)), ['Unknown']);
});

test('R4: an unknown face clears an unsure name', () => {
  const { tracker, track } = singleTrack();
  tracker.applyFace(track.id, { personId: 'alice', score: 0.33, status: 'unsure' }, 0);
  tracker.applyFace(track.id, { personId: null, score: 0.1, status: 'unknown' }, 100);
  assert.equal(track.personId, null);
  assert.deepEqual(textsOf(tracker.labels(people)), ['Unknown']);
});

test('R6: shirt matching does not apply to a track whose face is visible', () => {
  const { tracker, track } = singleTrack({ withFace: true });
  tracker.applyFace(track.id, { personId: null, score: 0.1, status: 'unknown' }, 0);
  assert.equal(track.personId, null);

  tracker.applyShirt(track.id, { personId: 'bob', score: 0.95 });
  assert.equal(track.personId, null);
  assert.equal(track.source, null);
  assert.equal(track.status, 'unknown');
  assert.deepEqual(textsOf(tracker.labels(people)), ['Unknown']);
});

test('R6: once the face is hidden the same track can take a shirt match', () => {
  const tracker = new Tracker();
  const [shown] = tracker.associate([observation(BOX, true)], 0);
  tracker.applyFace(shown.id, { personId: null, score: 0.1, status: 'unknown' }, 0);
  const [hidden] = tracker.associate([observation(BOX, false)], 100);
  assert.equal(hidden.id, shown.id);
  tracker.applyShirt(hidden.id, { personId: 'bob', score: 0.9 });
  assert.deepEqual(textsOf(tracker.labels(people)), ['Bob 👕']);
});

// ---------------------------------------------------------------- buildObservations

/** Face-mesh stand-in: a few landmarks whose bounding box is [x0..x1] x [y0..y1], normalised. */
const faceAt = (x0, y0, x1, y1) => ({
  landmarks: [
    { x: x0, y: y0, z: 0 },
    { x: x1, y: y1, z: 0 },
    { x: (x0 + x1) / 2, y: (y0 + y1) / 2, z: 0 },
  ],
});

/** Pose stand-in: landmark 0 is the nose; every landmark carries an explicit visibility. */
const poseOf = (nose, others = []) => ({
  landmarks: [
    { x: nose[0], y: nose[1], z: 0, visibility: 0.9 },
    ...others.map(([x, y, visibility = 0.9]) => ({ x, y, z: 0, visibility })),
  ],
});

const near = (actual, expected, message) =>
  assert.ok(Math.abs(actual - expected) < 1e-9, `${message}: expected ${expected}, got ${actual}`);

function assertBox(actual, expected, label) {
  near(actual.x, expected.x, `${label}.x`);
  near(actual.y, expected.y, `${label}.y`);
  near(actual.width, expected.width, `${label}.width`);
  near(actual.height, expected.height, `${label}.height`);
}

test('buildObservations: a face containing pose 0 nose links to it, a second face becomes face-only', () => {
  const faces = [faceAt(0.4, 0.1, 0.5, 0.3), faceAt(0.8, 0.1, 0.9, 0.3)];
  const poses = [poseOf([0.45, 0.2], [[0.3, 0.5], [0.6, 0.9]])];
  const result = buildObservations(faces, poses, 1000, 500);
  assert.equal(result.length, 2);

  const [linked, faceOnly] = result;
  assert.equal(linked.faceIndex, 0);
  assert.equal(linked.poseIndex, 0);
  assertBox(linked.faceBox, { x: 400, y: 50, width: 100, height: 100 }, 'linked.faceBox');
  // Body: nose (450,100), (300,250), (600,450). Union with the face box (400..500, 50..150).
  assertBox(linked.box, { x: 300, y: 50, width: 300, height: 400 }, 'linked.box');

  assert.equal(faceOnly.faceIndex, 1);
  assert.equal(faceOnly.poseIndex, null);
  assertBox(faceOnly.faceBox, { x: 800, y: 50, width: 100, height: 100 }, 'faceOnly.faceBox');
  assertBox(faceOnly.box, faceOnly.faceBox, 'faceOnly.box');
});

test('buildObservations: an unlinked pose becomes a pose-only observation', () => {
  const poses = [poseOf([0.2, 0.2], [[0.1, 0.8], [0.3, 0.9]])];
  const [only] = buildObservations([], poses, 1000, 500);
  assert.equal(only.faceIndex, null);
  assert.equal(only.faceBox, null);
  assert.equal(only.poseIndex, 0);
  assertBox(only.box, { x: 100, y: 100, width: 200, height: 350 }, 'only.box');
});

test('buildObservations: the body box ignores landmarks below visibility 0.5', () => {
  const poses = [poseOf([0.2, 0.2], [[0.3, 0.4, 0.5], [0.9, 0.9, 0.49], [0.0, 0.0, 0.1]])];
  const [only] = buildObservations([], poses, 1000, 1000);
  assertBox(only.box, { x: 200, y: 200, width: 100, height: 200 }, 'only.box');
});

test('buildObservations: a pose with no visible landmarks makes no observation', () => {
  const blind = {
    landmarks: [
      { x: 0.5, y: 0.5, z: 0, visibility: 0.2 },
      { x: 0.6, y: 0.7, z: 0, visibility: 0.4 },
    ],
  };
  assert.deepEqual(buildObservations([], [blind], 1000, 500), []);
});

test('buildObservations: a pose is claimed by only one face', () => {
  const faces = [faceAt(0.4, 0.1, 0.5, 0.3), faceAt(0.35, 0.05, 0.55, 0.35)];
  const poses = [poseOf([0.45, 0.2], [[0.5, 0.8]])];
  const result = buildObservations(faces, poses, 1000, 1000);
  assert.equal(result.length, 2);
  assert.equal(result[0].poseIndex, 0);
  assert.equal(result[1].poseIndex, null);
  assert.equal(result[1].faceIndex, 1);
});

test('buildObservations: a face links to the first unclaimed pose whose nose is inside it', () => {
  const faces = [faceAt(0.4, 0.1, 0.5, 0.3)];
  const poses = [
    poseOf([0.1, 0.2], [[0.15, 0.8]]), // nose outside the face
    poseOf([0.45, 0.2], [[0.5, 0.8]]), // nose inside
    poseOf([0.46, 0.21], [[0.5, 0.9]]), // also inside, but later
  ];
  const result = buildObservations(faces, poses, 1000, 1000);
  assert.deepEqual(
    result.map(({ faceIndex, poseIndex }) => [faceIndex, poseIndex]),
    [[0, 1], [null, 0], [null, 2]],
  );
});

test('buildObservations: nothing in, nothing out', () => {
  assert.deepEqual(buildObservations([], [], 640, 480), []);
});
