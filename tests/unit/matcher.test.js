import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cosine, scorePerson, assignFaces, matchShirt } from '../../src/recognition/matcher.js';
import { histogramSimilarity } from '../../src/recognition/shirt.js';
import { FACE_MATCH, FACE_UNSURE, SHIRT_MATCH, SHIRT_MARGIN } from '../../src/config.js';

const near = (actual, expected, tolerance = 1e-6, message) =>
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${message ?? 'value'}: expected ${expected} +/- ${tolerance}, got ${actual}`,
  );

// People are basis vectors in 4-D (plain arrays, as they come from storage).
const alice = { id: 'alice', name: 'Alice', faceEmbeddings: [[1, 0, 0, 0]] };
const bob = { id: 'bob', name: 'Bob', faceEmbeddings: [[0, 1, 0, 0]] };

/** Unit-length face (Float32Array) whose cosine to Alice is alpha and to Bob is beta. */
function face(alpha, beta) {
  return Float32Array.from([alpha, beta, Math.sqrt(1 - alpha * alpha - beta * beta), 0]);
}

/** Two-bin histogram whose similarity to ONE ([1, 0]) is exactly `similarity`. */
const ONE = [1, 0];
const histogramAt = (similarity) => [similarity * similarity, 1 - similarity * similarity];

test('config thresholds are the values the brief assumes', () => {
  assert.equal(FACE_MATCH, 0.363);
  assert.equal(FACE_UNSURE, 0.3);
  assert.equal(SHIRT_MATCH, 0.75);
  assert.equal(SHIRT_MARGIN, 0.1);
});

test('cosine of identical vectors is 1', () => {
  near(cosine([0.3, -0.4, 0.5, 0.1], [0.3, -0.4, 0.5, 0.1]), 1);
  near(cosine(Float32Array.from([2, 0, 0, 0]), [5, 0, 0, 0]), 1);
});

test('cosine of orthogonal vectors is 0', () => {
  near(cosine([1, 0, 0, 0], [0, 1, 0, 0]), 0);
});

test('cosine of opposite vectors is -1', () => {
  near(cosine([1, 2, 3], [-1, -2, -3]), -1);
});

test('cosine uses the full formula, so unnormalised vectors are scaled out', () => {
  near(cosine([3, 4], [4, 3]), 24 / 25);
  near(cosine([30, 40], [0.4, 0.3]), 24 / 25);
});

test('cosine with a zero or empty vector is 0', () => {
  assert.equal(cosine([0, 0, 0], [1, 2, 3]), 0);
  assert.equal(cosine([1, 2, 3], [0, 0, 0]), 0);
  assert.equal(cosine([], []), 0);
});

test('scorePerson is the max cosine over the stored samples', () => {
  const person = {
    id: 'p',
    name: 'P',
    faceEmbeddings: [
      [0, 1, 0, 0],
      [1, 0, 0, 0],
      [0.6, 0.8, 0, 0],
    ],
  };
  near(scorePerson(face(0.9, 0.1), person), 0.9);
  near(scorePerson(face(0.1, 0.9), person), 0.9);
});

test('scorePerson is -1 when the person has no samples', () => {
  assert.equal(scorePerson(face(0.5, 0.5), { id: 'x', name: 'X', faceEmbeddings: [] }), -1);
  assert.equal(scorePerson(face(0.5, 0.5), { id: 'x', name: 'X' }), -1);
});

test('scorePerson accepts Float32Array samples', () => {
  const person = { id: 'p', name: 'P', faceEmbeddings: [Float32Array.from([1, 0, 0, 0])] };
  near(scorePerson(face(0.7, 0.1), person), 0.7);
});

test('assignFaces pairs one-to-one greedily by score', () => {
  const result = assignFaces([face(0.5, 0.6), face(0.33, 0.55)], [alice, bob]);
  assert.equal(result.length, 2);
  assert.equal(result[0].personId, 'bob');
  assert.equal(result[0].status, 'match');
  near(result[0].score, 0.6);
  // Face 1 prefers Bob (0.55) but loses him to the higher-scoring face 0, so it takes Alice.
  assert.equal(result[1].personId, 'alice');
  assert.equal(result[1].status, 'unsure');
  near(result[1].score, 0.33);
});

test('assignFaces results are index-aligned however the faces are ordered', () => {
  const result = assignFaces([face(0.33, 0.55), face(0.5, 0.6)], [alice, bob]);
  assert.equal(result[0].personId, 'alice');
  assert.equal(result[0].status, 'unsure');
  assert.equal(result[1].personId, 'bob');
  assert.equal(result[1].status, 'match');
});

test('assignFaces: 0.35 is unsure, 0.40 is match, 0.20 is unknown', () => {
  const unsure = assignFaces([face(0.35, 0)], [alice, bob]);
  assert.equal(unsure[0].status, 'unsure');
  assert.equal(unsure[0].personId, 'alice');
  near(unsure[0].score, 0.35);

  const match = assignFaces([face(0.4, 0)], [alice, bob]);
  assert.equal(match[0].status, 'match');
  assert.equal(match[0].personId, 'alice');
  near(match[0].score, 0.4);

  const unknown = assignFaces([face(0.2, 0)], [alice, bob]);
  assert.equal(unknown[0].status, 'unknown');
  assert.equal(unknown[0].personId, null);
  near(unknown[0].score, 0.2);
});

test('assignFaces: a face that loses every candidate is unknown with its best score', () => {
  // Both faces only resemble Alice; the stronger one keeps her.
  const result = assignFaces([face(0.45, 0), face(0.6, 0)], [alice, bob]);
  assert.equal(result[1].personId, 'alice');
  assert.equal(result[1].status, 'match');
  assert.equal(result[0].personId, null);
  assert.equal(result[0].status, 'unknown');
  near(result[0].score, 0.45);
});

test('assignFaces: a null embedding is pending and does not take a person', () => {
  const result = assignFaces([null, face(0.8, 0)], [alice, bob]);
  assert.deepEqual(result[0], { personId: null, score: -1, status: 'pending' });
  assert.equal(result[1].personId, 'alice');
  assert.equal(result[1].status, 'match');
});

test('assignFaces: no people means every real face is unknown with score -1', () => {
  const result = assignFaces([face(0.5, 0.5), null], []);
  assert.deepEqual(result[0], { personId: null, score: -1, status: 'unknown' });
  assert.equal(result[1].status, 'pending');
});

test('assignFaces: empty input gives empty output', () => {
  assert.deepEqual(assignFaces([], [alice, bob]), []);
});

test('assignFaces: a person with no samples is never assigned', () => {
  const empty = { id: 'empty', name: 'Empty', faceEmbeddings: [] };
  const result = assignFaces([face(0.2, 0)], [empty]);
  assert.deepEqual(result[0], { personId: null, score: -1, status: 'unknown' });
});

test('assignFaces: equal scores are broken by face index, then person order', () => {
  // Two identical faces, one person: the lower face index wins.
  const same = assignFaces([face(0.5, 0), face(0.5, 0)], [alice, bob]);
  assert.equal(same[0].personId, 'alice');
  assert.equal(same[0].status, 'match');
  assert.equal(same[1].personId, null);
  assert.equal(same[1].status, 'unknown');
  near(same[1].score, 0.5);

  // One face equally close to two people: the earlier person in the list wins.
  const duo = assignFaces([face(0.5, 0.5)], [bob, alice]);
  assert.equal(duo[0].personId, 'bob');
});

test('assignFaces does not mutate its inputs', () => {
  const faces = [face(0.5, 0.6), null];
  const people = [alice, bob];
  const facesBefore = JSON.stringify(faces.map((f) => (f ? Array.from(f) : f)));
  const peopleBefore = JSON.stringify(people);
  assignFaces(faces, people);
  assert.equal(JSON.stringify(faces.map((f) => (f ? Array.from(f) : f))), facesBefore);
  assert.equal(JSON.stringify(people), peopleBefore);
});

test('matchShirt: scores 0.9 and 0.85 are too close to call', () => {
  const result = matchShirt(ONE, [
    { personId: 'a', histogram: histogramAt(0.9) },
    { personId: 'b', histogram: histogramAt(0.85) },
  ]);
  assert.equal(result.personId, null);
  near(result.score, 0.9);
});

test('matchShirt: scores 0.9 and 0.7 give the clear winner', () => {
  const result = matchShirt(ONE, [
    { personId: 'b', histogram: histogramAt(0.7) },
    { personId: 'a', histogram: histogramAt(0.9) },
  ]);
  assert.equal(result.personId, 'a');
  near(result.score, 0.9);
});

test('matchShirt: a single candidate at 0.8 wins, at 0.7 does not', () => {
  const win = matchShirt(ONE, [{ personId: 'a', histogram: histogramAt(0.8) }]);
  assert.equal(win.personId, 'a');
  near(win.score, 0.8);

  const lose = matchShirt(ONE, [{ personId: 'a', histogram: histogramAt(0.7) }]);
  assert.equal(lose.personId, null);
  near(lose.score, 0.7);
});

test('matchShirt: no candidates gives null and score 0', () => {
  assert.deepEqual(matchShirt(ONE, []), { personId: null, score: 0 });
});

test('matchShirt: score is the histogramSimilarity of the best candidate', () => {
  const histogram = histogramAt(0.82);
  const result = matchShirt(ONE, [{ personId: 'a', histogram }]);
  assert.equal(result.score, histogramSimilarity(ONE, histogram));
});

test('matchShirt: a close runner-up blocks a win even when the best is well above SHIRT_MATCH', () => {
  const result = matchShirt(ONE, [
    { personId: 'a', histogram: histogramAt(0.95) },
    { personId: 'b', histogram: histogramAt(0.9) },
    { personId: 'c', histogram: histogramAt(0.2) },
  ]);
  assert.equal(result.personId, null);
  near(result.score, 0.95);
});

test('matchShirt accepts Float32Array histograms', () => {
  const result = matchShirt(Float32Array.from(ONE), [
    { personId: 'a', histogram: Float32Array.from(histogramAt(0.9)) },
    { personId: 'b', histogram: Float32Array.from(histogramAt(0.5)) },
  ]);
  assert.equal(result.personId, 'a');
});
