// Matching logic: compare a face embedding against each person's stored samples, pair faces to
// people one-to-one per frame, and match a shirt histogram when the face is hidden.
// Pure module: no DOM, no CDN imports. Embeddings may be plain arrays (from storage) or
// Float32Arrays (live); histograms are normalised 28-bin arrays or Float32Arrays.

import { FACE_MATCH, FACE_UNSURE, SHIRT_MATCH, SHIRT_MARGIN } from '../config.js';
import { histogramSimilarity } from './shirt.js';

/** Cosine similarity: dot(a, b) / (|a| |b|). 0 when either vector has zero length. */
export function cosine(a, b) {
  const length = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** A person's score for a face: the best cosine over their stored samples, or -1 if they have none. */
export function scorePerson(embedding, person) {
  let best = -1;
  for (const sample of person.faceEmbeddings ?? []) {
    best = Math.max(best, cosine(embedding, sample));
  }
  return best;
}

/**
 * Pair faces with people for one frame. Every (face, person) pair scoring at least FACE_UNSURE is
 * a candidate; candidates are taken best-first, each face and each person at most once. Ties go to
 * the lower face index, then the earlier person. The result is index-aligned with `embeddings`:
 *  - null embedding (not computed yet)  -> { personId: null, score: -1, status: 'pending' }
 *  - assigned, score >= FACE_MATCH       -> status 'match'
 *  - assigned, score <  FACE_MATCH       -> status 'unsure'
 *  - not assigned                        -> status 'unknown', personId null, score = best score (or -1)
 */
export function assignFaces(embeddings, people) {
  const results = embeddings.map((embedding) => ({
    personId: null,
    score: -1,
    status: embedding ? 'unknown' : 'pending',
  }));

  const pairs = [];
  embeddings.forEach((embedding, faceIndex) => {
    if (!embedding) return;
    people.forEach((person, personIndex) => {
      const score = scorePerson(embedding, person);
      if (score > results[faceIndex].score) results[faceIndex].score = score;
      if (score >= FACE_UNSURE) pairs.push({ faceIndex, personIndex, score });
    });
  });

  pairs.sort((x, y) => y.score - x.score || x.faceIndex - y.faceIndex || x.personIndex - y.personIndex);

  const faceTaken = new Set();
  const personTaken = new Set();
  for (const { faceIndex, personIndex, score } of pairs) {
    if (faceTaken.has(faceIndex) || personTaken.has(personIndex)) continue;
    faceTaken.add(faceIndex);
    personTaken.add(personIndex);
    results[faceIndex] = {
      personId: people[personIndex].id,
      score,
      status: score >= FACE_MATCH ? 'match' : 'unsure',
    };
  }
  return results;
}

/**
 * Match a shirt histogram against candidates [{ personId, histogram }]. The best candidate wins
 * only if it scores at least SHIRT_MATCH and leads the runner-up by at least SHIRT_MARGIN (the
 * runner-up counts as 0 when there is a single candidate). `score` is the best similarity, or 0
 * when there are no candidates.
 */
export function matchShirt(histogram, candidates) {
  let best = null;
  let bestScore = 0;
  let secondScore = 0;
  for (const candidate of candidates) {
    const score = histogramSimilarity(histogram, candidate.histogram);
    if (best === null || score > bestScore) {
      secondScore = bestScore;
      best = candidate;
      bestScore = score;
    } else if (score > secondScore) {
      secondScore = score;
    }
  }
  if (best === null) return { personId: null, score: 0 };

  const wins = bestScore >= SHIRT_MATCH && bestScore - secondScore >= SHIRT_MARGIN;
  return { personId: wins ? best.personId : null, score: bestScore };
}
