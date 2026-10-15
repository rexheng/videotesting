// Feasibility gate: MediaPipe landmarks + SFace in the browser must give correct identity embeddings,
// checked against OpenCV's own pipeline (tests/fixtures/faces/reference.json, from
// scripts/reference_embeddings.py). Run with UPDATE_FIXTURES=1 to rewrite js-embeddings.json.
import { test, expect } from '@playwright/test';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { FACE_MATCH } from '../../src/config.js';

const NAMES = ['a1', 'a2', 'b1'];
const facesDir = new URL('../fixtures/faces/', import.meta.url);
const jsEmbeddingsFile = new URL('js-embeddings.json', facesDir);
const reference = JSON.parse(readFileSync(new URL('reference.json', facesDir), 'utf8'));

function cosine(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / Math.sqrt(normA * normB);
}

/** name -> { embedding, yaw, points, detectMs, embedMs } from the browser pipeline. */
let results;

test.beforeAll(async ({ browser }) => {
  const page = await browser.newPage();
  await page.goto('/tests/e2e/harness/embed.html');
  await page.evaluate(() => window.modelsReady.then(() => true));
  results = {};
  for (const name of NAMES) {
    results[name] = await page.evaluate((url) => window.runEmbedding(url), `/tests/fixtures/faces/${name}.jpg`);
  }
  await page.close();

  const fmt = (x) => x.toFixed(4);
  console.log(
    [
      `cos(a1,a2)=${fmt(cosine(results.a1.embedding, results.a2.embedding))}`,
      `cos(a1,b1)=${fmt(cosine(results.a1.embedding, results.b1.embedding))}`,
      `cos(a2,b1)=${fmt(cosine(results.a2.embedding, results.b1.embedding))}`,
      ...NAMES.map(
        (n) =>
          `${n}: vs OpenCV ${fmt(cosine(results[n].embedding, reference[n]))}, yaw ${results[n].yaw.toFixed(2)}, ` +
          `detect ${results[n].detectMs.toFixed(0)} ms, embed ${results[n].embedMs.toFixed(0)} ms`,
      ),
    ].join('\n'),
  );
});

test('same person matches and different people do not', () => {
  expect(cosine(results.a1.embedding, results.a2.embedding)).toBeGreaterThanOrEqual(FACE_MATCH);
  expect(cosine(results.a1.embedding, results.b1.embedding)).toBeLessThan(FACE_MATCH);
  expect(cosine(results.a2.embedding, results.b1.embedding)).toBeLessThan(FACE_MATCH);
});

test('embeddings agree with the OpenCV reference', () => {
  for (const name of NAMES) {
    expect(results[name].embedding).toHaveLength(128);
    expect(cosine(results[name].embedding, reference[name]), name).toBeGreaterThanOrEqual(0.9);
  }
});

test('landmarks are not mirrored and faces are roughly frontal', () => {
  for (const name of NAMES) {
    const { points, yaw } = results[name];
    // The subject's right eye is on the image's left.
    expect(points[0][0], name).toBeLessThan(points[1][0]);
    expect(Math.abs(yaw), name).toBeLessThanOrEqual(15);
  }
  // a1 must sit well inside the enrolment "look straight" window.
  expect(Math.abs(results.a1.yaw)).toBeLessThanOrEqual(8);
});

test('js-embeddings.json fixture exists and matches the pipeline', () => {
  if (process.env.UPDATE_FIXTURES === '1') {
    const data = Object.fromEntries(NAMES.map((n) => [n, results[n].embedding.map((v) => Number(v.toFixed(7)))]));
    writeFileSync(jsEmbeddingsFile, `${JSON.stringify(data)}\n`);
  }
  expect(existsSync(jsEmbeddingsFile)).toBe(true);
  const saved = JSON.parse(readFileSync(jsEmbeddingsFile, 'utf8'));
  for (const name of NAMES) {
    expect(saved[name], name).toHaveLength(128);
    // Loose: landmarks differ slightly between GPU and CPU delegates, but a stale file would not pass.
    expect(cosine(saved[name], results[name].embedding), name).toBeGreaterThanOrEqual(0.95);
  }
});
