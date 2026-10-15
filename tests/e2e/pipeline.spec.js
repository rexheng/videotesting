// The per-frame pipeline in src/app.js, driven frame by frame with stub detectors and a stub SFace
// session on a painted canvas, so face/shirt orchestration can be checked exactly. The real models
// are covered by tagging.spec.js.
//
// Each painted face is a solid red, green or blue square; the stub SFace session tells faces apart
// by the dominant colour of its 112x112 input and logs which face it embedded.
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { denyCamera } from './helpers/fakeCamera.js';

const embeddings = JSON.parse(readFileSync(new URL('../fixtures/faces/js-embeddings.json', import.meta.url), 'utf8'));

test.beforeEach(async ({ page }) => {
  // The camera is denied so the page's own app never starts; the tests build their own.
  await denyCamera(page);
  await page.goto('/');
  await page.evaluate(({ a1, b1 }) => {
    const W = 640;
    const H = 480;

    /** 478 face-mesh landmarks (normalised) for a frontal face centred on (cx, cy). */
    function face(cx, cy) {
      const points = Array.from({ length: 478 }, (_, i) => {
        const angle = (2 * Math.PI * i) / 478;
        return { x: cx + 0.06 * Math.cos(angle), y: cy + 0.06 * Math.sin(angle), z: 0 };
      });
      const set = (i, x, y) => (points[i] = { x: cx + x, y: cy + y, z: 0 });
      set(33, -0.03, -0.02);
      set(133, -0.03, -0.02);
      set(362, 0.03, -0.02);
      set(263, 0.03, -0.02);
      set(1, 0, 0.01);
      set(61, -0.025, 0.05);
      set(291, 0.025, 0.05);
      set(234, -0.06, 0);
      set(454, 0.06, 0);
      return points;
    }

    /** 33 pose landmarks: nose at (cx, cy), shoulders and hips below it, the rest not visible. */
    function pose(cx, cy) {
      const points = Array.from({ length: 33 }, () => ({ x: cx, y: cy, z: 0, visibility: 0 }));
      const set = (i, x, y) => (points[i] = { x: cx + x, y: cy + y, z: 0, visibility: 0.99 });
      set(0, 0, 0);
      set(11, 0.12, 0.2);
      set(12, -0.12, 0.2);
      set(23, 0.1, 0.6);
      set(24, -0.1, 0.6);
      return points;
    }

    const COLOURS = { R: '#ff0000', G: '#00ff00', B: '#0000ff' };

    /** A canvas posing as the camera <video>, plus a scene setter. */
    function createScene() {
      const video = document.createElement('canvas');
      video.width = W;
      video.height = H;
      Object.defineProperties(video, {
        videoWidth: { value: W },
        videoHeight: { value: H },
        readyState: { value: 4 },
      });
      const ctx = video.getContext('2d');
      let faces = [];
      let poses = [];
      return {
        video,
        // faces: [{ id: 'R'|'G'|'B', cx, cy }], poses: [{ cx, cy, shirt: '#rrggbb' }]
        set(nextFaces, nextPoses) {
          ctx.fillStyle = '#808080';
          ctx.fillRect(0, 0, W, H);
          for (const p of nextPoses) {
            ctx.fillStyle = p.shirt;
            ctx.fillRect((p.cx - 0.15) * W, (p.cy + 0.15) * H, 0.3 * W, 0.5 * H);
          }
          for (const f of nextFaces) {
            ctx.fillStyle = COLOURS[f.id];
            ctx.fillRect(f.cx * W - 70, f.cy * H - 70, 140, 140);
          }
          faces = nextFaces.map((f) => face(f.cx, f.cy));
          poses = nextPoses.map((p) => pose(p.cx, p.cy));
        },
        get faces() {
          return faces;
        },
        get poses() {
          return poses;
        },
      };
    }

    window.__pipeline = async function createHarness() {
      const { createApp } = await import('/src/app.js');
      const scene = createScene();
      const timestamps = [];
      const embedded = [];
      const models = {
        face: {
          detectForVideo: (_source, ts) => {
            timestamps.push(['face', ts]);
            return { faceLandmarks: scene.faces };
          },
        },
        pose: {
          detectForVideo: (_source, ts) => {
            timestamps.push(['pose', ts]);
            return { landmarks: scene.poses };
          },
        },
        sface: {
          inputNames: ['input'],
          outputNames: ['output'],
          async run(feeds) {
            const data = feeds.input.data;
            const plane = data.length / 3;
            const mean = [0, 1, 2].map((c) => data.subarray(c * plane, (c + 1) * plane).reduce((s, v) => s + v, 0));
            const id = 'RGB'[mean.indexOf(Math.max(...mean))];
            embedded.push(id);
            // R is Person A; G and B are strangers.
            return { output: { data: Float32Array.from(id === 'R' ? a1 : b1) } };
          },
        },
      };
      const personA = {
        id: 'person-a',
        name: 'Person A',
        colour: '#e6194b',
        faceEmbeddings: [a1],
        shirtHistogram: Array.from({ length: 28 }, () => 1 / 28),
        shirtSwatch: '#808080',
        createdAt: 0,
      };
      const store = { available: true, load: () => [personA], save: () => true };
      const overlay = document.createElement('canvas');
      const app = createApp({ video: scene.video, overlay, models, store });
      return {
        timestamps,
        embedded,
        /** Show the scene, run one frame at `now`; resolves with the sorted label texts and the faces embedded. */
        async frame(now, faces, poses = []) {
          scene.set(faces, poses);
          const before = embedded.length;
          const labels = await app.processFrame(now);
          return { texts: labels.map((l) => l.text).sort(), embedded: embedded.slice(before) };
        },
      };
    };
  }, embeddings);
});

test('embeds at most 2 faces a frame, unnamed first; faces not embedded keep their name', async ({ page }) => {
  const frames = await page.evaluate(async () => {
    const app = await window.__pipeline();
    const R = { id: 'R', cx: 0.2, cy: 0.25 };
    const G = { id: 'G', cx: 0.5, cy: 0.25 };
    const B = { id: 'B', cx: 0.8, cy: 0.25 };
    const out = [];
    out.push(await app.frame(1000, [R]));
    out.push(await app.frame(1600, [R])); // not due yet; keeps R's track alive
    // R is due its 1000 ms re-check, but the two new (unnamed) faces go first.
    out.push(await app.frame(2100, [R, G, B]));
    out.push(await app.frame(2110, [R, G, B]));
    // Same time again: MediaPipe still gets a later timestamp.
    out.push(await app.frame(2110, [R, G, B]));
    return { out, timestamps: app.timestamps };
  });

  expect(frames.out[0]).toEqual({ texts: ['Person A'], embedded: ['R'] });
  expect(frames.out[1]).toEqual({ texts: ['Person A'], embedded: [] });
  expect(frames.out[2]).toEqual({ texts: ['Person A', 'Unknown', 'Unknown'], embedded: ['G', 'B'] });
  expect(frames.out[3]).toEqual({ texts: ['Person A', 'Unknown', 'Unknown'], embedded: ['R'] });
  expect(frames.out[4].embedded).toEqual([]);

  const faceTs = frames.timestamps.filter(([model]) => model === 'face').map(([, ts]) => ts);
  const poseTs = frames.timestamps.filter(([model]) => model === 'pose').map(([, ts]) => ts);
  expect(faceTs).toEqual([1000, 1600, 2100, 2110, 2111]);
  expect(poseTs).toEqual(faceTs);
});

test('a hidden face is named by the live shirt, refreshed while the face was recognised', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const shirt = '#c03030';
    const R = { id: 'R', cx: 0.25, cy: 0.2 };
    const G = { id: 'G', cx: 0.75, cy: 0.2 };
    const bodyLeft = { cx: 0.25, cy: 0.2, shirt };
    const bodyRight = { cx: 0.75, cy: 0.2, shirt };

    // Without a live refresh, the stored (flat) shirt does not match a red one.
    const control = await window.__pipeline();
    const unrefreshed = await control.frame(1000, [], [bodyRight]);

    const app = await window.__pipeline();
    const recognised = [];
    for (let i = 0; i < 5; i++) recognised.push((await app.frame(1000 + i * 100, [R], [bodyLeft])).texts);
    // Elsewhere, a body with no face and the same shirt: a new track, named by the shirt.
    const byShirt = await app.frame(1500, [], [bodyRight]);
    // A stranger's face appears on that body: a visible face is judged by the face alone.
    const strangerFace = await app.frame(1600, [G], [bodyRight]);
    const later = await app.frame(1700, [G], [bodyRight]);
    return { unrefreshed, recognised, byShirt, strangerFace, later };
  });

  expect(result.unrefreshed.texts).toEqual([]);
  expect(result.recognised).toEqual(Array(5).fill(['Person A']));
  expect(result.byShirt).toEqual({ texts: ['Person A 👕'], embedded: [] });
  expect(result.strangerFace).toEqual({ texts: ['Unknown'], embedded: ['G'] });
  expect(result.later.texts).toEqual(['Unknown']);
});
