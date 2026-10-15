// The per-frame pipeline in src/app.js, driven frame by frame with stub detectors and a stub SFace
// session on a painted canvas, so face/shirt orchestration and sign-up can be checked exactly. The
// real models are covered by tagging.spec.js and enrol.spec.js.
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

    /**
     * 478 face-mesh landmarks (normalised) for a face centred on (cx, cy), `scale` times the
     * standard size, turned by `yaw` degrees (the cheek landmarks' depth gives yawDegrees).
     */
    function face(cx, cy, { yaw = 0, scale = 1 } = {}) {
      const points = Array.from({ length: 478 }, (_, i) => {
        const angle = (2 * Math.PI * i) / 478;
        return { x: cx + 0.06 * scale * Math.cos(angle), y: cy + 0.06 * scale * Math.sin(angle), z: 0 };
      });
      const set = (i, x, y, z = 0) => (points[i] = { x: cx + x * scale, y: cy + y * scale, z: z * scale });
      set(33, -0.03, -0.02);
      set(133, -0.03, -0.02);
      set(362, 0.03, -0.02);
      set(263, 0.03, -0.02);
      set(1, 0, 0.01);
      set(61, -0.025, 0.05);
      set(291, 0.025, 0.05);
      const depth = 0.06 * Math.tan((yaw * Math.PI) / 180);
      set(234, -0.06, 0, -depth);
      set(454, 0.06, 0, depth);
      return points;
    }

    /** 33 pose landmarks: nose at (cx, cy), shoulders and hips below it, the rest not visible. */
    function pose(cx, cy, scale = 1) {
      const points = Array.from({ length: 33 }, () => ({ x: cx, y: cy, z: 0, visibility: 0 }));
      const set = (i, x, y) => (points[i] = { x: cx + x * scale, y: cy + y * scale, z: 0, visibility: 0.99 });
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
        // faces: [{ id: 'R'|'G'|'B', cx, cy, yaw?, scale? }], poses: [{ cx, cy, shirt: '#rrggbb', scale? }]
        set(nextFaces, nextPoses) {
          ctx.fillStyle = '#808080';
          ctx.fillRect(0, 0, W, H);
          for (const p of nextPoses) {
            const s = p.scale ?? 1;
            ctx.fillStyle = p.shirt;
            ctx.fillRect((p.cx - 0.15 * s) * W, (p.cy + 0.15 * s) * H, 0.3 * s * W, 0.5 * s * H);
          }
          for (const f of nextFaces) {
            const s = f.scale ?? 1;
            ctx.fillStyle = COLOURS[f.id];
            ctx.fillRect(f.cx * W - 70 * s, f.cy * H - 70 * s, 140 * s, 140 * s);
          }
          faces = nextFaces.map((f) => face(f.cx, f.cy, f));
          poses = nextPoses.map((p) => pose(p.cx, p.cy, p.scale));
        },
        get faces() {
          return faces;
        },
        get poses() {
          return poses;
        },
      };
    }

    // `saveResult` is what store.save returns (default true). Person A is signed up to start with.
    window.__pipeline = async function createHarness({ saveResult = true } = {}) {
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
      const saves = [];
      const store = {
        available: true,
        load: () => [personA],
        save(next) {
          saves.push(JSON.parse(JSON.stringify(next)));
          return saveResult;
        },
      };
      // Sign-up states the app reports (null = cancelled), and its people changes.
      const updates = [];
      const changes = [];
      const overlay = document.createElement('canvas');
      const app = createApp({
        video: scene.video,
        overlay,
        models,
        store,
        onEnrolUpdate: (state) =>
          updates.push(state && { step: state.step, progress: state.progress, prompt: state.prompt, arrow: state.arrow }),
        onPeopleChange: (next, saved) => changes.push({ names: next.map((person) => person.name), saved }),
      });
      return {
        app,
        saves,
        updates,
        changes,
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

test.describe('sign-up', () => {
  test('a full sign-up uses the largest face and body, saves the new person, and live tagging carries on', async ({ page }) => {
    const result = await page.evaluate(async ({ b1 }) => {
      const harness = await window.__pipeline();
      // Person A (R) stands further back; the person signing up (G) is the nearest face and body.
      const far = (extra = {}) => ({ id: 'R', cx: 0.2, cy: 0.25, scale: 0.6, ...extra });
      const near = (extra = {}) => ({ id: 'G', cx: 0.6, cy: 0.25, ...extra });
      const bodies = [
        { cx: 0.2, cy: 0.25, scale: 0.6, shirt: '#ff00ff' },
        { cx: 0.6, cy: 0.25, shirt: '#ffff00' },
      ];

      const start = harness.app.startEnrolment('Gemma', { now: 1000 });
      const labels = [];
      let now = 1000;
      // 5 samples per step, 200 ms apart: straight, then turned left (+30°), then right (-30°), then the shirt.
      for (const yaw of [0, 0, 0, 0, 0, 30, 30, 30, 30, 30, -30, -30, -30, -30, -30, 0, 0, 0, 0, 0]) {
        now += 200;
        labels.push((await harness.frame(now, [far(), near({ yaw })], bodies)).texts);
      }
      const after = await harness.frame(now + 200, [far(), near()], bodies);
      const saved = harness.saves.at(-1)?.at(-1);
      return {
        start: start.step,
        updates: harness.updates,
        saves: harness.saves.length,
        saved,
        usedNearFace: saved?.faceEmbeddings.every((e) => Math.abs(e[0] - b1[0]) < 1e-3 && Math.abs(e[5] - b1[5]) < 1e-3),
        changes: harness.changes,
        labelsDuringSignUp: labels[0],
        after: after.texts,
      };
    }, embeddings);

    expect(result.start).toBe('straight');
    const steps = result.updates.map((update) => update.step);
    // The first update is sent when sign-up starts, then one per frame.
    expect(steps).toEqual([
      'straight',
      ...Array(4).fill('straight'),
      ...Array(5).fill('left'),
      ...Array(5).fill('right'),
      ...Array(5).fill('shirt'),
      'done',
    ]);
    expect(result.updates[0]).toEqual({ step: 'straight', progress: 0, prompt: 'Look straight at the camera', arrow: null });
    expect(result.updates[1].progress).toBeCloseTo(0.2);
    expect(result.updates[5]).toMatchObject({ step: 'left', arrow: 'left', prompt: 'Slowly turn your head LEFT' });
    expect(result.updates[10]).toMatchObject({ step: 'right', arrow: 'right', prompt: 'Now turn RIGHT' });
    expect(result.updates.at(-1)).toEqual({ step: 'done', progress: 1, prompt: 'Done! Hi Gemma 👋', arrow: null });

    // Saved once, at the end, with everyone: Person A unchanged plus the new person.
    expect(result.saves).toBe(1);
    expect(result.changes).toEqual([{ names: ['Person A', 'Gemma'], saved: true }]);
    expect(result.saved).toMatchObject({ name: 'Gemma', shirtSwatch: '#ffff00' });
    expect(result.saved.faceEmbeddings).toHaveLength(15);
    expect(result.usedNearFace).toBe(true);
    expect(result.saved.shirtHistogram).toHaveLength(28);
    expect(result.saved.colour).not.toBe('#e6194b'); // Person A's colour

    // Live tagging kept running during sign-up, and names the new person straight after.
    expect(result.labelsDuringSignUp).toEqual(['Person A', 'Unknown']);
    expect(result.after).toEqual(['Gemma', 'Person A']);
  });

  test('Re-scan shirt replaces only that person\'s shirt', async ({ page }) => {
    const result = await page.evaluate(async ({ a1 }) => {
      const harness = await window.__pipeline();
      const start = harness.app.startEnrolment('Person A', { personId: 'person-a', now: 1000 });
      for (let i = 1; i <= 5; i++) {
        await harness.frame(1000 + i * 200, [{ id: 'R', cx: 0.5, cy: 0.25 }], [{ cx: 0.5, cy: 0.25, shirt: '#00ffff' }]);
      }
      return { start, updates: harness.updates, saves: harness.saves, changes: harness.changes, a1 };
    }, embeddings);

    expect(result.start).toMatchObject({ step: 'shirt', prompt: 'Step back so we can see your shirt' });
    expect(result.updates.at(-1)).toMatchObject({ step: 'done', prompt: 'Shirt updated for Person A 👕' });
    expect(result.saves).toHaveLength(1);
    const [person] = result.saves[0];
    expect(result.saves[0]).toHaveLength(1);
    expect(person).toMatchObject({ id: 'person-a', name: 'Person A', colour: '#e6194b', createdAt: 0, shirtSwatch: '#00ffff' });
    expect(person.faceEmbeddings).toEqual([result.a1]);
    expect(person.shirtHistogram).toHaveLength(28);
    expect(person.shirtHistogram).not.toEqual(Array.from({ length: 28 }, () => 1 / 28));
    expect(result.changes).toEqual([{ names: ['Person A'], saved: true }]);
  });

  test('Cancel discards the sign-up; deleting a person stops their re-scan; a failed save is reported', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const face = [{ id: 'G', cx: 0.5, cy: 0.25 }];

      const cancelled = await window.__pipeline();
      cancelled.app.startEnrolment('Gemma', { now: 1000 });
      await cancelled.frame(1200, face);
      cancelled.app.cancelEnrolment();
      const updatesAtCancel = cancelled.updates.length;
      for (let i = 1; i <= 6; i++) await cancelled.frame(1200 + i * 200, face);

      const deleted = await window.__pipeline({ saveResult: false });
      deleted.app.startEnrolment('Person A', { personId: 'person-a', now: 1000 });
      deleted.app.deletePerson('person-a');
      const peopleAfterDelete = deleted.app.getPeople().length;
      for (let i = 1; i <= 6; i++) {
        await deleted.frame(1000 + i * 200, [], [{ cx: 0.5, cy: 0.25, shirt: '#00ffff' }]);
      }

      return {
        cancelled: { updates: cancelled.updates, updatesAtCancel, saves: cancelled.saves.length },
        deleted: { updates: deleted.updates, saves: deleted.saves, changes: deleted.changes, peopleAfterDelete },
      };
    });

    // Cancel reports null and nothing more happens.
    expect(result.cancelled.updates.at(-1)).toBeNull();
    expect(result.cancelled.updates).toHaveLength(result.cancelled.updatesAtCancel);
    expect(result.cancelled.saves).toBe(0);

    // Deleting the person being re-scanned ends the re-scan; only the delete is saved.
    expect(result.deleted.updates.map((update) => update && update.step)).toEqual(['shirt', null]);
    expect(result.deleted.saves).toEqual([[]]);
    expect(result.deleted.peopleAfterDelete).toBe(0);
    expect(result.deleted.changes).toEqual([{ names: [], saved: false }]);
  });
});
