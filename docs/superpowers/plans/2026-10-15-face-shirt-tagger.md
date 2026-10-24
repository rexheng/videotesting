# Face + Shirt Tagger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A no-build static web page that signs people up by face (straight/left/right) and shirt, then tags them live on the webcam, with the shirt as a fallback when the face is hidden.

**Architecture:** Pure-logic ES modules (geometry, shirt histograms, matcher, tracker, enrolment state machine, storage) are unit-tested in Node. Thin browser modules wrap MediaPipe Face/Pose Landmarker and an SFace ONNX model (onnxruntime-web), and Playwright drives them in headless Chromium with a fake canvas camera.

**Tech Stack:** HTML/CSS/JS ES modules (no bundler), `@mediapipe/tasks-vision@1.0.1`, `onnxruntime-web@1.30.0`, SFace ONNX (OpenCV Zoo), Node 22 `node --test`, `@playwright/test@1.56.1`.

**Spec:** `docs/superpowers/specs/2026-10-15-face-shirt-tagger-design.md`

> **Shipping target (2026-10-24): Arduino UNO Q.** Do not implement this plan as in-browser inference on the board. Section 7 of the spec and `docs/superpowers/specs/2026-10-24-uno-q-segmentation-audit.md` are binding. The runnable loop is `unoq/`. Recognition rules in the tasks below still apply; `unoq/face_shirt/scheduler.py` decides which model is allowed on a frame. Caps for the board: 2 faces, 2 poses, CPU delegate, 640×480 MJPG, clothes segmentation on its own frame.

## Global Constraints

- No build step. Browser code is plain ES modules loaded by `index.html`; CDN imports appear **only** in `src/vision/libs.js`.
- Pure modules (`src/config.js`, `src/vision/geometry.js`, `src/recognition/*`, `src/enrol/*`, `src/storage.js`, `src/camera.js` helpers, `src/ui/overlay.js` `mirrorBox`) must import nothing from CDNs and must run under Node 22 without a DOM.
- Pinned versions: `@mediapipe/tasks-vision@1.0.1`, `onnxruntime-web@1.30.0`, `@playwright/test@1.56.1`. No other runtime dependencies.
- All tunable numbers live in `src/config.js` (Task 2) and are imported from there, never re-typed.
- Nothing is uploaded: no network requests other than downloading libraries and model files.
- Coordinates: landmarks from MediaPipe are normalised (0–1); every function taking `width, height` converts to pixels of the **raw, unmirrored** camera frame. Mirroring happens only when drawing (Task 9).
- Yaw sign: **positive = the user turned to their own left** (nose moves to raw-image +x).
- UI copy is British English and must match the strings in this plan exactly.
- Camera requires a secure context: `npm start` serves on `http://localhost:8080` (localhost is secure).
- Tests: `npm test` runs `node --test tests/unit/`; `npm run test:e2e` runs Playwright. Both must pass at the end of every task.
- Playwright uses the preinstalled Chromium (`PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers`, revision 1194). Never run `playwright install`.

## Review Focus

1. **Camera blocked or missing** → a clear message in `#status`, no crash. Test: Task 1 `camera.spec.js` "shows blocked message".
2. **Mirroring** → label text reads normally and sits over the right person in the mirrored view; the "turn LEFT" arrow points to screen-left. Tests: Task 9 `mirrorBox` unit test; Task 8 arrow assertions.
3. **Two people on screen** → one name is never shown on two people at once. Test: Task 6 "one name per person".
4. **Model download fails** (offline, blocked CDN) → message plus **Try again** button, not a blank page. Test: Task 9 `tagging.spec.js` "model download failure".
5. **Browser storage blocked or corrupt** → the app still runs and shows "Not saved on this device"; corrupt data is ignored. Tests: Task 7 unit tests; Task 10 `enrol.spec.js` "storage unavailable".

---

### Task 1: Scaffold, camera and test harness

**Files:**
- Create: `package.json`, `.gitignore`, `README.md`, `index.html`, `styles.css`, `scripts/serve.js`, `src/camera.js`, `src/main.js`, `playwright.config.js`, `tests/e2e/helpers/fakeCamera.js`
- Test: `tests/unit/camera.test.js`, `tests/e2e/camera.spec.js`

**Interfaces:**
- Produces:
  - `src/camera.js`: `classifyCameraError(err): 'denied'|'not-found'|'unsupported'|'other'`, `cameraErrorMessage(reason): string`, `startCamera(videoEl): Promise<{width:number, height:number}>` (throws an Error with `.reason` set to one of the four reasons).
  - `index.html` element IDs: `#stage`, `#camera` (`<video playsinline muted autoplay>`), `#overlay` (`<canvas>`), `#sidebar`, `#add-person` (button "Add person"), `#people-list` (`<ul>`), `#status` (`role="status"`), `#enrol-panel` (`<section hidden>`), `#live-tags` (`<ul>`, visually hidden, for accessibility and tests), `#debug` (`<pre hidden>`).
  - `tests/e2e/helpers/fakeCamera.js`: `installFakeCamera(page, imageUrl?)` (overrides `navigator.mediaDevices.getUserMedia` via `page.addInitScript` to return a 640×480 `canvas.captureStream(15)`; draws `imageUrl` scaled to cover the canvas, or solid grey if none; exposes `window.__setFakeCameraImage(url)`), and `denyCamera(page)` (makes `getUserMedia` reject with `new DOMException('denied', 'NotAllowedError')`).
  - `scripts/serve.js`: a dependency-free static server for the repo root on `process.env.PORT || 8080`. MIME types for `.html .js .mjs .css .json .png .jpg .svg .wasm(application/wasm) .onnx(application/octet-stream)`. Rejects paths that escape the root.
  - `package.json`: `"type": "module"`; scripts `start` = `node scripts/serve.js`, `test` = `node --test tests/unit/`, `test:e2e` = `playwright test`; devDependency `"@playwright/test": "1.56.1"`.

- [ ] **Step 1: Write failing unit tests** in `tests/unit/camera.test.js`:

```js
assert.equal(classifyCameraError(new DOMException('x', 'NotAllowedError')), 'denied');
assert.equal(classifyCameraError(new DOMException('x', 'SecurityError')), 'denied');
assert.equal(classifyCameraError(new DOMException('x', 'NotFoundError')), 'not-found');
assert.equal(classifyCameraError(new DOMException('x', 'OverconstrainedError')), 'not-found');
assert.equal(classifyCameraError(Object.assign(new Error('no api'), { reason: 'unsupported' })), 'unsupported');
assert.equal(classifyCameraError(new Error('boom')), 'other');
assert.match(cameraErrorMessage('denied'), /^Camera access was blocked\./);
```

- [ ] **Step 2: Run `npm test`.** Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/camera.js`.** `startCamera` calls `getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })`. It throws `reason: 'unsupported'` when `navigator.mediaDevices?.getUserMedia` is missing, and resolves after `loadedmetadata` and `play()` with the video's intrinsic size. Exact messages:
  - denied: `Camera access was blocked. Click the camera icon in your browser's address bar, allow the camera, then reload the page.`
  - not-found: `No camera found. Plug in a webcam and reload the page.`
  - unsupported: `This browser can't use the camera. Try the latest Chrome, Edge, Safari or Firefox.`
  - other: `The camera couldn't start. Close other apps using it, then reload the page.`

- [ ] **Step 4: Run `npm test`.** Expected: PASS.

- [ ] **Step 5: Build the page.**
  - `index.html`: the element IDs above, the title `Face + Shirt Tagger`, and `<script type="module" src="src/main.js">`.
  - `styles.css`: the video is mirrored with `transform: scaleX(-1)`; `#overlay` sits exactly over the video and is **not** mirrored; the sidebar goes below the stage under 800 px width; large readable type (base 18 px); no horizontal scroll at 360 px.
  - `src/main.js` (Task 1 version): call `startCamera(#camera)`, size `#overlay` to the video size, and on failure put `cameraErrorMessage(classifyCameraError(err))` in `#status`.
  - `.gitignore`: `node_modules/`, `test-results/`, `playwright-report/`, `.superpowers/`, `.venv/`.
  - `README.md`: one paragraph on what the app does, then `npm install`, `npm start` → open `http://localhost:8080`, `npm test`, `npm run test:e2e`.

- [ ] **Step 6: Write e2e tests** in `tests/e2e/camera.spec.js`. `playwright.config.js`: `testDir: 'tests/e2e'`, `timeout: 120_000`, `webServer` = `PORT=8123 node scripts/serve.js` on port 8123 with `reuseExistingServer: true`, `baseURL: 'http://localhost:8123'`, Chromium headless with launch args `['--use-angle=swiftshader', '--enable-unsafe-swiftshader']`.
  - "shows the camera": `installFakeCamera(page)`, then `goto('/')`; poll until `#camera` `videoWidth > 0` and `paused === false`.
  - "shows blocked message": `denyCamera(page)`, then `goto('/')`; `#status` contains `Camera access was blocked`.

- [ ] **Step 7: Run `npm install && npm test && npm run test:e2e`.** Expected: all pass.

- [ ] **Step 8: Commit** — `git add -A && git commit -m "feat: scaffold page, camera start-up and test harness"`

---

### Task 2: Config and geometry

**Files:**
- Create: `src/config.js`, `src/vision/geometry.js`
- Test: `tests/unit/geometry.test.js`

**Interfaces:**
- Produces `src/config.js`, exactly these exports:

```js
export const VERSIONS = { mediapipe: '1.0.1', ort: '1.30.0' };
export const FACE_MATCH = 0.363;
export const FACE_UNSURE = 0.30;
export const EMBED_RECHECK_MS = 1000;
export const MAX_EMBEDS_PER_FRAME = 2;
export const TRACK_IOU = 0.3;
export const TRACK_TTL_MS = 1000;
export const SHIRT_MATCH = 0.75;
export const SHIRT_MARGIN = 0.10;
export const SHIRT_BLEND = 0.3;
export const MAX_FACES = 4;
export const MAX_POSES = 4;
export const YAW_STRAIGHT = 12;
export const YAW_TURNED = 25;
export const SAMPLES_PER_STEP = 5;
export const SAMPLE_GAP_MS = 150;
export const NO_FACE_TIP_MS = 3000;
export const STUCK_TIP_MS = 8000;
export const SHOULDER_VISIBILITY = 0.6;
export const HIP_VISIBILITY = 0.5;
export const MIN_TORSO_PX = 40;
export const TORSO_SHRINK = 0.15;
export const MAX_SHIRT_SAMPLES = 2000;
export const STORAGE_KEY = 'videotesting.people.v1';
export const NAME_MAX = 30;
export const LABEL_COLOURS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45'];
```

- Produces `src/vision/geometry.js`:
  - `SFACE_TEMPLATE` = `[[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]]` (order: subject's right eye, left eye, nose tip, right mouth corner, left mouth corner; the subject's right is image-left in the raw frame).
  - `fivePoints(landmarks, width, height): [number, number][]`, pixel points in template order: mean of landmarks 33 and 133; mean of 362 and 263; landmark 1; landmark 61; landmark 291.
  - `estimateSimilarity(src, dst): {a, b, c, d, e, f}`: least-squares similarity transform (rotation, uniform scale, translation, no reflection) in canvas `setTransform` order, so `x' = a·x + c·y + e` and `y' = b·x + d·y + f`.
  - `yawDegrees(landmarks): number` = `atan2(z454 − z234, x454 − x234)` in degrees (positive = user's left).
  - `bboxOfPoints(points: [number, number][]): {x, y, width, height}`
  - `iou(boxA, boxB): number`
  - `containsPoint(box, [x, y]): boolean`
  - `median(values: number[]): number`

- [ ] **Step 1: Write failing tests** in `tests/unit/geometry.test.js`:
  - `estimateSimilarity`: with src = `SFACE_TEMPLATE` scaled ×2, rotated 30°, and translated by (100, 50), applying the result to every src point reproduces `SFACE_TEMPLATE` within 1e-6. Identity input gives `{a:1, b:0, c:0, d:1, e:0, f:0}` within 1e-9.
  - `yawDegrees`: build landmarks where `[234] = {x: 0.5 − 0.1·cosθ, y: 0.5, z: −0.1·sinθ}` and `[454] = {x: 0.5 + 0.1·cosθ, y: 0.5, z: 0.1·sinθ}`. θ = 0 gives 0, θ = 30° gives 30 (±0.01), θ = −30° gives −30 (±0.01).
  - `fivePoints`: with landmark 33 at (0.2, 0.4), 133 at (0.3, 0.4), and size 1000×500, point 0 is `[250, 200]`.
  - `iou`: identical boxes give 1; disjoint boxes give 0; `{0,0,10,10}` vs `{5,0,10,10}` gives 1/3 (±1e-9).
  - `containsPoint`, `bboxOfPoints`: one case each. `median([3, 1, 2])` = 2; `median([1, 2, 3, 4])` = 2.5.

- [ ] **Step 2: Run `npm test`.** Expected: FAIL.

- [ ] **Step 3: Implement both files.** `estimateSimilarity` closed form: centre both point sets on their centroids (s̄, d̄); `n = Σ(xs² + ys²)`; `A = Σ(xs·xd + ys·yd)/n`; `B = Σ(xs·yd − ys·xd)/n`; so `a = d = A`, `b = B`, `c = −B`, `e = d̄x − (A·s̄x − B·s̄y)`, `f = d̄y − (B·s̄x + A·s̄y)`.

- [ ] **Step 4: Run `npm test`.** Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat: config constants and geometry maths"` (after `git add` of the new files).

---

### Task 3: Model loading and face embeddings (feasibility gate)

This task proves the core idea works. If the correctness checks in Step 7 cannot pass, stop and report BLOCKED with the measurements. Do not fake them.

**Files:**
- Create: `src/vision/libs.js`, `src/vision/models.js`, `src/vision/faceEmbedder.js`, `models/<chosen sface>.onnx`, `models/LICENSE-sface.txt`, `scripts/reference_embeddings.py`, `tests/fixtures/faces/{a1,a2,b1}.jpg`, `tests/fixtures/faces/CREDITS.md`, `tests/fixtures/faces/reference.json`, `tests/fixtures/faces/js-embeddings.json`, `tests/e2e/harness/embed.html`
- Test: `tests/unit/faceEmbedder.test.js`, `tests/e2e/faceEmbedding.spec.js`

**Interfaces:**
- Consumes: `fivePoints`, `estimateSimilarity`, `SFACE_TEMPLATE`, `yawDegrees` (Task 2); `MAX_FACES`, `MAX_POSES`, `VERSIONS` (config).
- Produces:
  - `src/vision/libs.js`: re-exports `FaceLandmarker`, `PoseLandmarker`, `FilesetResolver` from `@mediapipe/tasks-vision@1.0.1` and `ort` from `onnxruntime-web@1.30.0` (jsDelivr ESM URLs; set `ort.env.wasm.wasmPaths` to the matching jsDelivr `dist/` folder). Verify the exact ESM file paths exist before using them.
  - `src/vision/models.js`: `SFACE_MODEL_URL` (relative `models/...onnx`), `loadModels({ runningMode = 'VIDEO' } = {}): Promise<{ face, pose, sface }>`. Face: `numFaces: MAX_FACES`, `outputFaceBlendshapes: false`, `outputFacialTransformationMatrixes: false`, model `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`. Pose: `numPoses: MAX_POSES`, model `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`. Each landmarker tries `delegate: 'GPU'` and falls back to `'CPU'` if creation throws. SFace session: `executionProviders: ['wasm']`. Any failure rejects with the original error.
  - `src/vision/faceEmbedder.js`: `alignFace(source, landmarks, width, height, canvas112)` (draws the 112×112 aligned crop), `toInputTensorData(rgba: Uint8ClampedArray): Float32Array` (pure: NCHW, **RGB**, values 0–255, no normalisation; length 3·112·112), `l2normalize(vec): Float32Array` (pure), and `embedFace(session, source, landmarks, width, height): Promise<Float32Array>` (128 values, L2-normalised; reads the input name from `session.inputNames[0]`).
  - `tests/fixtures/faces/js-embeddings.json`: `{ "a1": number[128], "a2": [...], "b1": [...] }`, produced by our JS pipeline and used by Task 9's tests.

- [ ] **Step 1: Write failing unit tests** in `tests/unit/faceEmbedder.test.js`. In a 112·112·4 RGBA buffer with pixel 0 = (10, 20, 30, 255): `out.length === 37632`, `out[0] === 10`, `out[12544] === 20`, `out[25088] === 30`. Then `l2normalize([3, 4])` gives `[0.6, 0.8]`.

- [ ] **Step 2: Run `npm test`.** Expected: FAIL. Then implement the two pure functions until it passes.

- [ ] **Step 3: Fixtures.** Download three **public-domain** portraits from Wikimedia Commons (e.g. official US government portraits): two different photos of one person (`a1.jpg`, `a2.jpg`) and one of a different person (`b1.jpg`). Pick roughly frontal faces with shoulders visible, and resize to ≤ 640 px wide. In `CREDITS.md`, record the source URL, author and public-domain status for each.

- [ ] **Step 4: Model file.** Download `face_recognition_sface_2021dec_int8.onnx` and `face_recognition_sface_2021dec.onnx` from `https://huggingface.co/opencv/face_recognition_sface/resolve/main/`. Save the licence to `models/LICENSE-sface.txt` with an attribution line (OpenCV Zoo, Apache-2.0).

- [ ] **Step 5: Reference script** `scripts/reference_embeddings.py`. Use a git-ignored venv (`python3 -m venv .venv && .venv/bin/pip install opencv-python-headless numpy`). Detect with OpenCV `FaceDetectorYN` (YuNet `face_detection_yunet_2023mar.onnx` from `https://huggingface.co/opencv/face_detection_yunet`, downloaded to a git-ignored path), align and embed with `FaceRecognizerSF` (fp32 SFace), L2-normalise, and write `tests/fixtures/faces/reference.json` as `{a1, a2, b1}`. Print the pairwise cosines. Run it and commit `reference.json`.

- [ ] **Step 6: Implement `libs.js`, `models.js`, `alignFace`, `embedFace`**, and the harness page `tests/e2e/harness/embed.html`. The harness imports the src modules and calls `loadModels({ runningMode: 'IMAGE' })`. It exposes `window.runEmbedding(url) → { embedding: number[], yaw, points, detectMs, embedMs }` for the first face, and `window.modelsReady` (a Promise).

- [ ] **Step 7: Write and run `tests/e2e/faceEmbedding.spec.js`:**
  - `cos(a1, a2) >= 0.363`, `cos(a1, b1) < 0.363`, `cos(a2, b1) < 0.363`
  - for each fixture, `cos(js, reference[name]) >= 0.9`
  - for each fixture, `points[0][0] < points[1][0]` (subject's right eye is image-left) and `Math.abs(yaw) <= 15`
  - writes `js-embeddings.json` when run with `UPDATE_FIXTURES=1`; otherwise checks that the file exists
  
  If the reference agreement fails, check channel order (RGB vs BGR) and landmark indices against OpenCV's `FaceRecognizerSF::alignCrop` and `feature()` source before anything else.

- [ ] **Step 8: Choose int8 or fp32.** Run the suite with each model. Keep int8 if every check passes **and** `cos(int8, fp32) >= 0.98` on all three fixtures; otherwise keep fp32. Commit only the chosen `.onnx`. Record both sets of numbers, plus `detectMs`/`embedMs`, in the report.

- [ ] **Step 9: Run `npm test && npm run test:e2e`.** Expected: all pass.

- [ ] **Step 10: Commit** — `git add -A && git commit -m "feat: load models and compute SFace embeddings, verified against OpenCV"`

---

### Task 4: Shirt descriptor

**Files:**
- Create: `src/recognition/shirt.js`
- Test: `tests/unit/shirt.test.js`

**Interfaces:**
- Consumes: `SHOULDER_VISIBILITY`, `HIP_VISIBILITY`, `MIN_TORSO_PX`, `TORSO_SHRINK`, `MAX_SHIRT_SAMPLES` (config); `bboxOfPoints` (geometry).
- Produces:
  - `torsoQuad(poseLandmarks, width, height): [number, number][4] | null`. Pixel corners in order [shoulder 11, shoulder 12, hip 24, hip 23]. If either hip's visibility is below `HIP_VISIBILITY`, both hips are estimated as their shoulder + `(0, shoulderWidth)`. The quad is then shrunk by `TORSO_SHRINK` towards its centroid. Returns `null` if either shoulder's visibility is below `SHOULDER_VISIBILITY`, or if the quad's bbox is narrower or shorter than `MIN_TORSO_PX`.
  - `rgbToHsv(r, g, b): [h 0–360, s 0–1, v 0–1]`
  - `shirtHistogram(rgba, width, height, quad): { histogram: Float32Array(28), swatch: string } | null`. Samples pixels inside the quad on a stride grid (≤ `MAX_SHIRT_SAMPLES`). Grey bins 0–3 are for pixels with `s < 0.2 || v < 0.15`, binned by v (`<0.25`, `<0.5`, `<0.75`, else). Every other pixel goes to bin `4 + floor(h/30)·2 + (s < 0.5 ? 0 : 1)`. The histogram is normalised to sum 1. `swatch` is the `#rrggbb` mean colour of the fullest bin's pixels. Returns `null` if no pixels were sampled.
  - `histogramSimilarity(p, q): number`, the Bhattacharyya coefficient `Σ√(pᵢ·qᵢ)`.
  - `averageHistograms(list): Float32Array`, normalised.
  - `blendHistograms(old, fresh, weight): Float32Array` = normalise((1−weight)·old + weight·fresh).

- [ ] **Step 1: Write failing tests** in `tests/unit/shirt.test.js`, using a helper that makes a solid-colour RGBA image and a full-image quad:
  - solid (255, 0, 0): bin 5 equals 1 and swatch is `#ff0000`. Solid white: bin 3 equals 1. Solid black: bin 0 equals 1.
  - red vs blue similarity `< 0.1`; red vs red `=== 1` (±1e-9); an image half red and half blue vs red ≈ 0.7071 (±0.02)
  - `torsoQuad` returns `null` when shoulder 11 has visibility 0.3
  - hips invisible, shoulders at px (100, 100) and (200, 100) on a 1000×1000 frame: the unshrunk hips sit at y = 200, so after the shrink every corner is 15% closer to the centroid (150, 150). Check corner 0 = (107.5, 107.5).
  - `blendHistograms([1,0], [0,1], 0.3)` gives `[0.7, 0.3]`

- [ ] **Step 2: Run `npm test`.** Expected: FAIL.

- [ ] **Step 3: Implement `src/recognition/shirt.js`.**

- [ ] **Step 4: Run `npm test`.** Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: shirt torso region and colour histogram"`

---

### Task 5: Matcher

**Files:**
- Create: `src/recognition/matcher.js`
- Test: `tests/unit/matcher.test.js`

**Interfaces:**
- Consumes: `FACE_MATCH`, `FACE_UNSURE`, `SHIRT_MATCH`, `SHIRT_MARGIN` (config); `histogramSimilarity` (Task 4).
- Produces:
  - `cosine(a, b): number`. Full formula; returns 0 if either vector has zero length.
  - `scorePerson(embedding, person): number`, the max cosine over `person.faceEmbeddings`, or −1 if there are none.
  - `assignFaces(embeddings: (Float32Array|null)[], people): { personId: string|null, score: number, status: 'match'|'unsure'|'unknown'|'pending' }[]`. Index-aligned with the input. A `null` embedding gives `status: 'pending'`, `personId: null`, `score: −1`. Build every (face, person) pair with score ≥ `FACE_UNSURE`, sort by score descending, and assign greedily one-to-one: `'match'` if ≥ `FACE_MATCH`, else `'unsure'`. Unassigned faces get `'unknown'` with `personId: null` and score = their best score (or −1).
  - `matchShirt(histogram, candidates: { personId, histogram }[]): { personId: string|null, score: number }`. Best must be ≥ `SHIRT_MATCH` and (best − second) ≥ `SHIRT_MARGIN`, where second is 0 if there is only one candidate. Otherwise `personId: null`. `score` is the best similarity, or 0 when there are no candidates.

- [ ] **Step 1: Write failing tests.** Helper: people embeddings are basis vectors (Alice = e1, Bob = e2 in 4-D). A face with target cosines (α to Alice, β to Bob) is `α·e1 + β·e2 + √(1−α²−β²)·e3`.
  - `cosine` of identical vectors is 1; of orthogonal vectors 0; with a zero vector 0
  - greedy: face 0 = (0.5, 0.6) and face 1 = (0.33, 0.55) → `[{ Bob, 'match', 0.6 }, { Alice, 'unsure', 0.33 }]`, because face 1 loses Bob to the higher-scoring face 0
  - a single face scoring 0.35 → `'unsure'`; 0.40 → `'match'`; 0.20 → `'unknown'` with score 0.20
  - a `null` entry → `'pending'`
  - `matchShirt`: scores 0.9 / 0.85 → `null` (margin too small); 0.9 / 0.7 → winner; single candidate 0.8 → winner; 0.7 → `null`; empty candidates → `{ personId: null, score: 0 }`

- [ ] **Step 2: Run `npm test`.** Expected: FAIL.

- [ ] **Step 3: Implement `src/recognition/matcher.js`.**

- [ ] **Step 4: Run `npm test`.** Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: face and shirt matching"`

---

### Task 6: Observations and tracker

**Files:**
- Create: `src/recognition/tracker.js`
- Test: `tests/unit/tracker.test.js`

**Interfaces:**
- Consumes: `bboxOfPoints`, `iou`, `containsPoint` (geometry); `TRACK_IOU`, `TRACK_TTL_MS`, `EMBED_RECHECK_MS` (config).
- Produces:
  - `buildObservations(faces: { landmarks }[], poses: { landmarks }[], width, height): Observation[]` where `Observation = { box, faceBox: box|null, faceIndex: number|null, poseIndex: number|null }`. Pose body box = bbox of the landmarks with visibility ≥ 0.5 (px). Face box = bbox of all face landmarks (px). Face i links to the first unclaimed pose whose nose (landmark 0, px) lies inside the face box. A linked observation's `box` is the union of the body and face boxes. An unlinked face becomes a face-only observation; an unlinked pose becomes a pose-only observation.
  - `class Tracker` with `constructor({ ttlMs = TRACK_TTL_MS, iouThreshold = TRACK_IOU } = {})`:
    - `associate(observations, now): Track[]`: greedy IoU matching (highest first, ≥ threshold) of observations to existing tracks. Unmatched observations start new tracks; tracks unseen for more than `ttlMs` are deleted. Returns the tracks seen this frame. `Track = { id: number, box, observation, personId: string|null, source: 'face'|'shirt'|null, status: 'match'|'unsure'|'unknown'|'none', score: number, lastEmbedAt: number, lastSeen: number }`. New tracks start as `personId: null, source: null, status: 'none', score: 0, lastEmbedAt: -Infinity`.
    - `needsEmbedding(track, now): boolean`: true when the track has a face this frame **and** (`status !== 'match'` or `source !== 'face'` or `now − lastEmbedAt >= EMBED_RECHECK_MS`).
    - `applyFace(trackId, { personId, score, status }, now)`: always sets `lastEmbedAt = now`. On `'match'`, sets personId/score, `source = 'face'`, `status = 'match'`. On `'unsure'`, applies only if the track has no confident (`'match'`) identity. On `'unknown'`, sets `status = 'unknown'` only if `personId` is null. `'pending'` is ignored.
    - `applyShirt(trackId, { personId, score })`: applies only when the track's `personId` is null and `personId` is non-null. Sets `source = 'shirt'`, `status = 'match'`.
    - `labels(peopleById: Map<string, { name }>): Label[]`: first resolves conflicts (when several tracks seen this frame share a personId, the winner is the face-sourced track with a face this frame, otherwise the highest score; losers reset to `personId: null, source: null, status: 'none'`). Then it returns, for tracks seen this frame, `Label = { trackId, box, personId: string|null, text }`:
      - `source === 'face'`, `status === 'match'` and a face this frame → `Name`
      - `status === 'unsure'` → `Name?`
      - any other track with a personId → `Name 👕`
      - a face this frame with no personId → `Unknown`
      - otherwise no label

- [ ] **Step 1: Write failing tests** in `tests/unit/tracker.test.js` with small synthetic boxes and observations:
  - associate: the same box next frame keeps the same id; a far box gets a new id; a track unseen at `now = last + 1001` is gone
  - follow: face match Alice gives `Alice`; next frame same body, no face, gives `Alice 👕`
  - shirt: a new track with no face plus `applyShirt` Bob gives `Bob 👕`
  - face overrides shirt: a Bob-by-shirt track, then a face match for Alice, gives `Alice`
  - **one name per person**: two tracks both carrying Alice (one face, one shirt) → only the face track is labelled `Alice`; the other has no label
  - `needsEmbedding`: after a match at t = 0 it is false at t = 500 and true at t = 1000; false with no face
  - an unknown face gives `Unknown`; unsure Alice gives `Alice?`; no face and no identity gives no label
  - `buildObservations`: a face whose box contains pose 0's nose links to pose 0, and a second face with no pose becomes face-only

- [ ] **Step 2: Run `npm test`.** Expected: FAIL.

- [ ] **Step 3: Implement `src/recognition/tracker.js`.**

- [ ] **Step 4: Run `npm test`.** Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: observations and identity tracker"`

---

### Task 7: Storage

**Files:**
- Create: `src/storage.js`
- Test: `tests/unit/storage.test.js`

**Interfaces:**
- Consumes: `STORAGE_KEY`, `NAME_MAX`, `LABEL_COLOURS` (config).
- Produces:
  - `createStore(storage = globalThis.localStorage): { available: boolean, load(): Person[], save(people): boolean }`. Never throws. `available` is false when accessing storage throws or it is missing. `load` returns `[]` for missing or corrupt JSON and drops entries without a string `name`, a non-empty `faceEmbeddings` array, and a `shirtHistogram` array. `save` returns false on failure.
  - `Person = { id: string, name: string, colour: string, faceEmbeddings: number[][], shirtHistogram: number[], shirtSwatch: string, createdAt: number }`
  - `validateName(raw, people): { ok: true, name } | { ok: false, error }`. Trims the name. Messages: `Please type a name.` / `Names can be up to 30 characters.` / `Someone called {name} is already signed up.` (case-insensitive duplicate check).
  - `newPerson({ name, faceEmbeddings, shirtHistogram, shirtSwatch }, people, now = Date.now()): Person`. `id` comes from `crypto.randomUUID()`. `colour` is the first `LABEL_COLOURS` entry not used by `people`, cycling when all are used. Embedding and histogram numbers are rounded to 4 decimals.
  - `NOT_SAVED_MESSAGE = 'Not saved on this device'`

- [ ] **Step 1: Write failing tests** with a fake `{ getItem, setItem }` object and one whose methods throw:
  - save → load round trip
  - corrupt JSON gives `[]`
  - an invalid entry is filtered out
  - with throwing storage, `available` is false, `load()` returns `[]` and `save()` returns false
  - the three `validateName` errors, and trimming
  - colour assignment skips used colours
  - rounding: `0.123456` becomes `0.1235`

- [ ] **Step 2: Run `npm test`.** Expected: FAIL.

- [ ] **Step 3: Implement `src/storage.js`.**

- [ ] **Step 4: Run `npm test`.** Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: people storage with safe fallbacks"`

---

### Task 8: Enrolment state machine

**Files:**
- Create: `src/enrol/enrolment.js`
- Test: `tests/unit/enrolment.test.js`

**Interfaces:**
- Consumes: `YAW_STRAIGHT`, `YAW_TURNED`, `SAMPLES_PER_STEP`, `SAMPLE_GAP_MS`, `NO_FACE_TIP_MS`, `STUCK_TIP_MS` (config); `median` (geometry); `averageHistograms` (Task 4).
- Produces:
  - `createEnrolment(name, now, { startAt = 'straight' } = {}): EnrolState`. `startAt: 'shirt'` is used for **Re-scan shirt**.
  - `enrolmentWantsEmbedding(state, yaw, now): boolean`: true when the current step is a face step, yaw is in range for that step, and `now − lastSampleAt >= SAMPLE_GAP_MS`.
  - `stepEnrolment(state, frame, now): EnrolState` (returns a new object), where `frame = { face: { yaw: number, embedding: Float32Array|null } | null, shirt: { histogram, swatch } | null }`.
  - `EnrolState = { name, mode: 'full'|'shirt', step: 'straight'|'left'|'right'|'shirt'|'done', progress: number (0–1, current step), prompt: string, tip: string|null, arrow: 'left'|'right'|null, yawOffset: number, result: null | { faceEmbeddings: Float32Array[], shirtHistogram: Float32Array, shirtSwatch: string } }`. It may also hold private counters (samples, timestamps).
- Rules:
  - straight accepts raw `|yaw| <= YAW_STRAIGHT`. When it completes, `yawOffset = median(straight yaws)`.
  - left accepts `yaw − yawOffset >= YAW_TURNED`; right accepts `yaw − yawOffset <= −YAW_TURNED`.
  - A face sample is taken only when the embedding is non-null, the yaw is in range, and the gap has elapsed. Each step needs `SAMPLES_PER_STEP` samples.
  - The shirt step takes any frame with `shirt` non-null (gap-limited), needs `SAMPLES_PER_STEP` samples, and averages them.
  - Steps run in order. `'done'` sets `result`: all 15 embeddings (none in shirt mode), the averaged histogram, and the swatch of the last sample.
- Exact copy:
  - Prompts: straight `Look straight at the camera`; left `Slowly turn your head LEFT`; right `Now turn RIGHT`; shirt `Step back so we can see your shirt`; done (full) `Done! Hi {name} 👋`; done (shirt mode) `Shirt updated for {name} 👕`.
  - Arrows: left step `'left'`, right step `'right'`, otherwise `null` (screen direction; the display is mirrored, so the user's left is screen-left).
  - Tips:
    - face step with no face for ≥ `NO_FACE_TIP_MS`: `Move closer, or try more light`
    - face present but no new sample for ≥ `STUCK_TIP_MS` since the step or last sample began: straight `Face the camera and hold still`; left/right `Turn a bit further`
    - shirt step with no shirt for ≥ `NO_FACE_TIP_MS`: `Step back so your shoulders are in view`
    - the tip clears when a sample is taken

- [ ] **Step 1: Write failing tests** in `tests/unit/enrolment.test.js`:
  - full walk-through (frames 200 ms apart, embedding always present): 5 frames at yaw 5 → step `'left'`, `yawOffset === 5`. Frames at yaw 28 add no sample (28 − 5 < 25). 5 frames at yaw 31 → `'right'`. 5 frames at yaw −21 (−21 − 5 = −26 ≤ −25) → `'shirt'`. 5 shirt frames → `'done'`, `result.faceEmbeddings.length === 15`, prompt `Done! Hi Alice 👋`.
  - samples 100 ms apart are not accepted (gap)
  - a null embedding is not accepted, even with the yaw in range
  - tips: no face for 3000 ms gives `Move closer, or try more light`; stuck on left for 8000 ms gives `Turn a bit further`
  - arrows: `'left'` on the left step, `'right'` on the right step
  - shirt mode: `createEnrolment('Alice', 0, { startAt: 'shirt' })` → 5 shirt frames → done with empty `faceEmbeddings` and prompt `Shirt updated for Alice 👕`
  - `enrolmentWantsEmbedding` is false out of range and true in range once the gap has elapsed

- [ ] **Step 2: Run `npm test`.** Expected: FAIL.

- [ ] **Step 3: Implement `src/enrol/enrolment.js`.**

- [ ] **Step 4: Run `npm test`.** Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: enrolment state machine"`

---

### Task 9: Live pipeline and overlay

**Files:**
- Create: `src/ui/overlay.js`, `src/app.js`
- Modify: `src/main.js`, `index.html` / `styles.css` (status, retry and debug styling)
- Test: `tests/unit/overlay.test.js`, `tests/e2e/tagging.spec.js`

**Interfaces:**
- Consumes: everything above. `loadModels` (Task 3, `runningMode: 'VIDEO'`), `embedFace`, `yawDegrees`, `buildObservations` + `Tracker`, `assignFaces`, `matchShirt`, `torsoQuad` + `shirtHistogram` + `blendHistograms`, `createStore`, `MAX_EMBEDS_PER_FRAME`, `SHIRT_BLEND`.
- Produces:
  - `src/ui/overlay.js`: `mirrorBox(box, width): box` (pure: `x' = width − x − w`), and `drawLabels(ctx, labels, { width, colourFor(personId) })`. It clears the canvas and draws each label's mirrored box plus the text above it in normal reading direction. Unknown labels are grey `#9e9e9e`.
  - `src/app.js`: `createApp({ video, overlay, models, store, onLabels, onDebug })` returning `{ start(), stop(), getPeople(), setPeople(people), processFrame(now) }`. Task 10 adds enrolment methods to this object.
  - Per frame in `processFrame`:
    1. Run `face.detectForVideo` and `pose.detectForVideo` (strictly increasing timestamps).
    2. `buildObservations`, then `tracker.associate`.
    3. Embed faces for tracks where `needsEmbedding` is true, at most `MAX_EMBEDS_PER_FRAME`, tracks with `personId === null` first.
    4. `assignFaces`, then `applyFace`.
    5. For each track still unnamed with a torso, `matchShirt` against each person's live shirt (or the stored `shirtHistogram`), excluding people already on a track this frame, then `applyShirt`.
    6. For each confident face match with a torso, update that person's in-memory live shirt with `blendHistograms(live, fresh, SHIRT_BLEND)`.
    7. `labels`, then `drawLabels`, then `onLabels(labels)`.
    
    Read pixels for the shirt histogram from an offscreen canvas holding the current frame (drawn once per frame, only when a torso is needed).
  - `src/main.js`: camera start (Task 1), then `#status` = `Loading face models…`, then `loadModels()`. On failure, `#status` = `Couldn't download the face models. Check your internet connection.` plus a `<button id="retry">Try again</button>` that retries. On success, clear the status, create the app, and drive it with `video.requestVideoFrameCallback` (falling back to `requestAnimationFrame`). Each time the set of label texts changes, update `#live-tags` with one `<li>` per label text. Show `Not saved on this device` in `#status` when `store.available` is false. With `?debug=1`, unhide `#debug` and show FPS, face/pose/embed ms, and the first face's yaw.

- [ ] **Step 1: Write failing unit test** `tests/unit/overlay.test.js`: `mirrorBox({ x: 10, y: 5, width: 20, height: 30 }, 100)` → `{ x: 70, y: 5, width: 20, height: 30 }`.

- [ ] **Step 2: Run `npm test`.** Expected: FAIL. Then implement `mirrorBox` and run again. Expected: PASS.

- [ ] **Step 3: Write e2e tests** `tests/e2e/tagging.spec.js`. Seed `localStorage[STORAGE_KEY]` via `addInitScript` with one person named `Person A`, whose `faceEmbeddings` is `[js-embeddings.a1]` and whose `shirtHistogram` is 28 values of 1/28:
  - fake camera on `a2.jpg` → `#live-tags` contains `Person A` within 30 s
  - switch the fake camera image to `b1.jpg` → `#live-tags` contains `Unknown` and not `Person A` within 10 s
  - **model download failure**: `page.route('**/mediapipe-models/**', r => r.abort())` → `#status` contains `Couldn't download the face models` and `#retry` is visible

- [ ] **Step 4: Implement `drawLabels`, `src/app.js` and the `src/main.js` changes.**

- [ ] **Step 5: Run `npm test && npm run test:e2e`.** Expected: all pass.

- [ ] **Step 6: Commit** — `git commit -m "feat: live tagging pipeline and overlay"`

---

### Task 10: Enrolment panel, People list and README

**Files:**
- Create: `src/ui/enrolPanel.js`, `src/ui/peopleList.js`
- Modify: `src/app.js`, `src/main.js`, `index.html`, `styles.css`, `README.md`
- Test: `tests/e2e/enrol.spec.js`

**Interfaces:**
- Consumes: `createEnrolment`, `stepEnrolment`, `enrolmentWantsEmbedding` (Task 8); `validateName`, `newPerson`, `createStore`, `NOT_SAVED_MESSAGE` (Task 7); the app from Task 9.
- Produces:
  - `src/app.js` additions: `startEnrolment(name, { personId } = {})` (with `personId` it runs shirt mode for an existing person), `cancelEnrolment()`, and an `onEnrolUpdate(state)` callback option. During enrolment the app uses the largest face and pose, calls `enrolmentWantsEmbedding` before embedding, and runs `stepEnrolment` each frame. On `'done'` it adds the person (`newPerson`) or replaces their shirt, then saves. Live tagging keeps running during enrolment.
  - `src/ui/peopleList.js`: `renderPeopleList(ul, people, { onDelete(id), onRescan(id) })`. One `<li>` per person: name, a `.swatch` span with background `shirtSwatch`, and buttons `Re-scan shirt` and `Delete`. Delete asks `confirm('Delete {name}?')`.
  - `src/ui/enrolPanel.js`: `createEnrolPanel(section, { onStart(name), onCancel() })` returning `{ showNameForm(), showError(message), update(state), hide() }`.
    - The name form is a labelled `<input id="enrol-name">` (label `Your name`), a `Start` button and a `Cancel` button.
    - `update` shows `prompt` in a large `#enrol-prompt`, `tip` in `#enrol-tip`, and an SVG progress ring with `role="progressbar"`, `aria-valuemin="0"`, `aria-valuemax="100"` and `aria-valuenow` = round(progress·100). The arrow `←`/`→` shows when `arrow` is set.
    - On `'done'` it shows the done prompt for 2 s, then hides.
  - `#add-person` opens the name form. Validation errors from `validateName` show in the panel.
  - `README.md`: how to use (sign-up steps, label meanings), privacy statement, running locally, testing, limitations (similar shirts, lighting, a new shirt needs **Re-scan shirt**), and an "Ideas" section noting the planned shirt-recolour fork (MediaPipe selfie multiclass segmenter's clothes class).

- [ ] **Step 1: Write e2e tests** `tests/e2e/enrol.spec.js`:
  - click `#add-person`, then submit with an empty name → `Please type a name.` is visible
  - type `Alice` and press Start → `#enrol-prompt` has `Look straight at the camera`; with the fake camera on `a1.jpg`, the progress bar's `aria-valuenow` goes above 0 within 20 s; Cancel hides the panel
  - People list: seed `Person A` → its `<li>` shows `Person A` and a `.swatch`; Delete (accept the dialog) removes it from the list and from `localStorage`
  - **storage unavailable**: an `addInitScript` makes `localStorage` access throw → `#status` contains `Not saved on this device` and the page still shows the camera

- [ ] **Step 2: Run `npm run test:e2e`.** Expected: the new tests FAIL.

- [ ] **Step 3: Implement the panel, the list, the app additions, the wiring and the README.**

- [ ] **Step 4: Run `npm test && npm run test:e2e`.** Expected: all pass.

- [ ] **Step 5: Commit** — `git commit -m "feat: enrolment panel, people list and docs"`
