# Face + Shirt Tagger — Design

Date: 2026-10-15
Status: Part 1 (user experience) approved by Rex in chat. Part 2 (technical design) decided by the orchestrator after research; Rex asked to proceed straight to building.

## 1. Goal

A web page that uses the live webcam to:

1. **Sign people up**: scan their face (straight, turned left, turned right) and their shirt.
2. **Tag them live**: draw each signed-up person's name over them in the camera view.

Face is the main way to recognise someone. The shirt is the backup for when the face is hidden (turned away, looking down, walked off and came back).

Success means: a signed-up person standing in front of the webcam gets their name shown within about a second; when they turn around, the name stays on them; a stranger shows as "Unknown"; nothing leaves the device.

## 2. What the user said vs. what we assumed

| Said by Rex | Assumed (can be changed) |
|---|---|
| Live webcam only (not uploaded video) | Several people can sign up; up to 4 people tagged at once |
| Runs in a web browser | Data saved in this browser only (localStorage) |
| Shirt = backup when the face is hidden | Front pose added to the scan, alongside left and right |
| Approach A: MediaPipe + SFace | Hosting decided later (repo may become private; GitHub Pages then needs Pro). Runs locally via `npm start` |
| Roboflow ruled out (research) | Language for copy: British English, as Rex writes it |

## 3. User experience (approved)

**Screen**: big mirrored live camera view; a People list beside it (below on phones) with an **Add person** button.

**Sign-up** — big instruction text plus a progress ring:

1. Type your name → **Start**.
2. "Look straight at the camera" — ring fills while the face is centred.
3. "Slowly turn your head LEFT" — on-screen arrow points ←; ring fills once turned far enough.
4. "Now turn RIGHT" — arrow points →.
5. "Step back so we can see your shirt" — shirt colours captured.
6. "Done! Hi {name} 👋" — saved.

Tips when stuck: "Move closer", "Try more light", "Turn a bit further", "Step back so your shoulders are in view". **Cancel** at any time.

**Live tags**:

- `Name` — recognised by face.
- `Name 👕` — following the person or recognised by shirt (face hidden).
- `Name?` — not sure.
- `Unknown` — a face nobody signed up with.

Each person has their own label colour.

**People list**: name, shirt colour swatch, **Re-scan shirt**, **Delete**.

**Privacy**: no video, images or face data are uploaded. Only model files are downloaded.

## 4. Technical design

### 4.1 Stack

- Static site: plain HTML, CSS and JavaScript ES modules. **No build step.**
- Libraries from jsDelivr with pinned versions (each at least two weeks old):
  - `@mediapipe/tasks-vision@1.0.1`: Face Landmarker (faces, 478 landmarks) and Pose Landmarker lite (bodies, 33 landmarks).
  - `onnxruntime-web@1.30.0` (wasm backend): runs SFace.
- SFace ONNX model from OpenCV Zoo (Apache-2.0), **self-hosted in `models/`**. Use the int8 version (9.9 MB) if it runs in onnxruntime-web and agrees with fp32 (cosine ≥ 0.98 on the fixtures); otherwise fp32 (38.7 MB).
- MediaPipe `.task` model files are loaded from Google's `storage.googleapis.com/mediapipe-models` URLs; wasm comes from jsDelivr.
- Dev tooling: Node 22, `node --test` for unit tests, `@playwright/test@1.56.1` for browser tests (matches the preinstalled Chromium build 1194), and a tiny dependency-free Node static server for `npm start`.

### 4.2 Per-frame pipeline

```
webcam frame
  ├─ Face Landmarker (numFaces 4) → face boxes + landmarks
  │     ├─ yaw (head turn) from landmarks
  │     └─ 5-point alignment → 112×112 crop → SFace → 128-d embedding (throttled)
  ├─ Pose Landmarker (numPoses 4) → shoulders/hips → torso → shirt colour histogram
  └─ observations (face linked to the body whose nose landmark it contains)
        → Tracker (follows bodies frame to frame, keeps names on them)
        → Overlay (labels drawn on a non-mirrored canvas, x flipped to match the mirrored video)
```

### 4.3 Recognition rules

- **Face match**: cosine similarity of L2-normalised 128-d embeddings. A person's score is the max over their stored samples. `≥ 0.363` means a match (OpenCV's published SFace threshold), `0.30–0.363` means unsure (`Name?`), below that means `Unknown`. Faces and people are paired one-to-one per frame, greedily by score.
- **Embedding cadence**: embed a face when its track has no confident identity, or every 1000 ms to re-check. At most 2 embeddings per frame.
- **Following**: tracks are linked frame to frame by box overlap (IoU ≥ 0.3). A track keeps its name while visible even without a face (`Name 👕`). Tracks unseen for over 1000 ms are dropped.
- **Shirt fallback**: an unnamed track with a torso is compared with each person's shirt histogram (Bhattacharyya coefficient). It takes a name only if the best score is ≥ 0.75 **and** beats the second best by ≥ 0.10. A face match always overrides a shirt match.
- **Live refresh**: whenever a face match is confident, that person's in-memory "live" shirt histogram is refreshed from the current frame, which handles lighting changes. It is not saved automatically.
- **One name per person**: a name is never shown on two tracks in the same frame. Face-sourced wins, then the higher score.
- All thresholds live in one `src/config.js` so they can be tuned.

### 4.4 Head turn (yaw)

- Computed from face landmarks (MediaPipe's transformation matrix has an open bug, google-ai-edge/mediapipe#4759): `yaw = atan2(z454 − z234, x454 − x234)` in degrees, using pixel-scaled x and z. **Positive = the user turned to their own left.**
- Calibrated: subtract the median yaw from the "look straight" step.
- Enrolment thresholds: straight `|yaw| ≤ 12°`; left `yaw ≥ +25°`; right `yaw ≤ −25°`.

### 4.5 Enrolment rules

- Name: trimmed, 1–30 characters, must not match an existing name (case-insensitive).
- Each face step collects 5 samples, at least 150 ms apart, while the face is in range.
- Shirt step: both shoulders visible (visibility ≥ 0.6), torso region at least 40×40 px. Collects 5 histograms and averages them.
- Tips: no face for 3 s → "Move closer, or try more light"; face present but no new sample for 8 s → "Face the camera and hold still" (straight) or "Turn a bit further" (left/right); shirt step with no shoulders for 3 s → "Step back so your shoulders are in view".
- **Re-scan shirt** runs only the shirt step for an existing person and ends with "Shirt updated for {name} 👕".

### 4.6 Shirt descriptor

- Torso quad from shoulders (landmarks 11, 12) and hips (23, 24). If hip visibility < 0.5, hips are estimated at 1.0 × shoulder width below the shoulders. The quad is shrunk 15% towards its centre.
- At most ~2000 sampled pixels. HSV histogram with 28 bins: 4 grey bins by brightness (pixels with S < 0.2 or V < 0.15), plus 12 hue × 2 saturation chromatic bins. Normalised to sum 1.
- Swatch: the mean RGB of the pixels in the fullest bin.

### 4.7 Storage

- `localStorage["videotesting.people.v1"]` holds a JSON array of `{id, name, colour, faceEmbeddings: number[][], shirtHistogram: number[], shirtSwatch, createdAt}`.
- Unavailable or corrupt storage: the app still runs and shows "Not saved on this device".

### 4.8 Errors

- Camera denied or missing: a clear message explaining how to allow the camera.
- Model download failure: a message with a **Try again** button.
- GPU delegate failure: fall back to CPU silently.

### 4.9 Testing

- Unit tests (Node) for all pure logic: geometry/yaw/alignment maths, shirt histograms, matcher, tracker, enrolment state machine, storage.
- Browser tests (Playwright, headless Chromium) with a fake camera: `getUserMedia` is overridden to stream a canvas drawing fixture images.
- **Correctness check for the face pipeline**: public-domain portrait fixtures (two photos of person A, one of person B). Same person must score ≥ 0.363 and different people < 0.363. Our JS embedding must also agree with OpenCV's own `FaceRecognizerSF` embedding of the same photo (cosine ≥ 0.9), via a Python reference script.
- `?debug=1` shows FPS and per-model timings for checking phone speed.

## 5. Out of scope (for now)

Uploaded/recorded video, cloud sync, sharing people between devices, hosting/deploy, multiple shirts per person, back-of-shirt enrolment.

## 6. Planned fork: change someone's shirt colour

Rex wants to fork this later to recolour a person's shirt live. Suggested approach: MediaPipe's selfie multiclass segmenter labels each pixel (background, hair, body skin, face skin, **clothes**, other). Inside the torso quad from `src/recognition/shirt.js`, shift the hue of the "clothes" pixels and keep their brightness so folds and shading survive, then draw the result to a canvas instead of showing the raw video. Keeping the shirt logic in its own module (as this design does) keeps that fork small.
