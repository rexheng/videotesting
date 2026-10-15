# Face + Shirt Tagger

A small web app that uses your live webcam, entirely in the browser, to sign people up by their face and the colour of their shirt, and then tag them live as they appear on camera. Face is the main way it recognises someone. The shirt is the backup for when the face is hidden, for example when the person turns away. There is no build step.

## How to use it

Open the page and allow the camera. The page says `Loading face models…` for a few seconds. **Add person** works once that message has gone.

### Signing someone up

1. Click **Add person**, type the name in **Your name**, then press **Start** (or Enter).
2. **Look straight at the camera.** The ring fills while your face is centred.
3. **Slowly turn your head LEFT.** An arrow ← shows which way. The ring fills once you have turned far enough.
4. **Now turn RIGHT.** The arrow shows →.
5. **Step back so we can see your shirt.** Both shoulders need to be in view.
6. **Done! Hi {name} 👋**: you are saved. The panel closes by itself after 2 seconds.

If you get stuck, a tip appears under the instruction: `Move closer, or try more light`, `Face the camera and hold still`, `Turn a bit further` or `Step back so your shoulders are in view`. **Cancel** (or the Escape key) stops at any time, and nothing is saved.

Names can be up to 30 characters, and two people cannot share a name.

### The People list

Each person shows their name, a circle in their shirt colour, and two buttons:

- **Re-scan shirt**: scans only the shirt again, for example when they have changed clothes. It ends with `Shirt updated for {name} 👕`.
- **Delete**: asks `Delete {name}?` first, then removes the person.

### What the labels mean

| Label | Meaning |
|---|---|
| `Name` | Recognised by their face. |
| `Name 👕` | Their face is hidden, so the app is following them or recognising their shirt. |
| `Name?` | Not sure. The face looks a bit like this person. |
| `Unknown` | A face that nobody signed up with. |

Each person has their own label colour. Unknown faces are grey.

## Privacy

- Nothing is uploaded. The camera video, any pictures and all face data stay on this device.
- The only downloads are the app's code libraries (from jsDelivr) and the model files: the MediaPipe face and body models from Google, and the SFace face-recognition model from this site.
- People are saved only in this browser (`localStorage`). For each person it keeps the name, a label colour, 15 face "fingerprints" (lists of 128 numbers, not photos) and a shirt colour summary.
- **Delete** removes a person. Clearing this site's data in the browser removes everyone.
- If the browser blocks storage (some private windows do), the page still works and says `Not saved on this device`. People added then are forgotten when the page closes.

## Running it locally

You need Node.js 22.

```sh
npm install      # installs the Playwright test runner (the app itself has no dependencies)
npm start        # serves the app: open http://localhost:8080
```

Browsers only allow the camera on `localhost` or `https://` pages. Add `?debug=1` to the address to see the frame rate, per-model timings and head-turn angle.

## Testing

```sh
npm test         # unit tests (Node, no browser)
npm run test:e2e # browser tests (Playwright, headless Chromium, with a fake camera)
```

- The unit tests cover the pure logic: geometry and head turn, shirt colours, face and shirt matching, tracking, the sign-up steps and storage.
- The browser tests replace the webcam with a canvas that shows photos from `tests/fixtures/faces/` (see `CREDITS.md` there). They load the real models, so they need internet access. They check camera errors, live tagging, sign-up, the People list and blocked storage. `tests/e2e/pipeline.spec.js` drives the per-frame pipeline and a full sign-up with stand-in models.
- `scripts/reference_embeddings.py` computes OpenCV's own face embeddings for the fixtures. The browser tests check that ours agree with them.

## Limitations

- **Similar shirts**: the shirt backup only compares colours. Two people in similar colours (or both in dark suits) cannot be told apart by shirt, so the app leaves them unnamed rather than guess.
- **Lighting**: dim light or strong light from behind makes faces harder to find and changes how shirt colours look. The app slowly adjusts each person's shirt colour while it can see their face, but this is not saved.
- **A new shirt needs Re-scan shirt**: the saved shirt is the one from sign-up. After a change of clothes, use **Re-scan shirt**. The face still works without it.
- Up to 4 people are tagged at once, and each person has one shirt. Only the front of the shirt is scanned.
- Speed depends on the device. Older phones may manage only a few frames a second.

## Ideas

- **Change someone's shirt colour live (planned fork).** MediaPipe's selfie multiclass segmenter labels each pixel as background, hair, body skin, face skin, clothes or other. Inside the torso area from `src/recognition/shirt.js`, shift the hue of the clothes pixels but keep their brightness, so folds and shading survive. Then draw the result to a canvas instead of showing the raw video. The shirt logic already lives in its own module, which keeps that fork small.
