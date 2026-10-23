# Face + Shirt Tagger

A small web app that uses your live webcam, entirely in the browser, to sign people up by their face and the colour of their shirt, and then tag them live as they appear on camera. Face is the main way it recognises someone. The shirt is the backup for when the face is hidden, for example when the person turns away. There is no bundler: the browser runs the source files as they are.

Shirt Check (`shirtcheck.html`, or **Play Shirt Check** on the tagger page) is a separate game on the same camera. It finds your face, tells you to take the shirt off, and then says whether the chest reads as skin or as cloth. It does not recognise who you are.

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

### Shirt Check

Open `shirtcheck.html` and press **Start**. Allow the camera.

1. **Look straight at the camera.** A bar fills while your face stays in view.
2. **Take the shirt off.** A meter slides from **Cloth** towards **Skin** as the chest changes. Shoulders need to be in frame.
3. Hold still once it reads as skin. After about a second it stamps **OFF** and says `Shirt off. Confirmed.`
4. **Check me** calls it early from the last few readings. If the shirt is still on, the stamp is **ON** and the line is `Still wearing it.`
5. The round lasts 25 seconds. **Quit** gives up. **Try again** starts over.

`Step back so your shoulders are in view` means the torso is not readable. `Where did you go?` means the face left the frame. `Not enough yet` is the in-between case (a vest, an open shirt). Skin-coloured clothes can fool it, because it is comparing skin pixels with cloth pixels, not understanding fashion.

Nothing from this game is saved. **Quit** or closing the page drops the round.

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

## Deploy to Cloudflare Pages

The app is a static site, so it deploys to Cloudflare Pages straight from GitHub:

1. In the Cloudflare dashboard, go to **Workers & Pages** → **Create** → **Pages** → **Import an existing Git repository**.
2. Connect GitHub and pick `rexheng/videotesting` (private repositories work too).
3. Set **Build command** to `npm run build` and **Build output directory** to `dist`. Leave **Production branch** as `main`.
4. Click **Save and Deploy**.

Every pushed branch gets its own preview URL, and the production URL updates when `main` changes. Pages serves everything over HTTPS, which the camera needs.

`npm run build` (`scripts/build.js`, no dependencies) deletes and recreates `dist/` and copies only `index.html`, `shirtcheck.html`, `styles.css`, `shirtcheck.css`, `src/` and `models/` into it. It prints the number of files and the largest one, and fails if any file is over 25 MiB (26,214,400 bytes), which is Cloudflare Pages' limit per file.

### Why the face model is in two parts

The SFace model is 38,696,353 bytes, which is over that limit, so it is stored as two files and the browser joins them (`concatParts` in `src/vision/models.js`). The browser also checks the joined size and its SHA-256 (`0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`) before using it. The parts were made from the original `face_recognition_sface_2021dec.onnx` (OpenCV Zoo) with:

```sh
head -c 20971520  face_recognition_sface_2021dec.onnx > face_recognition_sface_2021dec.onnx.part1   # first 20 MiB
tail -c +20971521 face_recognition_sface_2021dec.onnx > face_recognition_sface_2021dec.onnx.part2   # the other 17,724,833 bytes
```

To rebuild the original file, join them in order: `cat face_recognition_sface_2021dec.onnx.part1 face_recognition_sface_2021dec.onnx.part2 > face_recognition_sface_2021dec.onnx`.

## Testing

```sh
npm test         # unit tests (Node, no browser)
npm run test:e2e # browser tests (Playwright, headless Chromium, with a fake camera)
```

- The unit tests cover the pure logic: geometry and head turn, shirt colours, face and shirt matching, tracking, the sign-up steps and storage. They also check that the two SFace parts join into the verified model and that `npm run build` produces a deployable `dist/`.
- The browser tests replace the webcam with a canvas that shows photos from `tests/fixtures/faces/` (see `CREDITS.md` there). They load the real models, so they need internet access. They check camera errors, live tagging, sign-up, the People list and blocked storage. `tests/e2e/pipeline.spec.js` drives the per-frame pipeline and a full sign-up with stand-in models.
- `scripts/reference_embeddings.py` computes OpenCV's own face embeddings for the fixtures. The browser tests check that ours agree with them.

## Limitations

- **Similar shirts**: the shirt backup only compares colours. Two people in similar colours (or both in dark suits) cannot be told apart by shirt, so the app leaves them unnamed rather than guess.
- **Lighting**: dim light or strong light from behind makes faces harder to find and changes how shirt colours look. The app slowly adjusts each person's shirt colour while it can see their face, but this is not saved.
- **A new shirt needs Re-scan shirt**: the saved shirt is the one from sign-up. After a change of clothes, use **Re-scan shirt**. The face still works without it.
- **Shirt Check can be fooled by skin-coloured cloth**, heavy tattoos, jewellery, or hair and a jacket over the chest. It only looks at the torso between the shoulders and the waist. A plain bare chest and a suit separate cleanly. The skin-or-cloth model runs on the CPU so its labels stay in the right order.
- Up to 4 people are tagged at once, and each person has one shirt. Only the front of the shirt is scanned.
- Speed depends on the device. Older phones may manage only a few frames a second.

## Ideas

- **Change someone's shirt colour live (planned fork).** MediaPipe's selfie multiclass segmenter labels each pixel as background, hair, body skin, face skin, clothes or other. Inside the torso area from `src/recognition/shirt.js`, shift the hue of the clothes pixels but keep their brightness, so folds and shading survive. Then draw the result to a canvas instead of showing the raw video. The shirt logic already lives in its own module, which keeps that fork small.
