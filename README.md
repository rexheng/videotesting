# Face + Shirt Tagger

A small web app that uses your live webcam, entirely in the browser, to sign people up by their face and the colour of their shirt, and then tag them live as they appear on camera. There is no build step.

```sh
npm install      # installs the Playwright test runner (the app itself has no dependencies)
npm start        # serves the app: open http://localhost:8080
npm test         # unit tests
npm run test:e2e # browser tests (Playwright, using a fake camera)
```
