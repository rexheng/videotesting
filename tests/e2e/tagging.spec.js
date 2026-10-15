// Live tagging end to end: seeded people, the fake camera, the real models and the overlay pipeline.
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { STORAGE_KEY, LABEL_COLOURS } from '../../src/config.js';
import { installFakeCamera } from './helpers/fakeCamera.js';

const embeddings = JSON.parse(readFileSync(new URL('../fixtures/faces/js-embeddings.json', import.meta.url), 'utf8'));

const personA = {
  id: 'person-a',
  name: 'Person A',
  colour: LABEL_COLOURS[0],
  faceEmbeddings: [embeddings.a1],
  shirtHistogram: Array.from({ length: 28 }, () => 1 / 28),
  shirtSwatch: '#808080',
  createdAt: 0,
};

/** Put `people` in localStorage before any page script runs. */
async function seedPeople(page, people) {
  await page.addInitScript(
    ([key, value]) => {
      window.localStorage.setItem(key, value);
    },
    [STORAGE_KEY, JSON.stringify(people)],
  );
}

test('tags a signed-up person by face, and a stranger as Unknown', async ({ page }) => {
  await seedPeople(page, [personA]);
  await installFakeCamera(page, '/tests/fixtures/faces/a2.jpg');
  await page.goto('/');

  const liveTags = page.locator('#live-tags');
  await expect(liveTags).toContainText('Person A', { timeout: 30_000 });
  await expect(page.locator('#debug')).toBeHidden();

  await page.evaluate(() => window.__setFakeCameraImage('/tests/fixtures/faces/b1.jpg'));
  await expect
    .poll(() => liveTags.locator('li').allTextContents(), { timeout: 10_000 })
    .toEqual(expect.arrayContaining(['Unknown']));
  await expect(liveTags).not.toContainText('Person A');
});

test('shows a retry button when the face models cannot be downloaded', async ({ page }) => {
  await page.route('**/mediapipe-models/**', (route) => route.abort());
  await installFakeCamera(page);
  await page.goto('/');

  await expect(page.locator('#status')).toContainText("Couldn't download the face models", { timeout: 30_000 });
  const retry = page.locator('#retry');
  await expect(retry).toBeVisible();
  await expect(retry).toHaveText('Try again');

  // "Try again" reloads the page (a failed module import cannot be retried in place).
  await Promise.all([page.waitForEvent('framenavigated'), retry.click()]);
});

test('?debug=1 shows frame rate and model timings', async ({ page }) => {
  await installFakeCamera(page, '/tests/fixtures/faces/a2.jpg');
  await page.goto('/?debug=1');

  const debug = page.locator('#debug');
  await expect(debug).toBeVisible();
  // Real numbers, not the placeholder shown while the models load. a2.jpg has a face, so yaw is set.
  await expect(debug).toContainText(/FPS \d/, { timeout: 30_000 });
  await expect(debug).toContainText(/face \d.* ms · pose \d.* ms · embed/);
  await expect(debug).toContainText(/yaw -?\d/);
});
