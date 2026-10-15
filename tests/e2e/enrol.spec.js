// Sign-up panel, People list and blocked storage, end to end with the real models and the fake
// camera. The full sign-up (all steps through to saving) is driven with stub models in
// pipeline.spec.js, because no fixture photo has a turned head.
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

/** "Add person" is enabled once the face models have loaded. */
async function waitUntilReady(page) {
  await expect(page.locator('#add-person')).toBeEnabled({ timeout: 60_000 });
}

/** Whether the camera is showing: the video has a size and is playing. */
const cameraShowing = (page) =>
  page.evaluate(() => {
    const video = document.querySelector('#camera');
    return video.videoWidth > 0 && video.paused === false;
  });

test('asks for a name before starting', async ({ page }) => {
  await installFakeCamera(page);
  await page.goto('/');
  await waitUntilReady(page);

  await page.locator('#add-person').click();
  const panel = page.locator('#enrol-panel');
  await expect(panel).toBeVisible();
  const name = page.getByLabel('Your name');
  await expect(name).toBeFocused();
  await expect(name).toHaveAttribute('id', 'enrol-name');

  await panel.getByRole('button', { name: 'Start' }).click();
  await expect(panel.getByText('Please type a name.')).toBeVisible();
  // Still on the name form.
  await expect(name).toBeVisible();

  // Enter submits a valid name; Escape closes the panel and gives "Add person" back.
  await name.fill('Bob');
  await name.press('Enter');
  await expect(page.locator('#enrol-prompt')).toHaveText('Look straight at the camera');
  await page.keyboard.press('Escape');
  await expect(panel).toBeHidden();
  await expect(page.locator('#add-person')).toBeEnabled();
  await expect(page.locator('#add-person')).toBeFocused();
});

test('signs up step by step: the ring fills while looking straight, and Cancel closes', async ({ page }) => {
  await installFakeCamera(page, '/tests/fixtures/faces/a1.jpg');
  await page.goto('/');
  await waitUntilReady(page);

  await page.locator('#add-person').click();
  await page.getByLabel('Your name').fill('Alice');
  await page.locator('#enrol-panel').getByRole('button', { name: 'Start' }).click();

  await expect(page.locator('#enrol-prompt')).toHaveText('Look straight at the camera');
  const ring = page.getByRole('progressbar');
  await expect(ring).toHaveAttribute('aria-valuemin', '0');
  await expect(ring).toHaveAttribute('aria-valuemax', '100');
  // a1.jpg is a frontal face, inside the "straight" window, so samples are taken.
  await expect
    .poll(async () => Number(await ring.getAttribute('aria-valuenow')), { timeout: 20_000 })
    .toBeGreaterThan(0);

  await page.locator('#enrol-panel').getByRole('button', { name: 'Cancel' }).click();
  await expect(page.locator('#enrol-panel')).toBeHidden();
  await expect(page.locator('#add-person')).toBeEnabled();
  // Nobody was saved.
  expect(await page.evaluate((key) => window.localStorage.getItem(key), STORAGE_KEY)).toBeNull();
});

test('People list shows each person with a shirt swatch, and Delete removes them', async ({ page }) => {
  await seedPeople(page, [personA]);
  await installFakeCamera(page);
  await page.goto('/');

  const item = page.locator('#people-list li', { hasText: 'Person A' });
  await expect(item).toBeVisible({ timeout: 60_000 });
  const swatch = item.locator('.swatch');
  await expect(swatch).toBeVisible();
  await expect(swatch).toHaveCSS('background-color', 'rgb(128, 128, 128)');
  await expect(item.getByRole('button', { name: 'Re-scan shirt' })).toBeVisible();

  const dialogs = [];
  page.on('dialog', (dialog) => {
    dialogs.push(dialog.message());
    dialog.accept();
  });
  await item.getByRole('button', { name: 'Delete' }).click();

  await expect(item).toHaveCount(0);
  expect(dialogs).toEqual(['Delete Person A?']);
  const stored = await page.evaluate((key) => JSON.parse(window.localStorage.getItem(key)), STORAGE_KEY);
  expect(stored).toEqual([]);
});

test('a failed save says people are not saved', async ({ page }) => {
  await seedPeople(page, [personA]);
  // Reading works, but every later write fails (e.g. storage full).
  await page.addInitScript(() => {
    Storage.prototype.setItem = () => {
      throw new DOMException('The quota has been exceeded.', 'QuotaExceededError');
    };
  });
  await installFakeCamera(page);
  await page.goto('/');

  const item = page.locator('#people-list li', { hasText: 'Person A' });
  await expect(item).toBeVisible({ timeout: 60_000 });
  await expect(page.locator('#status')).toHaveText('');

  page.on('dialog', (dialog) => dialog.accept());
  await item.getByRole('button', { name: 'Delete' }).click();
  await expect(item).toHaveCount(0);
  await expect(page.locator('#status')).toContainText('Not saved on this device');
});

test('storage unavailable: says people are not saved, and the camera still shows', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      get() {
        throw new DOMException('The operation is insecure.', 'SecurityError');
      },
    });
  });
  await installFakeCamera(page);
  await page.goto('/');

  // #status shows "Loading face models…" first, so this appears once the models have loaded.
  await expect(page.locator('#status')).toContainText('Not saved on this device', { timeout: 60_000 });
  await expect.poll(() => cameraShowing(page)).toBe(true);
});
