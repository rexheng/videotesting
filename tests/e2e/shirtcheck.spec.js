import { test, expect } from '@playwright/test';
import { installFakeCamera, denyCamera } from './helpers/fakeCamera.js';

test('a held shirtless reading is confirmed', async ({ page }) => {
  await installFakeCamera(page);
  await page.goto('/shirtcheck.html?scripted=1');
  await expect(page.locator('#prompt')).toHaveText('Show your face. Then take the shirt off.');

  await page.locator('#primary').click();
  await expect(page.locator('#prompt')).toHaveText('Look straight at the camera.');

  await page.evaluate(() => {
    window.__shirtFrame = { face: true, reading: null };
  });
  await expect(page.locator('#prompt')).toHaveText('Take the shirt off.', { timeout: 5000 });

  await page.evaluate(() => {
    window.__shirtFrame = {
      face: true,
      reading: { verdict: 'dressed', skinRatio: 0.12, relevant: 100, sampled: 100 },
    };
  });
  await expect(page.locator('#prompt')).toHaveText('That is still a shirt.');

  await page.evaluate(() => {
    window.__shirtFrame = {
      face: true,
      faceBox: { x: 0.3, y: 0.1, width: 0.3, height: 0.3 },
      polygon: [
        [0.3, 0.4],
        [0.7, 0.4],
        [0.68, 0.85],
        [0.32, 0.85],
      ],
      reading: { verdict: 'shirtless', skinRatio: 0.86, relevant: 200, sampled: 200 },
    };
  });
  await expect(page.locator('#prompt')).toHaveText('Shirt off. Confirmed.', { timeout: 5000 });
  await expect(page.locator('#stamp')).toHaveText('OFF');
  await expect(page.locator('#tip')).toContainText('% of the chest reads as skin');
});

test('Check me calls a shirt that stays on', async ({ page }) => {
  await installFakeCamera(page);
  await page.goto('/shirtcheck.html?scripted=1');
  await page.locator('#primary').click();
  await page.evaluate(() => {
    window.__shirtFrame = {
      face: true,
      reading: { verdict: 'dressed', skinRatio: 0.1, relevant: 80, sampled: 80 },
    };
  });
  await expect(page.locator('#prompt')).toHaveText('That is still a shirt.', { timeout: 5000 });
  await expect(page.locator('#primary')).toBeEnabled({ timeout: 3000 });
  await page.locator('#primary').click();
  await expect(page.locator('#prompt')).toHaveText('Still wearing it.');
  await expect(page.locator('#stamp')).toHaveText('ON');
});

test('a blocked camera is explained', async ({ page }) => {
  await denyCamera(page);
  await page.goto('/shirtcheck.html?scripted=1');
  await page.locator('#primary').click();
  await expect(page.locator('#status')).toContainText('Camera access was blocked');
  await expect(page.locator('#prompt')).toHaveText('Show your face. Then take the shirt off.');
});

test('a bare chest reads as shirtless', async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto('/tests/e2e/harness/bare.html');
  await page.waitForFunction(() => window.modelsReady, null, { timeout: 120_000 });
  const result = await page.evaluate(() => window.classify('/tests/fixtures/shirtcheck/chest.jpg'));
  expect(result.torso).toBe(true);
  expect(result.reading.verdict).toBe('shirtless');
  expect(result.reading.skinRatio).toBeGreaterThanOrEqual(0.6);
});

test('a suited portrait reads as still dressed', async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto('/tests/e2e/harness/bare.html');
  await page.waitForFunction(() => window.modelsReady, null, { timeout: 120_000 });
  const result = await page.evaluate(() => window.classify('/tests/fixtures/faces/a1.jpg'));
  expect(result.torso).toBe(true);
  expect(result.reading.verdict).toBe('dressed');
  expect(result.reading.skinRatio).toBeLessThanOrEqual(0.4);
});
