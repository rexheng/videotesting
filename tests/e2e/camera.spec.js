import { test, expect } from '@playwright/test';
import { installFakeCamera, denyCamera } from './helpers/fakeCamera.js';

test('shows the camera', async ({ page }) => {
  await installFakeCamera(page);
  await page.goto('/');

  await expect
    .poll(() =>
      page.evaluate(() => {
        const video = document.querySelector('#camera');
        return video.videoWidth > 0 && video.paused === false;
      }),
    )
    .toBe(true);

  // The overlay's drawing surface is sized to the video once the camera is running.
  await expect
    .poll(() =>
      page.evaluate(() => {
        const video = document.querySelector('#camera');
        const overlay = document.querySelector('#overlay');
        return [overlay.width, overlay.height, video.videoWidth, video.videoHeight];
      }),
    )
    .toEqual([640, 480, 640, 480]);
  await expect(page.locator('#status')).toHaveText('');
});

test('shows blocked message', async ({ page }) => {
  await denyCamera(page);
  await page.goto('/');

  await expect(page.locator('#status')).toContainText('Camera access was blocked');
});
