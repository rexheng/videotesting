// Fake webcam for e2e tests: replaces getUserMedia with a 640x480 canvas stream.

/**
 * Make `navigator.mediaDevices.getUserMedia` return a 640x480 `canvas.captureStream(15)`.
 * The canvas shows `imageUrl` scaled to cover it, or solid grey when none is given.
 * The page also gets `window.__setFakeCameraImage(url)` to swap the picture later; it
 * resolves once the new image has loaded.
 */
export async function installFakeCamera(page, imageUrl) {
  await page.addInitScript((initialImageUrl) => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    let image = null;

    function draw() {
      ctx.fillStyle = '#808080';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (image) {
        // "cover": fill the canvas, keep the aspect ratio, crop the overflow.
        const scale = Math.max(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
        const w = image.naturalWidth * scale;
        const h = image.naturalHeight * scale;
        ctx.drawImage(image, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
      }
    }

    window.__setFakeCameraImage = (url) =>
      new Promise((resolve, reject) => {
        if (!url) {
          image = null;
          resolve();
          return;
        }
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          image = img;
          resolve();
        };
        img.onerror = () => reject(new Error(`Fake camera could not load image: ${url}`));
        img.src = url;
      });

    draw();
    const stream = canvas.captureStream(15);
    // Keep repainting so the stream keeps producing frames.
    setInterval(draw, 1000 / 15);
    if (initialImageUrl) window.__setFakeCameraImage(initialImageUrl).catch(console.error);

    if (!navigator.mediaDevices) {
      Object.defineProperty(navigator, 'mediaDevices', { value: {}, configurable: true });
    }
    navigator.mediaDevices.getUserMedia = async () => stream;
  }, imageUrl ?? null);
}

/** Make `getUserMedia` reject as if the user blocked camera access. */
export async function denyCamera(page) {
  await page.addInitScript(() => {
    if (!navigator.mediaDevices) {
      Object.defineProperty(navigator, 'mediaDevices', { value: {}, configurable: true });
    }
    navigator.mediaDevices.getUserMedia = () =>
      Promise.reject(new DOMException('denied', 'NotAllowedError'));
  });
}
