import { test } from 'node:test';
import assert from 'node:assert/strict';
import { classifyCameraError, cameraErrorMessage } from '../../src/camera.js';

test('classifyCameraError maps permission failures to denied', () => {
  assert.equal(classifyCameraError(new DOMException('x', 'NotAllowedError')), 'denied');
  assert.equal(classifyCameraError(new DOMException('x', 'SecurityError')), 'denied');
});

test('classifyCameraError maps missing hardware to not-found', () => {
  assert.equal(classifyCameraError(new DOMException('x', 'NotFoundError')), 'not-found');
  assert.equal(classifyCameraError(new DOMException('x', 'OverconstrainedError')), 'not-found');
});

test('classifyCameraError keeps an explicit unsupported reason', () => {
  assert.equal(
    classifyCameraError(Object.assign(new Error('no api'), { reason: 'unsupported' })),
    'unsupported',
  );
});

test('classifyCameraError falls back to other', () => {
  assert.equal(classifyCameraError(new Error('boom')), 'other');
});

test('cameraErrorMessage gives the exact friendly copy for every reason', () => {
  assert.match(cameraErrorMessage('denied'), /^Camera access was blocked\./);
  assert.equal(
    cameraErrorMessage('denied'),
    "Camera access was blocked. Click the camera icon in your browser's address bar, allow the camera, then reload the page.",
  );
  assert.equal(cameraErrorMessage('not-found'), 'No camera found. Plug in a webcam and reload the page.');
  assert.equal(
    cameraErrorMessage('unsupported'),
    "This browser can't use the camera. Try the latest Chrome, Edge, Safari or Firefox.",
  );
  assert.equal(
    cameraErrorMessage('other'),
    "The camera couldn't start. Close other apps using it, then reload the page.",
  );
});

test('cameraErrorMessage treats an unknown reason as other', () => {
  assert.equal(cameraErrorMessage('???'), cameraErrorMessage('other'));
});
