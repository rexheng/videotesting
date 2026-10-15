import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { VERSIONS } from '../../src/config.js';

const srcDir = new URL('../../src/', import.meta.url);

test('libs.js loads exactly the library versions pinned in VERSIONS', async () => {
  const source = await readFile(new URL('vision/libs.js', srcDir), 'utf8');
  const versionsOf = (pkg) => new Set([...source.matchAll(new RegExp(`${pkg}@([^/'"]+)`, 'g'))].map((m) => m[1]));

  assert.deepEqual(versionsOf('@mediapipe/tasks-vision'), new Set([VERSIONS.mediapipe]));
  assert.deepEqual(versionsOf('onnxruntime-web'), new Set([VERSIONS.ort]));
});

test('no source module other than libs.js mentions a CDN', async () => {
  const files = (await readdir(srcDir, { recursive: true })).filter((f) => f.endsWith('.js'));
  const offenders = [];
  for (const file of files) {
    if (file.replace(/\\/g, '/') === 'vision/libs.js') continue;
    const source = await readFile(new URL(file, srcDir), 'utf8');
    if (/cdn\.jsdelivr\.net|unpkg\.com|esm\.sh/.test(source)) offenders.push(file);
  }
  assert.deepEqual(offenders, []);
});
