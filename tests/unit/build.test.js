import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const dist = path.join(root, 'dist');
const MAX_FILE_BYTES = 26_214_400; // 25 MiB, Cloudflare Pages' limit per file.

async function filesUnder(dir) {
  const found = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) found.push(...(await filesUnder(full)));
    else found.push(full);
  }
  return found;
}

test('npm run build copies only the site into dist/, every file within the size limit', async () => {
  const output = execFileSync(process.execPath, [path.join(root, 'scripts/build.js')], {
    cwd: root,
    encoding: 'utf8',
  });
  assert.match(output, /files/);

  for (const expected of [
    'index.html',
    'styles.css',
    'src/main.js',
    'models/face_recognition_sface_2021dec.onnx.part1',
    'models/face_recognition_sface_2021dec.onnx.part2',
    'models/LICENSE-sface.txt',
  ]) {
    assert.ok(existsSync(path.join(dist, expected)), `dist/${expected} should exist`);
  }
  for (const unwanted of ['tests', 'docs', 'scripts', 'node_modules', '.superpowers', 'package.json']) {
    assert.ok(!existsSync(path.join(dist, unwanted)), `dist/${unwanted} should not exist`);
  }

  const files = await filesUnder(dist);
  assert.ok(files.length > 0);
  for (const file of files) {
    const { size } = await stat(file);
    assert.ok(size <= MAX_FILE_BYTES, `${path.relative(dist, file)} is ${size} bytes`);
  }
});
