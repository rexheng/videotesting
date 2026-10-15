import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';
import {
  SFACE_MODEL_PARTS,
  SFACE_MODEL_BYTES,
  SFACE_MODEL_SHA256,
  concatParts,
} from '../../src/vision/models.js';

const MAX_FILE_BYTES = 25 * 1024 * 1024; // Cloudflare Pages limit per file.
const appRoot = new URL('../../', import.meta.url);

test('concatParts joins the buffers in order', () => {
  const joined = concatParts([Uint8Array.of(1, 2).buffer, Uint8Array.of(3).buffer]);
  assert.ok(joined instanceof Uint8Array);
  assert.deepEqual(Array.from(joined), [1, 2, 3]);
});

test('concatParts of an empty list is empty', () => {
  assert.equal(concatParts([]).length, 0);
});

test('the SFace parts are two files, each within the Cloudflare Pages file limit', async () => {
  assert.equal(SFACE_MODEL_PARTS.length, 2);
  for (const part of SFACE_MODEL_PARTS) {
    const { size } = await stat(new URL(part, appRoot));
    assert.ok(size > 0 && size <= MAX_FILE_BYTES, `${part} is ${size} bytes`);
  }
});

test('the SFace parts join into exactly the verified model', async () => {
  const buffers = await Promise.all(
    SFACE_MODEL_PARTS.map(async (part) => {
      const bytes = await readFile(new URL(part, appRoot));
      return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    }),
  );
  const joined = concatParts(buffers);
  assert.equal(joined.length, SFACE_MODEL_BYTES);
  assert.equal(createHash('sha256').update(joined).digest('hex'), SFACE_MODEL_SHA256);
});
