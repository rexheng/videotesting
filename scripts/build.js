// Dependency-free static build for Cloudflare Pages. Usage: node scripts/build.js  (npm run build)
// Copies only the files the site needs into dist/, and fails if any of them is over Cloudflare Pages'
// 25 MiB per-file limit.
import { cp, mkdir, readdir, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.join(root, 'dist');
const MAX_FILE_BYTES = 25 * 1024 * 1024;

// Everything else (tests/, docs/, scripts/, node_modules/, .superpowers/ ...) stays out of dist/.
const SITE_PATHS = ['index.html', 'styles.css', 'src', 'models'];

async function filesUnder(dir) {
  const found = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) found.push(...(await filesUnder(full)));
    else found.push(full);
  }
  return found;
}

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });
for (const site of SITE_PATHS) {
  await cp(path.join(root, site), path.join(dist, site), { recursive: true });
}

const sizes = [];
for (const file of await filesUnder(dist)) {
  sizes.push({ name: path.relative(dist, file).split(path.sep).join('/'), size: (await stat(file)).size });
}

const tooBig = sizes.filter(({ size }) => size > MAX_FILE_BYTES);
if (tooBig.length > 0) {
  for (const { name, size } of tooBig) {
    console.error(`dist/${name} is ${size} bytes, over the Cloudflare Pages limit of ${MAX_FILE_BYTES} bytes (25 MiB).`);
  }
  console.error('Build failed: split large files into parts (see "Deploy to Cloudflare Pages" in README.md).');
  process.exit(1);
}

const largest = sizes.reduce((a, b) => (b.size > a.size ? b : a));
console.log(`Built dist/: ${sizes.length} files. Largest: ${largest.name} (${largest.size} bytes).`);
