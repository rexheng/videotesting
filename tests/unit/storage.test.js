import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createStore, validateName, newPerson, withNewShirt, NOT_SAVED_MESSAGE } from '../../src/storage.js';
import { STORAGE_KEY, NAME_MAX, LABEL_COLOURS } from '../../src/config.js';

/** In-memory stand-in for localStorage with just getItem and setItem. */
function fakeStorage(initial = {}) {
  const data = new Map(Object.entries(initial));
  return {
    data,
    getItem: (key) => (data.has(key) ? data.get(key) : null),
    setItem: (key, value) => {
      data.set(key, String(value));
    },
  };
}

/** Storage whose every method throws, like a browser with blocked site data. */
function throwingStorage() {
  return {
    getItem() {
      throw new Error('storage is blocked');
    },
    setItem() {
      throw new Error('storage is blocked');
    },
  };
}

const embedding = (seed) => Array.from({ length: 128 }, (_, i) => (seed + i) / 1000);
const histogram = (seed) => Array.from({ length: 28 }, (_, i) => (seed + i) / 1000);

function samplePerson(overrides = {}) {
  return {
    id: 'id-1',
    name: 'Alice',
    colour: LABEL_COLOURS[0],
    faceEmbeddings: [embedding(1), embedding(2)],
    shirtHistogram: histogram(1),
    shirtSwatch: '#aa3322',
    createdAt: 1700000000000,
    ...overrides,
  };
}

test('config values the storage brief assumes', () => {
  assert.equal(STORAGE_KEY, 'videotesting.people.v1');
  assert.equal(NAME_MAX, 30);
  assert.equal(LABEL_COLOURS.length, 8);
  assert.equal(NOT_SAVED_MESSAGE, 'Not saved on this device');
});

test('save then load round-trips people', () => {
  const storage = fakeStorage();
  const store = createStore(storage);
  assert.equal(store.available, true);
  const people = [samplePerson(), samplePerson({ id: 'id-2', name: 'Bob', colour: LABEL_COLOURS[1] })];
  assert.equal(store.save(people), true);
  assert.ok(storage.data.has(STORAGE_KEY), 'saved under STORAGE_KEY');
  assert.deepEqual(createStore(storage).load(), people);
});

test('load returns [] when nothing has been saved', () => {
  assert.deepEqual(createStore(fakeStorage()).load(), []);
});

test('corrupt JSON loads as []', () => {
  const store = createStore(fakeStorage({ [STORAGE_KEY]: '{not json' }));
  assert.equal(store.available, true);
  assert.deepEqual(store.load(), []);
});

test('JSON that is not an array loads as []', () => {
  for (const raw of ['{"name":"Alice"}', '"text"', '42', 'null', 'true']) {
    assert.deepEqual(createStore(fakeStorage({ [STORAGE_KEY]: raw })).load(), [], raw);
  }
});

test('invalid entries are filtered out and valid ones kept', () => {
  const good = samplePerson();
  const stored = [
    good,
    samplePerson({ id: 'no-name', name: undefined }),
    samplePerson({ id: 'numeric-name', name: 42 }),
    samplePerson({ id: 'no-faces', faceEmbeddings: undefined }),
    samplePerson({ id: 'empty-faces', faceEmbeddings: [] }),
    samplePerson({ id: 'faces-not-array', faceEmbeddings: 'nope' }),
    samplePerson({ id: 'no-histogram', shirtHistogram: undefined }),
    samplePerson({ id: 'histogram-not-array', shirtHistogram: { 0: 1 } }),
    null,
    'a string',
    7,
    [],
  ];
  const store = createStore(fakeStorage({ [STORAGE_KEY]: JSON.stringify(stored) }));
  assert.deepEqual(store.load(), [good]);
});

test('with throwing storage, available is false, load() is [] and save() is false', () => {
  const store = createStore(throwingStorage());
  assert.equal(store.available, false);
  assert.deepEqual(store.load(), []);
  assert.equal(store.save([samplePerson()]), false);
});

test('a missing storage object is unavailable and never throws', () => {
  const store = createStore(null);
  assert.equal(store.available, false);
  assert.deepEqual(store.load(), []);
  assert.equal(store.save([samplePerson()]), false);
  assert.equal(createStore({}).available, false);
});

test('save returns false when writing fails (quota exceeded) but reading still works', () => {
  const storage = fakeStorage();
  storage.setItem = () => {
    throw new Error('QuotaExceededError');
  };
  const store = createStore(storage);
  assert.equal(store.available, true);
  assert.equal(store.save([samplePerson()]), false);
  assert.deepEqual(store.load(), []);
});

test('save returns false for data that cannot be serialised', () => {
  const circular = samplePerson();
  circular.self = circular;
  assert.equal(createStore(fakeStorage()).save([circular]), false);
});

test('createStore() defaults to globalThis.localStorage without throwing', () => {
  // Node has no localStorage (or may throw on access); either way this must not throw.
  const store = createStore();
  assert.equal(typeof store.available, 'boolean');
  assert.ok(Array.isArray(store.load()));
  assert.equal(typeof store.save([]), 'boolean');
});

test('createStore() copes with a localStorage getter that throws', () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    get() {
      throw new Error('SecurityError');
    },
  });
  try {
    const store = createStore();
    assert.equal(store.available, false);
    assert.deepEqual(store.load(), []);
    assert.equal(store.save([samplePerson()]), false);
  } finally {
    if (original) Object.defineProperty(globalThis, 'localStorage', original);
    else delete globalThis.localStorage;
  }
});

test('createStore() uses globalThis.localStorage when it exists', () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  const storage = fakeStorage();
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: storage });
  try {
    const store = createStore();
    assert.equal(store.available, true);
    assert.equal(store.save([samplePerson()]), true);
    assert.ok(storage.data.has(STORAGE_KEY));
  } finally {
    if (original) Object.defineProperty(globalThis, 'localStorage', original);
    else delete globalThis.localStorage;
  }
});

test('validateName: empty or blank asks for a name', () => {
  for (const raw of ['', '   ', '\t\n', undefined, null]) {
    assert.deepEqual(validateName(raw, []), { ok: false, error: 'Please type a name.' });
  }
});

test('validateName: too long', () => {
  assert.deepEqual(validateName('a'.repeat(NAME_MAX + 1), []), {
    ok: false,
    error: 'Names can be up to 30 characters.',
  });
  assert.deepEqual(validateName('a'.repeat(NAME_MAX), []), { ok: true, name: 'a'.repeat(NAME_MAX) });
});

test('validateName: padding does not count towards the length', () => {
  assert.deepEqual(validateName(`  ${'a'.repeat(NAME_MAX)}  `, []), { ok: true, name: 'a'.repeat(NAME_MAX) });
});

test('validateName: duplicates are rejected case-insensitively, using the trimmed name as typed', () => {
  const people = [samplePerson({ name: 'Alice' })];
  assert.deepEqual(validateName('  aLiCe ', people), {
    ok: false,
    error: 'Someone called aLiCe is already signed up.',
  });
  assert.deepEqual(validateName('ALICE', people), {
    ok: false,
    error: 'Someone called ALICE is already signed up.',
  });
});

test('validateName: trims a good name', () => {
  assert.deepEqual(validateName('  Bob  ', [samplePerson()]), { ok: true, name: 'Bob' });
});

test('newPerson builds a person with the documented shape', () => {
  const faces = [embedding(1), embedding(2)];
  const person = newPerson(
    { name: 'Carol', faceEmbeddings: faces, shirtHistogram: histogram(3), shirtSwatch: '#112233' },
    [],
    1234,
  );
  assert.deepEqual(Object.keys(person).sort(), [
    'colour',
    'createdAt',
    'faceEmbeddings',
    'id',
    'name',
    'shirtHistogram',
    'shirtSwatch',
  ]);
  assert.equal(typeof person.id, 'string');
  assert.ok(person.id.length > 0);
  assert.equal(person.name, 'Carol');
  assert.equal(person.colour, LABEL_COLOURS[0]);
  assert.equal(person.shirtSwatch, '#112233');
  assert.equal(person.createdAt, 1234);
  assert.equal(person.faceEmbeddings.length, 2);
  assert.equal(person.faceEmbeddings[0].length, 128);
  assert.equal(person.shirtHistogram.length, 28);
});

test('newPerson gives each person a different id', () => {
  const input = { name: 'A', faceEmbeddings: [embedding(1)], shirtHistogram: histogram(1), shirtSwatch: '#000000' };
  assert.notEqual(newPerson(input, []).id, newPerson(input, []).id);
});

test('newPerson defaults createdAt to the current time', () => {
  const before = Date.now();
  const person = newPerson(
    { name: 'A', faceEmbeddings: [embedding(1)], shirtHistogram: histogram(1), shirtSwatch: '#000000' },
    [],
  );
  assert.ok(person.createdAt >= before && person.createdAt <= Date.now());
});

test('newPerson rounds embeddings and histograms to 4 decimals', () => {
  const person = newPerson(
    {
      name: 'Dan',
      faceEmbeddings: [[0.123456, -0.123456, 0.00004, 1]],
      shirtHistogram: [0.123456, 0.5],
      shirtSwatch: '#000000',
    },
    [],
  );
  assert.deepEqual(person.faceEmbeddings, [[0.1235, -0.1235, 0, 1]]);
  assert.deepEqual(person.shirtHistogram, [0.1235, 0.5]);
});

test('newPerson turns Float32Arrays into plain, JSON-serialisable number arrays', () => {
  const person = newPerson(
    {
      name: 'Eve',
      faceEmbeddings: [Float32Array.from([0.123456, 0.5])],
      shirtHistogram: Float32Array.from([0.25, 0.123456]),
      shirtSwatch: '#000000',
    },
    [],
  );
  assert.ok(Array.isArray(person.faceEmbeddings));
  assert.ok(Array.isArray(person.faceEmbeddings[0]));
  assert.ok(Array.isArray(person.shirtHistogram));
  assert.deepEqual(person.faceEmbeddings, [[0.1235, 0.5]]);
  assert.deepEqual(person.shirtHistogram, [0.25, 0.1235]);
  assert.deepEqual(JSON.parse(JSON.stringify(person)), person);
});

test('newPerson does not change its inputs', () => {
  const faces = [[0.123456, 0.2]];
  const hist = [0.123456];
  const people = [samplePerson()];
  newPerson({ name: 'Fay', faceEmbeddings: faces, shirtHistogram: hist, shirtSwatch: '#000000' }, people);
  assert.deepEqual(faces, [[0.123456, 0.2]]);
  assert.deepEqual(hist, [0.123456]);
  assert.equal(people.length, 1);
});

test('newPerson trims the name', () => {
  const person = newPerson(
    { name: '  Gus  ', faceEmbeddings: [embedding(1)], shirtHistogram: histogram(1), shirtSwatch: '#000000' },
    [],
  );
  assert.equal(person.name, 'Gus');
});

test('colour assignment takes the first colour not used by anyone', () => {
  const input = { name: 'New', faceEmbeddings: [embedding(1)], shirtHistogram: histogram(1), shirtSwatch: '#000000' };
  assert.equal(newPerson(input, []).colour, LABEL_COLOURS[0]);
  assert.equal(newPerson(input, [samplePerson({ colour: LABEL_COLOURS[0] })]).colour, LABEL_COLOURS[1]);
  // A gap is filled before moving on.
  const gap = [samplePerson({ colour: LABEL_COLOURS[0] }), samplePerson({ colour: LABEL_COLOURS[2] })];
  assert.equal(newPerson(input, gap).colour, LABEL_COLOURS[1]);
  // The order people were added in does not matter.
  const shuffled = [samplePerson({ colour: LABEL_COLOURS[1] }), samplePerson({ colour: LABEL_COLOURS[0] })];
  assert.equal(newPerson(input, shuffled).colour, LABEL_COLOURS[2]);
});

test('colour assignment cycles once every colour is used', () => {
  const input = { name: 'New', faceEmbeddings: [embedding(1)], shirtHistogram: histogram(1), shirtSwatch: '#000000' };
  const people = [];
  for (let i = 0; i < LABEL_COLOURS.length * 2 + 3; i++) {
    const person = newPerson({ ...input, name: `P${i}` }, people);
    assert.equal(person.colour, LABEL_COLOURS[i % LABEL_COLOURS.length], `person ${i}`);
    people.push(person);
  }
});

test('withNewShirt returns a new person with the new shirt, rounded like newPerson, and the rest unchanged', () => {
  const person = samplePerson({ name: 'Hana', shirtSwatch: '#000000' });
  const before = JSON.parse(JSON.stringify(person));
  const shirt = Float32Array.from([0.123456, 0.25]);
  const updated = withNewShirt(person, { shirtHistogram: shirt, shirtSwatch: '#00ffff' });

  assert.notEqual(updated, person);
  assert.ok(Array.isArray(updated.shirtHistogram));
  assert.deepEqual(updated.shirtHistogram, [0.1235, 0.25]);
  assert.equal(updated.shirtSwatch, '#00ffff');
  const { shirtHistogram, shirtSwatch, ...rest } = updated;
  const { shirtHistogram: oldHistogram, shirtSwatch: oldSwatch, ...oldRest } = before;
  assert.deepEqual(rest, oldRest);
  assert.equal(updated.faceEmbeddings, person.faceEmbeddings);
  // Inputs are not changed.
  assert.deepEqual(person, before);
  assert.deepEqual(Array.from(shirt), [Math.fround(0.123456), 0.25]);
});
