// People storage: load, validate and save signed-up people in localStorage, validate names typed at
// sign-up, and create new person records.
// Pure module: no DOM, no CDN imports. The storage object is a parameter, so it runs under Node.
// Storage may be missing, blocked (private mode, blocked site data) or full, so nothing here throws.
//
// Person = {
//   id: string, name: string, colour: string,
//   faceEmbeddings: number[][], shirtHistogram: number[], shirtSwatch: string, createdAt: number
// }

import { STORAGE_KEY, NAME_MAX, LABEL_COLOURS } from './config.js';

/** Shown by the UI when people cannot be saved (storage unavailable, or save() returned false). */
export const NOT_SAVED_MESSAGE = 'Not saved on this device';

/** globalThis.localStorage, or undefined when it is missing or the access itself throws. */
function defaultStorage() {
  try {
    return globalThis.localStorage;
  } catch {
    return undefined;
  }
}

/** True when the storage object has the methods we need and reading from it does not throw. */
function isUsable(storage) {
  try {
    if (!storage || typeof storage.getItem !== 'function' || typeof storage.setItem !== 'function') {
      return false;
    }
    storage.getItem(STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}

/** A stored entry we can still use: an object with a string name, face samples and a shirt histogram. */
function isValidPerson(entry) {
  return (
    entry !== null &&
    typeof entry === 'object' &&
    typeof entry.name === 'string' &&
    Array.isArray(entry.faceEmbeddings) &&
    entry.faceEmbeddings.length > 0 &&
    Array.isArray(entry.shirtHistogram)
  );
}

/**
 * Wrap a Storage-like object ({ getItem, setItem }) as a people store. Never throws.
 * `available` is false when the storage is missing or reading it throws.
 */
export function createStore(storage = defaultStorage()) {
  const available = isUsable(storage);

  return {
    available,

    /** Saved people, or [] when storage is unavailable, empty or holds corrupt data. */
    load() {
      if (!available) return [];
      try {
        const raw = storage.getItem(STORAGE_KEY);
        if (raw === null || raw === undefined) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed.filter(isValidPerson) : [];
      } catch {
        return [];
      }
    },

    /** Save everyone. Returns false (never throws) when that fails, e.g. quota exceeded. */
    save(people) {
      if (!available) return false;
      try {
        storage.setItem(STORAGE_KEY, JSON.stringify(people));
        return true;
      } catch {
        return false;
      }
    },
  };
}

/**
 * Check a name typed at sign-up. Trims it, then rejects empty, over-long and duplicate
 * (case-insensitive) names. The duplicate message uses the trimmed name as typed.
 */
export function validateName(raw, people = []) {
  const name = typeof raw === 'string' ? raw.trim() : '';
  if (name === '') return { ok: false, error: 'Please type a name.' };
  if (Array.from(name).length > NAME_MAX) {
    return { ok: false, error: `Names can be up to ${NAME_MAX} characters.` };
  }
  const wanted = name.toLowerCase();
  if (people.some((person) => typeof person.name === 'string' && person.name.trim().toLowerCase() === wanted)) {
    return { ok: false, error: `Someone called ${name} is already signed up.` };
  }
  return { ok: true, name };
}

/** Round to 4 decimals (the "+ 0" turns -0 into 0). */
function round4(value) {
  return Math.round(value * 10000) / 10000 + 0;
}

/**
 * The first label colour no one is using. When every colour is taken, the least-used one (the
 * earliest on ties), so colours cycle in order as more people sign up.
 */
function pickColour(people) {
  const uses = LABEL_COLOURS.map(() => 0);
  for (const person of people) {
    const index = LABEL_COLOURS.findIndex(
      (colour) => typeof person.colour === 'string' && colour.toLowerCase() === person.colour.toLowerCase(),
    );
    if (index !== -1) uses[index] += 1;
  }
  return LABEL_COLOURS[uses.indexOf(Math.min(...uses))];
}

/**
 * Build a new person record. Embeddings and histogram become plain number arrays rounded to
 * 4 decimals, so the record is JSON-serialisable. Inputs are not modified.
 */
export function newPerson({ name, faceEmbeddings, shirtHistogram, shirtSwatch }, people = [], now = Date.now()) {
  return {
    id: crypto.randomUUID(),
    name: typeof name === 'string' ? name.trim() : name,
    colour: pickColour(people),
    faceEmbeddings: Array.from(faceEmbeddings, (embedding) => Array.from(embedding, round4)),
    shirtHistogram: Array.from(shirtHistogram, round4),
    shirtSwatch,
    createdAt: now,
  };
}
