// The sign-up panel: a name form, then one large instruction at a time with a progress ring, an
// arrow for head turns and a tip when stuck. DOM only: the state it shows comes from the sign-up
// state machine (src/enrol/enrolment.js) via the app, and the name is checked by the caller.

import { ENROL_DONE_MS } from '../config.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const RING_RADIUS = 52;
const RING_LENGTH = 2 * Math.PI * RING_RADIUS;
const ARROWS = { left: '←', right: '→' };
const FULL_STEPS = ['straight', 'left', 'right', 'shirt'];

function el(tag, attributes = {}, text = '') {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text) node.textContent = text;
  return node;
}

function svgEl(tag, attributes) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  return node;
}

/** "Step 2 of 4" for a full sign-up; Re-scan shirt has one step, so nothing. */
function stepText(state) {
  if (state.mode !== 'full' || state.step === 'done') return '';
  return `Step ${FULL_STEPS.indexOf(state.step) + 1} of ${FULL_STEPS.length}`;
}

/**
 * Build the panel inside `section` (the hidden #enrol-panel).
 *   onStart(name)   the name form was submitted (the raw text; the caller validates it)
 *   onCancel()      Cancel (or Escape) was pressed; the panel has already hidden itself
 *   onHide(hadFocus) optional; the panel was hidden. hadFocus is true when focus was inside it.
 * Returns { showNameForm(), showError(message), update(state), hide() }.
 */
export function createEnrolPanel(section, { onStart, onCancel, onHide = () => {} }) {
  const heading = el('h2', { id: 'enrol-heading' });
  section.setAttribute('aria-labelledby', 'enrol-heading');

  // The name form.
  const form = el('form', { id: 'enrol-form', novalidate: '' });
  const label = el('label', { for: 'enrol-name' }, 'Your name');
  const input = el('input', {
    id: 'enrol-name',
    name: 'name',
    type: 'text',
    autocomplete: 'off',
    autocapitalize: 'words',
    spellcheck: 'false',
    'aria-describedby': 'enrol-error',
  });
  const error = el('p', { id: 'enrol-error', class: 'enrol-error', role: 'alert' });
  const startButton = el('button', { type: 'submit', class: 'button button-primary' }, 'Start');
  const formCancel = el('button', { type: 'button', class: 'button button-secondary' }, 'Cancel');
  const formActions = el('div', { class: 'enrol-actions' });
  formActions.append(startButton, formCancel);
  form.append(label, input, error, formActions);

  // The steps: progress ring (with the arrow inside), step count, prompt and tip.
  const steps = el('div', { id: 'enrol-steps', class: 'enrol-steps' });
  const ring = svgEl('svg', {
    class: 'enrol-ring',
    viewBox: '0 0 120 120',
    role: 'progressbar',
    'aria-label': 'Progress for this step',
    'aria-valuemin': '0',
    'aria-valuemax': '100',
    'aria-valuenow': '0',
  });
  const track = svgEl('circle', { class: 'enrol-ring-track', cx: '60', cy: '60', r: String(RING_RADIUS) });
  const fill = svgEl('circle', {
    class: 'enrol-ring-fill',
    cx: '60',
    cy: '60',
    r: String(RING_RADIUS),
    'stroke-dasharray': String(RING_LENGTH),
    'stroke-dashoffset': String(RING_LENGTH),
  });
  ring.append(track, fill);
  // The prompt already says LEFT or RIGHT, so screen readers skip the arrow.
  const arrow = el('span', { id: 'enrol-arrow', class: 'enrol-arrow', 'aria-hidden': 'true' });
  const ringWrap = el('div', { class: 'enrol-ring-wrap' });
  ringWrap.append(ring, arrow);
  const stepCount = el('p', { id: 'enrol-step-count', class: 'enrol-step-count' });
  const prompt = el('p', { id: 'enrol-prompt', class: 'enrol-prompt', 'aria-live': 'polite', tabindex: '-1' });
  const tip = el('p', { id: 'enrol-tip', class: 'enrol-tip', 'aria-live': 'polite' });
  const stepsCancel = el('button', { type: 'button', class: 'button button-secondary' }, 'Cancel');
  steps.append(ringWrap, stepCount, prompt, tip, stepsCancel);

  section.replaceChildren(heading, form, steps);
  form.hidden = true;
  steps.hidden = true;

  let doneTimer = null;
  let shownName = null; // whose sign-up the steps view shows

  function clearDoneTimer() {
    clearTimeout(doneTimer);
    doneTimer = null;
  }

  function hide() {
    clearDoneTimer();
    if (section.hidden) return;
    const hadFocus = section.contains(document.activeElement);
    section.hidden = true;
    shownName = null;
    onHide(hadFocus);
  }

  function cancel() {
    hide();
    onCancel();
  }

  function showError(message) {
    error.textContent = message;
    input.setAttribute('aria-invalid', 'true');
    input.focus();
  }

  function clearError() {
    error.textContent = '';
    input.removeAttribute('aria-invalid');
  }

  function showNameForm() {
    clearDoneTimer();
    heading.textContent = 'Add person';
    input.value = '';
    clearError();
    steps.hidden = true;
    form.hidden = false;
    section.hidden = false;
    shownName = null;
    input.focus();
  }

  /** Show a sign-up state from the app (see createEnrolment). */
  function update(state) {
    const opening = steps.hidden || section.hidden || shownName !== state.name;
    const hadFocus = section.contains(document.activeElement);
    clearDoneTimer();
    heading.textContent = state.mode === 'shirt' ? `Re-scan shirt: ${state.name}` : `Signing up ${state.name}`;
    form.hidden = true;
    steps.hidden = false;
    section.hidden = false;
    shownName = state.name;

    const percent = Math.round(Math.min(1, Math.max(0, state.progress)) * 100);
    ring.setAttribute('aria-valuenow', String(percent));
    fill.setAttribute('stroke-dashoffset', String(RING_LENGTH * (1 - percent / 100)));

    const done = state.step === 'done';
    ringWrap.classList.toggle('is-done', done);
    arrow.textContent = done ? '✓' : (ARROWS[state.arrow] ?? '');
    arrow.hidden = arrow.textContent === '';
    stepCount.textContent = stepText(state);
    stepCount.hidden = stepCount.textContent === '';
    if (prompt.textContent !== state.prompt) prompt.textContent = state.prompt;
    // The tip stays in the page when empty (CSS hides its box), so screen readers hear new tips.
    const tipText = done ? '' : (state.tip ?? '');
    if (tip.textContent !== tipText) tip.textContent = tipText;
    stepsCancel.hidden = done;

    // Focus the instruction when the steps first appear (the Start button has gone), and when
    // done hides the Cancel button, so focus stays in the panel and can be handed back on hide.
    if (opening || (done && hadFocus)) prompt.focus();
    if (done) doneTimer = setTimeout(hide, ENROL_DONE_MS);
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    onStart(input.value);
  });
  input.addEventListener('input', () => {
    if (error.textContent) clearError();
  });
  formCancel.addEventListener('click', cancel);
  stepsCancel.addEventListener('click', cancel);
  section.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !section.hidden) {
      event.preventDefault();
      cancel();
    }
  });

  return { showNameForm, showError, update, hide };
}
