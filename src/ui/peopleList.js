// The People list: one row per signed-up person with their shirt colour, Re-scan shirt and Delete.
// DOM only: changes go back to the caller through onRescan and onDelete.

/**
 * Fill `ul` with one <li> per person: a .swatch in their shirt colour, their name, and the buttons
 * Re-scan shirt and Delete. Delete asks "Delete {name}?" first and calls onDelete(id) only if the
 * user agrees. With nobody signed up, a single line says so. `disabled` disables every button
 * (while someone is signing up).
 */
export function renderPeopleList(ul, people, { onDelete, onRescan, disabled = false }) {
  if (people.length === 0) {
    const empty = document.createElement('li');
    empty.className = 'people-empty';
    empty.textContent = 'Nobody has signed up yet.';
    ul.replaceChildren(empty);
    return;
  }

  ul.replaceChildren(
    ...people.map((person) => {
      const item = document.createElement('li');
      item.className = 'person';

      const swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.setAttribute('aria-hidden', 'true');
      // Invalid colours (e.g. from edited storage) are ignored by the browser.
      swatch.style.backgroundColor = person.shirtSwatch ?? '';

      const name = document.createElement('span');
      name.className = 'person-name';
      name.textContent = person.name;

      const rescan = document.createElement('button');
      rescan.type = 'button';
      rescan.className = 'button button-secondary';
      rescan.textContent = 'Re-scan shirt';
      rescan.disabled = disabled;
      rescan.setAttribute('aria-label', `Re-scan shirt for ${person.name}`);
      rescan.addEventListener('click', () => onRescan(person.id));

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'button button-danger';
      remove.textContent = 'Delete';
      remove.disabled = disabled;
      remove.setAttribute('aria-label', `Delete ${person.name}`);
      remove.addEventListener('click', () => {
        if (window.confirm(`Delete ${person.name}?`)) onDelete(person.id);
      });

      const actions = document.createElement('div');
      actions.className = 'person-actions';
      actions.append(rescan, remove);

      item.append(swatch, name, actions);
      return item;
    }),
  );
}
