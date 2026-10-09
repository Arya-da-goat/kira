export const $ = id => document.getElementById(id);
export function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.classList.add('fa-icon'); svg.setAttribute('aria-hidden', 'true'); svg.setAttribute('focusable', 'false');
  const use = document.createElementNS(svg.namespaceURI, 'use');
  use.setAttribute('href', `./vendor/fontawesome.svg#fa-${name}`); svg.append(use);
  return svg;
}
export function button(name, label, action, className = 'icon-button') {
  const element = document.createElement('button'); element.type = 'button'; element.className = className;
  element.title = label; element.setAttribute('aria-label', label); element.append(icon(name));
  if (action) element.onclick = action;
  return element;
}
export function decorateIcons(root = document) {
  for (const element of root.querySelectorAll('[data-icon]')) {
    element.prepend(icon(element.dataset.icon)); element.removeAttribute('data-icon');
  }
}
export function notice(message, error = false) {
  if (error) { $('errorText').textContent = message; $('errorBanner').hidden = false; }
  else { $('generationStatus').textContent = message; $('errorBanner').hidden = true; }
}
export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); notice('Copied to clipboard.'); }
  catch { notice('Clipboard access is unavailable. Select the text and copy it.', true); }
}
export function initNavigation() {
  decorateIcons();
  for (const opener of document.querySelectorAll('[data-open]')) opener.onclick = () => $(opener.dataset.open).showModal();
  for (const closer of document.querySelectorAll('[data-close]')) closer.onclick = () => $(closer.dataset.close).close();
  const narrow = matchMedia('(max-width: 760px)');
  let mobileOpen = false;
  let collapsed = false;
  try { collapsed = localStorage.getItem('kira-sidebar-collapsed') === 'true'; } catch { /* Optional preference. */ }
  function apply() {
    const open = narrow.matches ? mobileOpen : !collapsed;
    document.body.classList.toggle('sidebar-collapsed', !narrow.matches && collapsed);
    document.body.classList.toggle('drawer-open', narrow.matches && mobileOpen);
    $('sidebar').inert = !open;
    $('mainPanel').inert = narrow.matches && mobileOpen;
    $('sidebarBackdrop').hidden = !(narrow.matches && mobileOpen);
    $('openChats').setAttribute('aria-expanded', String(open));
  }
  function close() {
    if (narrow.matches) mobileOpen = false; else collapsed = true;
    try { localStorage.setItem('kira-sidebar-collapsed', String(collapsed)); } catch { /* Optional preference. */ }
    apply(); $('openChats').focus();
  }
  $('openChats').onclick = () => { mobileOpen = true; collapsed = false; apply(); $('closeChats').focus(); };
  $('closeChats').onclick = close; $('sidebarBackdrop').onclick = close;
  $('sidebar').addEventListener('keydown', event => {
    if (!narrow.matches || !mobileOpen) return;
    if (event.key === 'Escape') { event.preventDefault(); close(); }
    if (event.key === 'Tab') {
      const items = [...$('sidebar').querySelectorAll('button:not(:disabled),input')].filter(el => el.getClientRects().length);
      const first = items[0], last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  });
  narrow.addEventListener('change', () => { mobileOpen = false; apply(); });
  $('dismissError').onclick = () => { $('errorBanner').hidden = true; };
  apply();
  return () => { if (narrow.matches) close(); };
}
