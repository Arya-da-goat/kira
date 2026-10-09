import {$, notice} from './ui.js';
const KEY = 'kira-generation-settings-v1';
const fields = {temperature: 'temperature', top_k: 'topK', top_p: 'topP', max_new_tokens: 'maxTokens', repetition_penalty: 'penalty', seed: 'seed'};
export function generationSettings() {
  for (const id of Object.values(fields)) if (!$(id).checkValidity()) {
    $('settingsDialog').showModal(); $(id).reportValidity(); throw new Error('Check the highlighted generation setting.');
  }
  return Object.fromEntries(Object.entries(fields).map(([key, id]) => [key, Number($(id).value)]));
}
export function initSettings(onChange) {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || '{}');
    for (const [key, id] of Object.entries(fields)) if (Number.isFinite(saved[key])) {
      const original = $(id).value; $(id).value = saved[key]; if (!$(id).checkValidity()) $(id).value = original;
    }
    $('useHistory').checked = saved.use_history === true;
  } catch { /* Defaults work without browser storage. */ }
  $('settingsForm').onsubmit = event => {
    event.preventDefault();
    try { localStorage.setItem(KEY, JSON.stringify({...generationSettings(), use_history: $('useHistory').checked})); }
    catch { notice('Settings apply to this tab; browser storage is unavailable.', true); }
    $('settingsDialog').close(); onChange();
  };
  $('useHistory').onchange = onChange;
}
