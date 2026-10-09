import {api} from './lab.js';
import {$, icon, button, notice} from './ui.js';

export function initContext(getConversationId, isBusy) {
  let items = [];
  function render() {
    $('attachments').replaceChildren();
    for (const item of items) {
      const chip = document.createElement('div'); chip.className = 'attachment';
      const label = document.createElement('span');
      label.textContent = `${item.name} · ${item.type} · ${item.bytes.toLocaleString()} B`;
      chip.title = 'Prepared text context; sent with your next generation request.';
      const remove = button('xmark', `Remove ${item.name}`, () => {
        if (isBusy()) return;
        items = items.filter(entry => entry !== item); render();
      });
      remove.disabled = isBusy(); chip.append(icon(item.type === 'SEARCH' ? 'globe' : 'file'), label, remove);
      $('attachments').append(chip);
    }
  }
  function add(item) {
    if (items.length >= 8) throw new Error('You can attach up to 8 context items. Remove one to add another.');
    if (items.reduce((sum, entry) => sum + entry.bytes, 0) + item.bytes > 48000) throw new Error('Combined context is limited to 48 KB. Remove or shorten an attachment.');
    items.push(item); render();
  }
  $('attachFile').onclick = () => { if (!isBusy()) $('fileInput').click(); };
  $('fileInput').onchange = async event => {
    const file = event.target.files[0]; if (!file) return;
    const conversation = getConversationId();
    try {
      if (!/\.(txt|md|json|jsonl|csv|py|js|log)$/i.test(file.name) || file.size > 32768) throw new Error('Choose a text file up to 32 KiB. Images, PDFs, audio and binary files are not supported.');
      let text;
      try { text = new TextDecoder('utf-8', {fatal: true}).decode(await file.arrayBuffer()); }
      catch { throw new Error('This file is not valid UTF-8 text. Save it as UTF-8 and try again.'); }
      if (conversation !== getConversationId() || isBusy()) return;
      if (text.includes('\0')) throw new Error('This file contains binary data. Choose a UTF-8 text file.');
      add({name: file.name, type: file.name.split('.').at(-1).toUpperCase(), bytes: file.size, text: `File: ${file.name}\n${text}`});
      notice('Text attached as context. It will be sent with your next prompt; it does not train the model.');
    } catch (error) { notice(error.message, true); }
    event.target.value = '';
  };
  $('searchForm').onsubmit = async event => {
    event.preventDefault(); if (isBusy()) return;
    const query = $('searchQuery').value.trim(); if (!query) return;
    const conversation = getConversationId();
    $('webSearch').disabled = true; $('searchResults').textContent = 'Searching Wikipedia…';
    try {
      const {results} = await api('/api/tools/search', {query});
      $('searchResults').replaceChildren();
      if (!results.length) $('searchResults').textContent = 'No results found.';
      for (const result of results) {
        const article = document.createElement('article'), link = document.createElement('a');
        link.textContent = result.title;
        const url = new URL(result.url);
        if (url.protocol !== 'https:') continue;
        link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer';
        const snippet = document.createElement('p'); snippet.textContent = result.snippet;
        const attach = button('plus', 'Add result as context', () => {
          if (isBusy()) return;
          if (getConversationId() !== conversation) { notice('Search again in this conversation before attaching a result.', true); return; }
          const text = `Search result: ${result.title}\n${result.url}\n${result.snippet}`;
          try { add({name: result.title, type: 'SEARCH', bytes: new TextEncoder().encode(text).length, text}); attach.disabled = true; }
          catch (error) { notice(error.message, true); }
        }, '');
        attach.append(document.createTextNode('Add as context')); article.append(link, snippet, attach); $('searchResults').append(article);
      }
    } catch (error) { $('searchResults').textContent = error.message; }
    finally { $('webSearch').disabled = false; }
  };
  return {texts: () => items.map(item => item.text), refresh: render, clear: () => { items = []; render(); $('searchResults').replaceChildren(); }};
}
