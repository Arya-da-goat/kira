import {api, initLab} from './lab.js';
import {plotLoss} from './chart.js';
const $ = (id) => document.getElementById(id);
const STORAGE = 'kira-local-lab-v2';
let conversations = [];
try {
  const stored = JSON.parse(localStorage.getItem(STORAGE) || '[]');
  if (Array.isArray(stored)) conversations = stored.filter(item => typeof item.id === 'string' && Array.isArray(item.messages));
} catch { /* A corrupt or unavailable browser store starts a fresh session. */ }
let current;
let busy = false;
let ready = false;
let modelStatus = {};
let contexts = [];

function persist() {
  try { localStorage.setItem(STORAGE, JSON.stringify(conversations)); }
  catch { $('generationStatus').textContent = 'Browser storage is full or unavailable. Export to keep this conversation.'; }
}
function freshChat() {
  if (busy) return;
  current = {id: crypto.randomUUID(), messages: [], memory: ''};
  conversations.unshift(current);
  contexts = [];
  $('attachments').textContent = '';
  $('memory').value = '';
  persist(); render();
}
function renderThreads() {
  $('threadList').replaceChildren();
  const query = $('searchChats').value.toLowerCase();
  for (const chat of conversations) {
    const title = chat.messages.find(message => message.role === 'user')?.content || 'New conversation';
    if (!chat.messages.some(message => String(message.content).toLowerCase().includes(query)) && !title.toLowerCase().includes(query)) continue;
    const button = document.createElement('button');
    button.textContent = title.slice(0, 60);
    button.classList.toggle('active', chat.id === current.id);
    button.disabled = busy;
    button.onclick = () => {
      current = chat; contexts = []; $('attachments').textContent = '';
      $('memory').value = current.memory || ''; render(); closeChats();
    };
    $('threadList').append(button);
  }
}
function render() {
  renderThreads();
  $('chat').replaceChildren();
  if (!current.messages.length) {
    const welcome = document.createElement('div');
    welcome.className = 'welcome';
    const heading = document.createElement('h1'); heading.textContent = 'Your model. Your training.';
    const description = document.createElement('p');
    description.textContent = 'Kira runs a text Transformer from local weights. The example dataset is a software test, not language pretraining. Expect poor output until you train on a suitable corpus.';
    welcome.append(heading, description); $('chat').append(welcome);
  }
  for (const message of current.messages) {
    const article = document.createElement('article'); article.className = `message ${message.role === 'user' ? 'user' : 'assistant'}`;
    const header = document.createElement('header'); header.textContent = message.role === 'user' ? 'You' : 'Kira · local generation';
    const text = document.createElement('div'); text.className = 'text'; text.textContent = message.content || '(No visible text generated)';
    const footer = document.createElement('footer');
    if (message.metrics) {
      const m = message.metrics;
      footer.textContent = `${m.generated_tokens} tokens · ${m.tokens_per_second.toFixed(1)} tokens/sec · ${m.prompt_tokens} context tokens · stop: ${m.stop_reason}`;
    }
    const copy = document.createElement('button'); copy.textContent = 'Copy';
    copy.onclick = async () => {
      try { await navigator.clipboard.writeText(message.content); copy.textContent = 'Copied'; }
      catch { $('generationStatus').textContent = 'Clipboard unavailable; select and copy the message text.'; }
    };
    footer.append(copy); article.append(header, text, footer); $('chat').append(article);
  }
  $('chat').scrollTop = $('chat').scrollHeight;
}
async function refreshStatus() {
  try {
    modelStatus = await api('/api/status'); ready = modelStatus.ready;
    $('modelStatus').textContent = ready
      ? `${modelStatus.training_status} · step ${modelStatus.training_step} · ${modelStatus.parameters.toLocaleString()} parameters · context ${modelStatus.config.max_seq_len} · ${modelStatus.hardware.device.toUpperCase()}. Quality is not established by training steps.`
      : modelStatus.detail;
    return true;
  } catch (error) {
    ready = false;
    $('modelStatus').textContent = `Backend disconnected. Open Connect to set it up. ${error.message}`;
    return false;
  } finally {
    $('sendBtn').disabled = busy || !ready;
  }
}
$('composer').onsubmit = async (event) => {
  event.preventDefault();
  if (busy || !ready || !$('prompt').value.trim()) return;
  const prompt = $('prompt').value;
  const history = $('useHistory').checked ? current.messages.map(({role, content}) => ({role, content})) : [];
  const request = {prompt, messages: history, memory: current.memory || '', context: [...contexts],
    temperature: Number($('temperature').value), top_k: Number($('topK').value),
    top_p: Number($('topP').value), max_new_tokens: Number($('maxTokens').value),
    repetition_penalty: Number($('penalty').value), seed: Number($('seed').value)};
  busy = true; $('sendBtn').disabled = true; $('newChat').disabled = true; $('deleteChat').disabled = true;
  $('generationStatus').textContent = 'Generating…';
  current.messages.push({role: 'user', content: prompt}); persist(); render();
  try {
    const result = await api('/api/chat', request);
    current.messages.push({role: 'assistant', content: result.text, metrics: {
      generated_tokens: result.generated_tokens, tokens_per_second: result.tokens_per_second,
      prompt_tokens: result.prompt_tokens, stop_reason: result.stop_reason
    }});
    $('prompt').value = '';
    $('generationStatus').textContent = `Generated ${result.generated_tokens} tokens in ${result.seconds.toFixed(2)}s`;
  } catch (error) {
    // An HTTP failure is a UI error, never an invented model reply.
    current.messages.pop();
    $('generationStatus').textContent = error.message;
  } finally {
    busy = false; $('sendBtn').disabled = !ready; $('newChat').disabled = false; $('deleteChat').disabled = false;
    persist(); render();
  }
};
$('prompt').onkeydown = (event) => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('composer').requestSubmit(); }
};
function closeChats() { $('sidebar').classList.remove('is-open'); $('openChats').setAttribute('aria-expanded', 'false'); }
$('openChats').onclick = () => { $('sidebar').classList.add('is-open'); $('openChats').setAttribute('aria-expanded', 'true'); };
$('closeChats').onclick = closeChats;
$('newChat').onclick = () => { freshChat(); closeChats(); };
$('searchChats').oninput = renderThreads;
$('memory').oninput = () => { current.memory = $('memory').value; persist(); };
$('deleteChat').onclick = () => {
  if (busy) return;
  conversations = conversations.filter(chat => chat.id !== current.id);
  if (!conversations.length) freshChat();
  else { current = conversations[0]; contexts = []; $('attachments').textContent = ''; $('memory').value = current.memory || ''; persist(); render(); }
};
$('exportChat').onclick = () => {
  const blob = new Blob([JSON.stringify(current, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob); const link = document.createElement('a');
  link.href = url; link.download = 'kira-conversation.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
function addContext(text, label) {
  if (contexts.length >= 8) throw new Error('At most 8 context items are supported. Clear attachments first.');
  contexts.push(text); $('attachments').textContent += `${label}\n`;
}
$('fileInput').onchange = async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const chatId = current.id;
  try {
    if (!/\.(txt|md|json|jsonl|csv|py|js|log)$/i.test(file.name) || file.size > 32768) throw new Error('Attach a supported text file under 32 KiB. Images are not supported.');
    const text = await file.text();
    if (current.id !== chatId) return;
    if (text.includes('\0')) throw new Error('This file contains binary data. Attach UTF-8 text.');
    addContext(`File: ${file.name}\n${text}`, `File: ${file.name} (${file.size} bytes)`);
  } catch (error) { $('generationStatus').textContent = error.message; }
  event.target.value = '';
};
$('clearContext').onclick = () => { contexts = []; $('attachments').textContent = ''; };
$('webSearch').onclick = async () => {
  const query = $('searchQuery').value.trim(); if (!query) return;
  $('webSearch').disabled = true; $('searchResults').textContent = 'Searching Wikipedia…';
  try {
    const data = await api('/api/tools/search', {query}); $('searchResults').replaceChildren();
    if (!data.results.length) $('searchResults').textContent = 'No results.';
    for (const result of data.results) {
      const article = document.createElement('article'); const link = document.createElement('a');
      link.textContent = result.title; link.href = result.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
      const snippet = document.createElement('p'); snippet.textContent = result.snippet;
      const attach = document.createElement('button'); attach.textContent = 'Add as model context';
      attach.onclick = () => {
        try { addContext(`Search result: ${result.title}\n${result.url}\n${result.snippet}`, `Search: ${result.title}`); attach.disabled = true; }
        catch (error) { $('generationStatus').textContent = error.message; }
      };
      article.append(link, snippet, attach); $('searchResults').append(article);
    }
  } catch (error) { $('searchResults').textContent = error.message; }
  finally { $('webSearch').disabled = false; }
};
$('inspectModel').onclick = async () => {
  $('modelDialog').showModal(); await refreshStatus();
  $('modelDetails').textContent = JSON.stringify(modelStatus, null, 2);
  try { plotLoss((await api('/api/metrics')).history, $('lossChart'), $('chartLegend')); }
  catch (error) { $('chartLegend').textContent = error.message; }
};
$('closeModel').onclick = () => $('modelDialog').close();
if (conversations.length) { current = conversations[0]; $('memory').value = current.memory || ''; render(); }
else freshChat();
initLab(refreshStatus, () => busy);
