import {api, initLab} from './lab.js';
import {plotLoss} from './chart.js';
import {$, icon, initNavigation, notice} from './ui.js';
import {loadConversations, saveConversations, newConversation, memoryContext, precedingHistory, replaceTurn} from './conversations.js';
import {createMessageView} from './messages.js';
import {initContext} from './context.js';
import {initSettings, generationSettings} from './settings.js';

let conversations = [];
try { conversations = loadConversations(localStorage); } catch { /* Private browsing may disable storage. */ }
if (!conversations.length) conversations = [newConversation()];
let current = conversations[0], busy = false, ready = false, modelStatus = {}, editing = null;
const closeDrawer = initNavigation();
const context = initContext(() => current.id, () => busy);
const messages = createMessageView($('messages'), {edit, regenerate});

function persist() {
  try { saveConversations(localStorage, conversations); }
  catch { notice('Browser storage is full or unavailable. Export to keep this conversation.', true); }
}
function updateControls() {
  $('sendBtn').disabled = busy || !ready;
  for (const id of ['newChat','deleteChat','attachFile']) $(id).disabled = busy;
  $('prompt').readOnly = busy;
  $('generating').hidden = !busy;
  $('composer').setAttribute('aria-busy', String(busy));
  for (const element of document.querySelectorAll('[data-model-action]')) element.disabled = busy || !ready;
  for (const element of $('threadList').querySelectorAll('button')) element.disabled = busy;
  context.refresh();
}
function renderThreads() {
  $('threadList').replaceChildren();
  const query = $('searchChats').value.toLowerCase();
  for (const chat of conversations) {
    const title = chat.messages.find(message => message.role === 'user')?.content || 'New conversation';
    if (query && !chat.messages.some(message => message.content.toLowerCase().includes(query)) && !title.toLowerCase().includes(query)) continue;
    const button = document.createElement('button'), label = document.createElement('span');
    label.textContent = title.slice(0, 70); button.title = title.slice(0, 200);
    button.append(icon('message'), label); button.classList.toggle('active', chat.id === current.id);
    if (chat.id === current.id) button.setAttribute('aria-current', 'page');
    button.disabled = busy;
    button.onclick = () => { if (!busy) selectConversation(chat); };
    $('threadList').append(button);
  }
  if (!$('threadList').children.length) { const empty = document.createElement('p'); empty.className = 'empty-history'; empty.textContent = 'No matching conversations.'; $('threadList').append(empty); }
}
function renderConversation(scroll = false) {
  $('welcome').hidden = current.messages.length > 0 || busy;
  messages.sync(current.messages); renderThreads(); updateControls();
  if (scroll) requestAnimationFrame(() => { $('chat').scrollTop = $('chat').scrollHeight; });
}
function updateContextHint() {
  const labels = [$('useHistory').checked ? 'History included' : 'Text completion'];
  if (current.memoryEnabled && current.memory) labels.push('Memory on');
  $('contextHint').textContent = labels.join(' · ');
}
function selectConversation(chat) {
  current = chat; context.clear(); editing = null; $('editBar').hidden = true;
  $('memory').value = current.memory; $('memoryEnabled').checked = current.memoryEnabled;
  $('prompt').value = ''; resizeComposer(); messages.reset(); updateContextHint(); renderConversation(true); closeDrawer();
}
function statusLabel(state, label) { $('statusPill').dataset.state = state; $('statusText').textContent = label; }
async function refreshStatus() {
  try {
    modelStatus = await api('/api/status');
    ready = Boolean(modelStatus.ready && !modelStatus.training_active && !modelStatus.busy);
    if (modelStatus.busy || busy) statusLabel('generating', 'Model busy');
    else if (modelStatus.training_active) statusLabel('training', 'Training');
    else statusLabel(modelStatus.ready ? 'ready' : 'unloaded', modelStatus.ready ? 'Model loaded' : 'No weights');
    $('modelStatus').textContent = modelStatus.training_active
      ? 'Training is running. Chat becomes available when the worker finishes or is stopped.'
      : modelStatus.detail || `Loaded on ${modelStatus.hardware?.device?.toUpperCase()} · ${modelStatus.config?.max_seq_len} token context · quality unverified`;
    $('connectionBanner').hidden = ready || busy;
    $('sidebarModelValue').textContent = modelStatus.ready
      ? `${modelStatus.hardware.device.toUpperCase()} · ${modelStatus.config.max_seq_len} context · step ${modelStatus.training_step}`
      : modelStatus.busy ? 'Model operation in progress' : modelStatus.training_active ? 'Training in progress' : 'No checkpoint loaded';
    return true;
  } catch (error) {
    ready = false; statusLabel('offline', 'Backend offline'); $('connectionBanner').hidden = false;
    $('sidebarModelValue').textContent = 'Backend disconnected';
    $('modelStatus').textContent = error.message;
    modelStatus = {ready: false, detail: error.message};
    return false;
  } finally { updateControls(); }
}
function resizeComposer() { $('prompt').style.height = 'auto'; $('prompt').style.height = `${Math.min($('prompt').scrollHeight, 160)}px`; }
async function submit(prompt, index = current.messages.length) {
  if (busy || !ready || !prompt.trim()) return;
  let request;
  try { request = {prompt, messages: precedingHistory(current, index, $('useHistory').checked), memory: memoryContext(current), context: context.texts(), ...generationSettings()}; }
  catch (error) { notice(error.message, true); return; }
  const followScroll = $('chat').scrollHeight - $('chat').scrollTop - $('chat').clientHeight < 180;
  busy = true; statusLabel('generating', 'Generating'); notice('Generating from local weights…'); renderConversation(followScroll);
  try {
    const result = await api('/api/chat', request);
    current.messages = replaceTurn(current, index, prompt, result); persist();
    $('prompt').value = ''; editing = null; $('editBar').hidden = true; resizeComposer();
    notice(`Generated ${result.generated_tokens} tokens in ${result.seconds.toFixed(2)}s.`);
  } catch (error) {
    notice(error.message, true); // Preserve the draft and previous messages; no fabricated reply.
  } finally {
    busy = false; renderConversation(followScroll); await refreshStatus();
  }
}
function edit(index) {
  if (busy || !ready) return;
  editing = index; $('prompt').value = current.messages[index].content; $('editBar').hidden = false;
  resizeComposer(); $('prompt').focus();
}
async function regenerate(index) {
  if (busy || !ready) return;
  const userIndex = index - 1;
  if (current.messages[userIndex]?.role !== 'user') return;
  await submit(current.messages[userIndex].content, userIndex);
}
$('composer').onsubmit = event => { event.preventDefault(); submit($('prompt').value, editing ?? current.messages.length); };
$('prompt').oninput = resizeComposer;
$('prompt').onkeydown = event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); }
};
$('cancelEdit').onclick = () => { editing = null; $('editBar').hidden = true; $('prompt').value = ''; resizeComposer(); };
$('newChat').onclick = () => { if (busy) return; const chat = newConversation(); conversations.unshift(chat); selectConversation(chat); persist(); $('prompt').focus(); };
$('searchChats').oninput = renderThreads;
$('deleteChat').onclick = () => {
  if (busy) return;
  conversations = conversations.filter(chat => chat.id !== current.id);
  if (!conversations.length) conversations.push(newConversation());
  selectConversation(conversations[0]); persist();
};
$('exportChat').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(current, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'kira-conversation.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('memory').oninput = () => { current.memory = $('memory').value; updateContextHint(); persist(); };
$('memoryEnabled').onchange = () => { current.memoryEnabled = $('memoryEnabled').checked; updateContextHint(); persist(); };
$('clearMemory').onclick = () => { current.memory = ''; $('memory').value = ''; updateContextHint(); persist(); };
for (const starter of document.querySelectorAll('[data-prompt]')) starter.onclick = () => { $('prompt').value = starter.dataset.prompt; resizeComposer(); $('prompt').focus(); };
document.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k' && !document.querySelector('dialog[open]')) { event.preventDefault(); $('newChat').click(); }
});
$('retryConnection').onclick = refreshStatus;
$('inspectModel').onclick = async () => {
  closeDrawer(); $('modelDialog').showModal(); await refreshStatus();
  $('modelDetails').textContent = JSON.stringify(modelStatus, null, 2);
  const facts = modelStatus.ready ? {'Device': modelStatus.hardware.device.toUpperCase(), 'Parameters': modelStatus.parameters.toLocaleString(), 'Context': `${modelStatus.config.max_seq_len} tokens`, 'Training step': modelStatus.training_step, 'Layers': modelStatus.config.num_layers, 'Query / KV heads': `${modelStatus.config.num_attention_heads} / ${modelStatus.config.num_kv_heads}`} : {'Model': modelStatus.busy ? 'Operation in progress' : 'No checkpoint loaded'};
  $('modelFacts').replaceChildren();
  for (const [label,value] of Object.entries(facts)) { const wrapper = document.createElement('div'); wrapper.className = 'fact'; const term = document.createElement('dt'), detail = document.createElement('dd'); term.textContent = label; detail.textContent = value; wrapper.append(term,detail); $('modelFacts').append(wrapper); }
  try { plotLoss((await api('/api/metrics')).history, $('lossChart'), $('chartLegend')); }
  catch (error) { $('chartLegend').textContent = error.message; }
};
$('closeModel').onclick = () => $('modelDialog').close();
initSettings(updateContextHint); selectConversation(current); initLab(refreshStatus, () => busy);
// Status polling changes status controls only, never reparses or rebuilds message bodies.
setInterval(() => { if (!document.hidden && !busy) refreshStatus(); }, 12000);
document.addEventListener('visibilitychange', () => { if (!document.hidden && !busy) refreshStatus(); });
