import {plotLoss} from './chart.js';

const $ = id => document.getElementById(id);
const URL_KEY = 'kira-backend-url';
let backendUrl = '';
let accessToken = ''; // Deliberately never persisted, placed in a URL, or exported.
let selectedRun = null;
let pending = false;
let polling = null;
try { backendUrl = localStorage.getItem(URL_KEY) || ''; } catch { /* Storage is optional. */ }

export async function api(path, body) {
  if (!backendUrl && location.hostname.endsWith('.github.io')) {
    throw new Error('GitHub Pages needs a connected Python backend.');
  }
  const headers = {};
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  let response;
  try {
    response = await fetch(`${backendUrl}${path}`, {
      method: body === undefined ? 'GET' : 'POST', headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: 'omit', redirect: 'error', signal: AbortSignal.timeout(120000)
    });
  } catch {
    throw new Error('Cannot reach backend. Check its URL, HTTPS, allowed origin, and that it is running.');
  }
  let data;
  try { data = await response.json(); }
  catch { throw new Error('This URL did not return the Kira API. Check the forwarded port and its visibility.'); }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));
  return data;
}

export function initLab(refreshModel, isGenerating) {
  $('backendUrl').value = backendUrl;
  $('openConnection').onclick = () => $('connectionDialog').showModal();
  $('closeConnection').onclick = () => $('connectionDialog').close();
  $('connectionForm').onsubmit = async event => {
    event.preventDefault();
    if (pending || isGenerating()) {
      $('connectionStatus').textContent = 'Wait for the current request before changing connections.';
      return;
    }
    try {
      const value = $('backendUrl').value.trim().replace(/\/+$/, '');
      if (value) {
        const url = new URL(value);
        if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
          throw new Error('Use only the backend origin, such as https://your-codespace-3000.app.github.dev');
        }
        if (location.protocol === 'https:' && url.protocol !== 'https:') throw new Error('This HTTPS page needs an HTTPS backend.');
      }
      backendUrl = value;
      accessToken = $('backendToken').value.trim();
      selectedRun = null;
      $('backendToken').value = '';
      try { localStorage.setItem(URL_KEY, backendUrl); } catch { /* Connection still works without storage. */ }
      $('connectionStatus').textContent = 'Connecting…';
      const connected = await refreshModel();
      $('connectionStatus').textContent = connected ? 'Connected. Open Train to create or load a checkpoint.' : $('modelStatus').textContent;
      if (connected) { $('connectionDialog').close(); await refreshRuns(); }
    } catch (error) { $('connectionStatus').textContent = error.message; }
  };

  function showRun(run) {
    selectedRun = run.id;
    const last = run.history.at(-1);
    $('trainingStatus').textContent = `${run.status} · ${run.phase} · ${last?.step || 0}/${run.max_steps} measured updates${run.error ? ` · ${run.error}` : ''}`;
    $('trainingMetrics').textContent = last ? JSON.stringify(last, null, 2) : 'No optimizer measurements yet.';
    plotLoss(run.history, $('trainingChart'), $('trainingLegend'));
  }
  async function action(run, verb) {
    $('trainingError').textContent = '';
    if (pending || isGenerating()) return;
    pending = true;
    try {
      await api(`/api/training/${run.id}/${verb}`, {});
      await refreshModel();
      await refreshRuns();
      if (verb === 'load') {
        $('trainingDialog').close();
        $('generationStatus').textContent = 'Checkpoint loaded. Enter a short text prefix to test it.';
        $('prompt').focus();
      }
    } catch (error) { $('trainingError').textContent = error.message; }
    finally { pending = false; }
  }
  async function refreshRuns() {
    try {
      const {runs} = await api('/api/training');
      $('trainingRuns').replaceChildren();
      $('trainingFields').disabled = runs.some(run => run.status === 'running');
      for (const run of runs) {
        const row = document.createElement('article'); row.className = 'run-row';
        const summary = document.createElement('button');
        summary.textContent = `${new Date(run.created_at).toLocaleString()} · ${run.status} · step ${run.history.at(-1)?.step || 0}`;
        summary.onclick = () => showRun(run);
        row.append(summary);
        if (run.status === 'running') {
          const stop = document.createElement('button'); stop.textContent = 'Stop training';
          stop.onclick = () => action(run, 'stop'); row.append(stop);
        } else if (run.checkpoint_available) {
          const load = document.createElement('button'); load.textContent = 'Load into chat';
          load.className = 'primary'; load.disabled = runs.some(item => item.status === 'running');
          load.onclick = () => action(run, 'load'); row.append(load);
        }
        $('trainingRuns').append(row);
      }
      const selected = runs.find(run => run.id === selectedRun) || runs[0];
      if (selected) showRun(selected);
      else {
        $('trainingStatus').textContent = 'No training runs yet. Use the tiny example to test the pipeline.';
        $('trainingMetrics').textContent = 'No measurements yet.';
        plotLoss([], $('trainingChart'), $('trainingLegend'));
      }
    } catch (error) {
      $('trainingError').textContent = error.message;
      $('trainingFields').disabled = true;
    }
  }
  async function poll() {
    await refreshRuns();
    if ($('trainingDialog').open) polling = setTimeout(poll, 2000);
  }
  $('openTraining').onclick = () => {
    $('trainingDialog').showModal(); clearTimeout(polling); poll();
  };
  $('closeTraining').onclick = () => $('trainingDialog').close();
  $('trainingDialog').addEventListener('close', () => clearTimeout(polling));
  $('exampleDataset').onclick = async () => {
    try {
      const data = await api('/api/training/example');
      $('datasetText').value = data.text; $('datasetFormat').value = data.format;
      $('trainingStatus').textContent = 'Tiny synthetic example loaded. This is test data, not language pretraining.';
    } catch (error) { $('trainingError').textContent = error.message; }
  };
  $('datasetFile').onchange = async event => {
    const file = event.target.files[0]; if (!file) return;
    try {
      const format = file.name.split('.').at(-1).toLowerCase();
      if (!['txt', 'json', 'jsonl'].includes(format) || file.size > 32768) throw new Error('Use a .txt, .json or .jsonl file up to 32 KiB.');
      $('datasetText').value = await file.text(); $('datasetFormat').value = format;
      $('trainingStatus').textContent = `${file.name} loaded locally. Start training to send it to your backend.`;
    } catch (error) { $('trainingError').textContent = error.message; }
    event.target.value = '';
  };
  $('trainingForm').onsubmit = async event => {
    event.preventDefault();
    if (pending || isGenerating()) return;
    pending = true; $('trainingFields').disabled = true; $('trainingError').textContent = '';
    try {
      const text = $('datasetText').value;
      if (new TextEncoder().encode(text).length > 32768) throw new Error('Web datasets are limited to 32 KiB. Use Python for larger corpora.');
      const run = await api('/api/training', {text, format: $('datasetFormat').value,
        steps: Number($('trainingSteps').value), batch_size: Number($('trainingBatch').value),
        learning_rate: Number($('trainingRate').value), sequence_length: Number($('trainingContext').value),
        vocab_size: Number($('trainingVocab').value), seed: Number($('trainingSeed').value)});
      selectedRun = run.id;
      await refreshRuns();
      $('trainingStatus').scrollIntoView({block: 'center', behavior: 'smooth'});
    } catch (error) { $('trainingError').textContent = error.message; $('trainingFields').disabled = false; }
    finally { pending = false; }
  };
  refreshModel();
}
