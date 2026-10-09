import {plotLoss} from './chart.js';
import {api, connectionURL, setConnection} from './api.js';
import {icon} from './ui.js';
export {api} from './api.js';

const $ = id => document.getElementById(id);
let selectedRun = null;
let pending = false;
let polling = null;
let runsRevision = 0;

export function initLab(refreshModel, isGenerating) {
  $('backendUrl').value = connectionURL();
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
      setConnection(value, $('backendToken').value.trim());
      selectedRun = null;
      $('backendToken').value = '';
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
    $('trainingMetrics').replaceChildren();
    const measurements = last ? {
      'Training loss': last.train_loss.toFixed(4), 'Validation loss': last.validation_loss?.toFixed(4) ?? 'Not evaluated at this step',
      'Tokens processed': last.tokens_processed.toLocaleString(), 'Tokens / second': last.tokens_per_second.toFixed(1),
      'Gradient norm': last.gradient_norm.toFixed(4), 'Learning rate': last.learning_rate.toExponential(2),
      'Perplexity': last.perplexity?.toFixed(2) ?? 'Not available at this step', 'Token accuracy': last.token_accuracy == null ? 'Not evaluated at this step' : `${(last.token_accuracy * 100).toFixed(2)}%`,
      'Training documents': run.train_documents, 'Validation documents': run.validation_documents
    } : {'Updates': 'No measurements yet'};
    for (const [label,value] of Object.entries(measurements)) {
      const wrapper = document.createElement('div'); wrapper.className = 'fact';
      const term = document.createElement('dt'), detail = document.createElement('dd');
      term.textContent = label; detail.textContent = value; wrapper.append(term, detail); $('trainingMetrics').append(wrapper);
    }
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
    const revision = ++runsRevision;
    try {
      const {runs} = await api('/api/training');
      if (revision !== runsRevision) return;
      $('trainingRuns').replaceChildren();
      $('trainingFields').disabled = runs.some(run => run.status === 'running');
      for (const run of runs) {
        const row = document.createElement('article'); row.className = 'run-row';
        const summary = document.createElement('button');
        summary.textContent = `${new Date(run.created_at).toLocaleString()} · ${run.status} · step ${run.history.at(-1)?.step || 0}`;
        summary.prepend(icon('file'));
        summary.onclick = () => showRun(run);
        row.append(summary);
        if (run.status === 'running') {
          const stop = document.createElement('button'); stop.textContent = 'Stop training'; stop.prepend(icon('stop'));
          stop.onclick = () => action(run, 'stop'); row.append(stop);
        } else if (run.checkpoint_available) {
          const load = document.createElement('button'); load.textContent = 'Load into chat'; load.prepend(icon('check'));
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
      if (revision !== runsRevision) return;
      $('trainingError').textContent = error.message;
      $('trainingFields').disabled = true;
    }
  }
  async function poll() {
    if (!pending && !isGenerating()) { await refreshRuns(); await refreshModel(); }
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
  $('uploadDataset').onclick = () => $('datasetFile').click();
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
      await refreshModel();
      $('trainingStatus').scrollIntoView({block: 'center', behavior: 'smooth'});
    } catch (error) { $('trainingError').textContent = error.message; $('trainingFields').disabled = false; }
    finally { pending = false; }
  };
  refreshModel();
}
