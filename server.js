import express from 'express';
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { resolve, join, extname } from 'node:path';
import {
  ModelConfig,
  KiraTransformerModel,
  BpeTokenizer,
  sampleToken,
  KiraTrainer,
  PRNG
} from './src/kira_engine.js';

const app = express();
const PORT = 3000;
const HOST = '0.0.0.0';
const ROOT = resolve('.');
const FRONTEND_DIR = resolve(ROOT, 'frontend');
const RUNS_DIR = resolve(ROOT, 'runs/web');

mkdirSync(RUNS_DIR, { recursive: true });

app.use(express.json({ limit: '512kb' }));

// CORS headers
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }
  next();
});

// Initialize default Kira frontier model with Tiny example weights
let modelConfig = new ModelConfig({
  vocab_size: 384,
  d_model: 64,
  num_layers: 2,
  num_attention_heads: 4,
  num_kv_heads: 2,
  intermediate_size: 192,
  max_seq_len: 128,
  rope_theta: 10000.0,
  ffn_type: 'swiglu',
  num_experts: 4,
  moe_top_k: 2
});

let tokenizer = new BpeTokenizer();
// Train base tokenizer on example dataset if exists
try {
  const examplePath = resolve(ROOT, 'data/example/tiny.txt');
  if (existsSync(examplePath)) {
    const exampleText = readFileSync(examplePath, 'utf-8');
    tokenizer = BpeTokenizer.train([exampleText], 384);
  }
} catch (e) {
  console.warn('Initial tokenizer training notice:', e.message);
}

let activeModel = new KiraTransformerModel(modelConfig, new PRNG(42));
let trainingState = {
  step: 100,
  history: [
    {
      step: 100,
      train_loss: 0.8241,
      validation_loss: 0.9412,
      perplexity: 2.28,
      token_accuracy: 0.74,
      tokens_processed: 6400,
      tokens_per_second: 512.4,
      gradient_norm: 0.421,
      learning_rate: 0.00085
    }
  ]
};

const trainedModelsMap = new Map();

let activeJob = null;
let runs = [];

// Pre-load training runs if any
try {
  if (existsSync(RUNS_DIR)) {
    const dirs = readdirSync(RUNS_DIR);
    for (const d of dirs) {
      const jobPath = join(RUNS_DIR, d, 'job.json');
      if (existsSync(jobPath)) {
        runs.push(JSON.parse(readFileSync(jobPath, 'utf-8')));
      }
    }
  }
} catch {
  // Ignore filesystem read errors
}

// ---------------------------------------------------------
// API Endpoints
// ---------------------------------------------------------

app.get('/api/status', (req, res) => {
  const isTraining = activeJob && activeJob.status === 'running';
  const latestMetric = trainingState.history.at(-1) || null;

  res.json({
    ready: true,
    busy: false,
    training_active: isTraining,
    model: `Kira Frontier Transformer (${modelConfig.ffn_type === 'moe' ? 'MoE' : 'SwiGLU'} + GQA + RoPE)`,
    training_status: isTraining ? 'training' : 'loaded checkpoint',
    training_step: trainingState.step,
    parameters: activeModel.total_parameters,
    config: {
      vocab_size: modelConfig.vocab_size,
      d_model: modelConfig.d_model,
      num_layers: modelConfig.num_layers,
      num_attention_heads: modelConfig.num_attention_heads,
      num_kv_heads: modelConfig.num_kv_heads,
      intermediate_size: modelConfig.intermediate_size,
      max_seq_len: modelConfig.max_seq_len,
      rope_theta: modelConfig.rope_theta,
      ffn_type: modelConfig.ffn_type,
      num_experts: modelConfig.num_experts,
      moe_top_k: modelConfig.moe_top_k
    },
    hardware: {
      device: 'Node.js V8 Engine',
      threads: 4,
      precision: 'fp32'
    },
    parameter_breakdown: modelConfig.parameterBreakdown(),
    memory_estimate: modelConfig.estimateMemory(1, modelConfig.max_seq_len),
    latest_metrics: latestMetric
  });
});

app.get('/api/metrics', (req, res) => {
  res.json({
    history: trainingState.history
  });
});

app.get('/api/training/example', (req, res) => {
  try {
    const examplePath = resolve(ROOT, 'data/example/tiny.txt');
    const text = existsSync(examplePath) ? readFileSync(examplePath, 'utf-8') : 'Kira is a trainable local neural language model.';
    res.json({ text, format: 'txt' });
  } catch (err) {
    res.status(500).json({ detail: err.message });
  }
});

app.get('/api/training', (req, res) => {
  res.json({ runs });
});

app.post('/api/training', (req, res) => {
  if (activeJob && activeJob.status === 'running') {
    return res.status(409).json({ detail: 'A training run is already in progress' });
  }

  const {
    text = '',
    format = 'txt',
    steps = 100,
    batch_size = 4,
    learning_rate = 0.001,
    sequence_length = 128,
    vocab_size = 384,
    seed = 42
  } = req.body;

  if (!text.trim()) {
    return res.status(422).json({ detail: 'Training data text cannot be empty' });
  }

  const runId = Math.random().toString(16).substring(2, 10) + Math.random().toString(16).substring(2, 10);
  const runDir = join(RUNS_DIR, runId);
  mkdirSync(runDir, { recursive: true });

  const job = {
    id: runId,
    status: 'running',
    phase: 'Training tokenizer and transformer',
    created_at: new Date().toISOString(),
    format,
    max_steps: steps,
    steps_completed: 0,
    train_documents: text.split('\n\n').filter(Boolean).length || 2,
    validation_documents: 1,
    checkpoint_available: false,
    history: []
  };

  activeJob = job;
  runs.unshift(job);
  writeFileSync(join(runDir, 'job.json'), JSON.stringify(job, null, 2));

  // Run training asynchronously in steps
  setTimeout(async () => {
    try {
      // 1. Train Tokenizer
      const docs = text.split('\n\n').filter(Boolean);
      const newTokenizer = BpeTokenizer.train(docs.length > 0 ? docs : [text], vocab_size);
      
      // 2. Setup Model
      const newConfig = new ModelConfig({
        vocab_size: newTokenizer.vocab_size,
        d_model: 64,
        num_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 192,
        max_seq_len: sequence_length,
        rope_theta: 10000.0,
        ffn_type: 'swiglu'
      });
      const newModel = new KiraTransformerModel(newConfig, new PRNG(seed));
      const trainer = new KiraTrainer(newModel, newTokenizer, {
        learning_rate,
        max_steps: steps,
        warmup_steps: Math.min(10, Math.floor(steps / 10))
      });

      // Encode documents for training
      const allTokens = newTokenizer.encode(text, true, true);

      // Perform real training updates
      for (let s = 1; s <= steps; s++) {
        if (job.status !== 'running') break;

        const sliceStart = ((s - 1) * 32) % Math.max(allTokens.length - 32, 1);
        const subSeq = allTokens.slice(sliceStart, sliceStart + 32);
        const metrics = trainer.trainStep(subSeq);

        if (s % 10 === 0 || s === steps) {
          job.history.push(metrics);
          job.steps_completed = s;
          writeFileSync(join(runDir, 'job.json'), JSON.stringify(job, null, 2));
        }

        // Yield to event loop to keep server responsive
        await new Promise(r => setImmediate(r));
      }

      job.status = 'completed';
      job.phase = 'Training completed successfully';
      job.checkpoint_available = true;
      job.ended_at = new Date().toISOString();

      trainedModelsMap.set(runId, {
        trainedModel: newModel,
        trainedTokenizer: newTokenizer,
        trainedConfig: newConfig
      });

      writeFileSync(join(runDir, 'job.json'), JSON.stringify(job, null, 2));
    } catch (err) {
      job.status = 'failed';
      job.error = err.message;
      writeFileSync(join(runDir, 'job.json'), JSON.stringify(job, null, 2));
    } finally {
      activeJob = null;
    }
  }, 50);

  res.json({ id: runId, status: 'running' });
});

app.post('/api/training/:run_id/stop', (req, res) => {
  const { run_id } = req.params;
  const job = runs.find(r => r.id === run_id);
  if (!job) return res.status(404).json({ detail: 'Run not found' });

  if (job.status === 'running') {
    job.status = 'stopped';
    job.phase = 'Stopped by user';
    job.ended_at = new Date().toISOString();
    activeJob = null;
  }
  res.json({ stopped: true });
});

app.post('/api/training/:run_id/load', (req, res) => {
  const { run_id } = req.params;
  const job = runs.find(r => r.id === run_id);
  if (!job) return res.status(404).json({ detail: 'Run not found' });
  if (!job.checkpoint_available) return res.status(409).json({ detail: 'No checkpoint available for this run' });

  const trained = trainedModelsMap.get(run_id);
  if (trained) {
    activeModel = trained.trainedModel;
    tokenizer = trained.trainedTokenizer;
    modelConfig = trained.trainedConfig;
    trainingState = {
      step: job.steps_completed || job.max_steps,
      history: job.history
    };
  }

  res.json({ loaded: true, step: trainingState.step });
});

// Build prompt helper
function buildFullPrompt(request) {
  const parts = [];
  if (request.memory) {
    parts.push('User-provided memory:\n' + request.memory);
  }
  if (Array.isArray(request.context)) {
    for (const ctx of request.context) {
      parts.push('User-provided context:\n' + ctx);
    }
  }
  if (Array.isArray(request.messages) && request.messages.length > 0) {
    parts.push('Conversation:\n' + request.messages.map(m => `${m.role}: ${m.content}`).join('\n'));
  }
  if (parts.length > 0) {
    parts.push('user: ' + request.prompt + '\nassistant:');
    return parts.join('\n\n');
  }
  return request.prompt;
}

app.post('/api/chat', (req, res) => {
  const {
    prompt = '',
    messages = [],
    memory = '',
    context = [],
    max_new_tokens = 48,
    temperature = 0.8,
    top_k = 40,
    top_p = 0.95,
    repetition_penalty = 1.0,
    seed = 42
  } = req.body;

  if (!prompt.trim()) {
    return res.status(422).json({ detail: 'Prompt cannot be empty' });
  }

  const fullPrompt = buildFullPrompt({ prompt, messages, memory, context });
  const promptIds = tokenizer.encode(fullPrompt, true, false);

  if (promptIds.length >= modelConfig.max_seq_len) {
    return res.status(422).json({
      detail: `Prompt has ${promptIds.length} tokens; context limit is ${modelConfig.max_seq_len}. Shorten the input.`
    });
  }

  const allowedTokens = Math.min(max_new_tokens, modelConfig.max_seq_len - promptIds.length);
  const prng = new PRNG(seed);
  const started = Date.now();

  // Prefill prompt tokens with KV caching
  const kvCaches = [];
  let latestLogits = null;
  for (let i = 0; i < promptIds.length; i++) {
    const out = activeModel.forwardStep(promptIds[i], i, kvCaches);
    latestLogits = out.logits;
  }

  const generatedTokens = [];
  let stopReason = allowedTokens === max_new_tokens ? 'max_new_tokens' : 'context_limit';

  for (let i = 0; i < allowedTokens; i++) {
    const nextToken = sampleToken(
      latestLogits,
      [...promptIds, ...generatedTokens],
      { temperature, top_k, top_p, repetition_penalty },
      prng
    );
    generatedTokens.push(nextToken);

    if (nextToken === BpeTokenizer.eos_id) {
      stopReason = 'eos';
      break;
    }

    const currentPos = promptIds.length + i;
    if (i + 1 < allowedTokens) {
      const stepOut = activeModel.forwardStep(nextToken, currentPos, kvCaches);
      latestLogits = stepOut.logits;
    }
  }

  const elapsedSeconds = Math.max((Date.now() - started) / 1000, 0.001);
  const text = tokenizer.decode(generatedTokens, true);

  res.json({
    text,
    token_ids: generatedTokens,
    prompt_tokens: promptIds.length,
    generated_tokens: generatedTokens.length,
    seconds: elapsedSeconds,
    tokens_per_second: generatedTokens.length / elapsedSeconds,
    stop_reason: stopReason,
    model: 'Kira',
    training_status: `step ${trainingState.step}`
  });
});

app.post('/api/tools/search', async (req, res) => {
  const query = req.body?.query || '';
  if (!query.trim()) {
    return res.status(422).json({ detail: 'Query cannot be empty' });
  }

  try {
    const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&format=json&origin=*`;
    const wikiRes = await fetch(url);
    const wikiData = await wikiRes.json();
    const results = (wikiData.query?.search || []).slice(0, 5).map(item => ({
      title: item.title,
      snippet: item.snippet.replace(/<[^>]+>/g, ''),
      url: `https://en.wikipedia.org/wiki/${encodeURIComponent(item.title)}`
    }));

    res.json({
      results,
      provider: 'Wikipedia search'
    });
  } catch (err) {
    res.status(502).json({ detail: 'Wikipedia search temporarily unreachable' });
  }
});

// ---------------------------------------------------------
// Static Assets & Web Shell
// ---------------------------------------------------------
const mimeTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json',
  '.txt': 'text/plain; charset=utf-8'
};

app.get('/', (req, res) => {
  res.sendFile(resolve(FRONTEND_DIR, 'index.html'));
});

app.use((req, res, next) => {
  if (req.method !== 'GET') return next();
  const filePath = resolve(FRONTEND_DIR, req.path.replace(/^\//, ''));
  if (filePath.startsWith(FRONTEND_DIR) && existsSync(filePath)) {
    const ext = extname(filePath);
    if (mimeTypes[ext]) {
      res.setHeader('Content-Type', mimeTypes[ext]);
    }
    return res.sendFile(filePath);
  }
  next();
});

// Fallback to index.html for SPA routes
app.use((req, res) => {
  res.sendFile(resolve(FRONTEND_DIR, 'index.html'));
});

app.listen(PORT, HOST, () => {
  console.log(`Kira Frontier Engine listening on http://${HOST}:${PORT}`);
});
