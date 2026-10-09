# Kira — a local, trainable text Transformer

Kira is a language model foundation implemented in Python and PyTorch, with its own architecture, trainable byte BPE tokenizer, next-token training, resumable checkpoints, local generation, evaluation, and browser interface. It uses **no external LLM inference service and needs no AI service API key**. You can chat and start real training in the browser, including from a GitHub Pages frontend connected to your own Python backend.

**Architecture code does not equal intelligence. Useful language behavior comes from training the model on data.**

No pretrained language model is bundled. A fresh checkout has no weights. Training starts from random initialization. The original example corpus is only for exercising the pipeline; training on it does not create a useful assistant. A finished schedule is labeled **quality unverified**. The browser never invents replies or training measurements.

## Quick Start

Supported Python: **3.9+**, with **3.11 recommended for CPU testing**. The reference CPU constraints target Python 3.9 and 3.11. Node is optional for running Kira. Pinned browser assets are included locally, so no npm install, CDN or build step is needed to chat. Frontend development/tests use Node 22.12 LTS or Node 24+ (24 tested). Run the commands from the repository root. Internet is needed to install packages; core training/inference then runs locally without credentials.

### 1. Clone and create a Python environment

```bash
git clone https://github.com/Arya-da-goat/kira.git
cd kira
python3 -m venv .venv
source .venv/bin/activate
```

On Windows use `python -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell instead. All Python module commands below are the same. The optional npm shortcut and smoke shell script assume a POSIX shell; direct Python commands do not.

### 2. Install dependencies

For the reference **Linux/Windows CPU** environment:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -c constraints-cpu.txt
python -m pip check
```

`constraints-cpu.txt` pins the reference dependency versions. For macOS/MPS or CUDA, install the PyTorch wheel appropriate to your hardware first, then `python -m pip install -r requirements.txt` using the bounded dependencies in `pyproject.toml`. No environment file or API key is needed.

### 3. Prepare data and train the tokenizer

A fresh clone includes `data/example/tiny.txt`, a tiny original synthetic fixture. No preparation or download is needed for this test corpus. For your own corpus, first follow [dataset preparation](data/README.md), place files in `data/private/`, and replace `data/example` in both commands below. Tokenizer fitting must follow dataset preparation so it uses the correct training split.

```bash
python -m kira.tokenizer.train --data data/example --config configs/tiny.json
```

### 4. Prove tiny training works

```bash
python -m pytest -q -s tests/test_core.py::test_tiny_overfit
```

This creates temporary random weights, performs real updates, and requires decreasing loss and the learned sequence. It does not download or depend on committed weights.

### 5. Train Kira from scratch

```bash
python -m kira.training.train --data data/example --config configs/tiny.json
```

The default is a 100-update CPU development run. It creates ignored checkpoint files. For a subsequent run, use a fresh `--output checkpoints/another-run` or resume an unfinished run; the trainer refuses to overwrite an existing run.

### 6. Run local inference

```bash
python -m kira.inference.generate --checkpoint checkpoints/latest.pt --prompt "hello"
```

### 7. Start the web interface

```bash
python -m backend.server --checkpoint checkpoints/latest.pt
```

Open **http://127.0.0.1:3000**. Without a checkpoint the server serves the interface, reports missing weights, and returns HTTP 503 for generation. Python is required; there is no hosted inference or JavaScript model fallback. Open **Train** to upload a small dataset, start real training, inspect measured losses, and **Load into chat** when a checkpoint is saved. `npm start` is an optional shortcut using `.venv/bin/python`.

The tiny configuration supports CPU development. Choose `--device cpu`, `cuda`, `mps`, or `auto`. CPU/MPS use FP32. CUDA auto selects BF16 when supported, otherwise FP16 with gradient scaling. CLI commands default to two CPU threads. Training prints device/precision and GPU name when available. Larger models need more memory for gradients, AdamW states, activations, and attention; scale gradually and measure on your hardware.

## GitHub Pages and mobile setup

**GitHub Pages serves the interface; it cannot run Python/PyTorch.** Your phone can use the interface while a backend runs in GitHub Codespaces or on a machine you control. No inference provider is involved. A stopped Codespace means chat and training are unavailable. Codespaces usage is subject to your account's quota/billing. [About GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages).

### Publish the interface (once)

1. Merge this change into your repository's `main` branch.
2. In your phone **browser**, open the repository → **Settings → Pages**. Enable Desktop site if the setting is hidden.
3. Under **Build and deployment**, choose **Deploy from a branch**, then **main** and **/ (root)**, and save.
4. Wait for GitHub to publish, then open **https://arya-da-goat.github.io/kira/** (use your own account/site URL for a fork).

The root `index.html` opens `frontend/`. Assets use relative paths so project subpaths work. `.nojekyll` enables plain static publishing. No npm build or custom Actions workflow is required. Publishing still depends on your repository's Pages permissions/settings and GitHub's deployment service. [Publishing source settings](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

### Start the Python backend from your phone

1. In the repository's **Code → Codespaces** menu, create/open a Codespace on the branch containing these changes. Use the phone browser's Desktop site if necessary.
2. In its terminal, run this **one command** from the repository root:

   ```bash
   bash scripts/start_web.sh
   ```

   The Linux CPU helper creates `.venv` and installs dependencies if needed, then starts the actual backend. Initial installation needs internet and may take several minutes. For a fork/custom Pages domain, pass your exact site **origin without a path**: `bash scripts/start_web.sh https://YOUR-ACCOUNT.github.io`. On an already prepared machine you can instead run:

   ```bash
   python -m backend.server --host 0.0.0.0 --device cpu --allow-origin https://arya-da-goat.github.io
   ```

3. Copy the **backend access token** printed in that terminal. Keep it private. This is a password for your own backend, not an AI API key.
4. In the Codespace **Ports** panel, forward **3000** if it is not already listed. Set its visibility to **Public** so the Pages origin can reach it, then copy its **HTTPS forwarded address**. Keep the backend's token protection enabled. An organization policy may prohibit public ports; in that case use the backend's own authenticated preview or another HTTPS backend you control. [Codespaces ports](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace).
5. Open your Pages site, tap **Connect**, paste the forwarded URL and access token, and tap **Connect backend**.
6. Tap **Train → Use tiny example** (or upload your own `.txt`, `.json`, `.jsonl`), review the settings, check the data permission box, and tap **Start real training**.
7. Wait for measured optimizer updates. Tap **Load into chat**, enter a short prefix such as `hello`, then **Generate**.

A tiny run will produce poor text: this is a real training test, not a pretrained assistant. The training panel shows actual cross entropy, validation loss/perplexity/accuracy, learning rate, gradient norm, tokens processed, throughput, and loss curves. Each new run starts from random weights and trains its own tokenizer on the training split. Training and generation are serialized. Use Python CLI configuration for larger runs, resume, and evaluation exports.

### Connection, storage, and troubleshooting

- A Pages URL such as `https://arya-da-goat.github.io/kira/` has origin **`https://arya-da-goat.github.io`**; do not include `/kira/` in `--allow-origin`.
- Use HTTPS for a backend connected to an HTTPS Pages site. Paste the forwarded backend address, not your repository, Pages, or Codespaces editor URL. Private port redirects cannot be used by this cross-origin client.
- The backend URL is remembered locally. The access token exists only in tab memory; **re-enter it after reloading**. Requests send it in an Authorization header, never a URL. No shared token is included in the public site.
- On non-loopback binds or when `--allow-origin` is set, the server generates/reuses a private token in ignored `runs/backend-access.txt`. Delete this local file and restart to rotate the token. Stop the server when finished. Only share the token with people allowed to train, generate, and read your run measurements.
- Training uploads are capped at **32 KiB**, 2–2,000 optimizer steps, and the small dense architecture. Blank lines delimit text documents; JSON/JSONL accept strings or objects with `text`. At least two distinct documents are required. Data leaves your browser only when you start training (or explicitly supply chat context).
- Datasets, configs, logs, tokenizer and checkpoints are under ignored **`runs/web/RUN_ID/`** in the backend. The panel lists the latest 20 runs. Closing a tab/panel does not cancel training; **Stop training** terminates the worker and retains any already saved checkpoint. Load shows the actual saved step, which can lag the last logged update.
- Normal backend shutdown stops its worker. After an abrupt crash, a run is marked interrupted and cannot be loaded through the UI; inspect/stop any leftover worker before using CLI resume on its `latest.pt`. Run one backend process per run directory. There is no public multi-user job scheduler or automated artifact retention.
- Runs remain on that machine across server restarts; load them explicitly from Train. Back up wanted checkpoints/data privately before deleting a Codespace. **Never commit them to Git or publish them on Pages.**
- Only local loopback use is unauthenticated by default. Remote serving requires the generated token and explicit allowed Pages origin. This is a personal development lab, not hardened public model hosting.

## Architecture

`configs/tiny.json` defines the development model:

| Component | Default |
|---|---|
| Model | Autoregressive decoder-only dense Transformer |
| Layers / width | 2 / 64 |
| Attention | 4 query heads, 2 KV heads, head dimension 16 |
| FFN | SwiGLU, intermediate dimension 192 |
| Normalization | Pre-RMSNorm and final RMSNorm |
| Positions | RoPE on Q/K, base 10,000 |
| Context | 128 tokens |
| Vocabulary | Up to 384; actual trained tokenizer size is authoritative |
| Output | Tied embedding / LM head |

Each block is RMSNorm → causal attention → residual → RMSNorm → SwiGLU → residual. Attention uses PyTorch scaled-dot-product attention with an explicit causal mask, including cached offsets. GQA repeats contiguous KV head groups. The cache retains rotated keys and values per layer; after prompt prefill, generation processes one token at a time. Cache memory grows with context. Context overflow is reported or generation stops at the configured limit; there is no silent truncation.

## Tokenizer and datasets

Deterministic byte-level BPE merges are learned from training documents. All 256 bytes provide UTF-8 coverage; `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` have separate IDs. Literal spellings in user text remain ordinary text. Invalid IDs decode as `<UNK>`; sampled invalid UTF-8 bytes display replacement characters. Vocabulary byte sequences and merge rules persist as JSON and are embedded in checkpoints.

See [data/README.md](data/README.md). Inputs may be:

- `.txt`: blank-line-separated documents.
- `.json`: a string, text object, or array of strings / `{"text": "..."}` objects.
- `.jsonl`: one string or text object per line.
- A directory of these files, traversed recursively.

Provide at least two distinct documents. Cleaning normalizes newlines, removes NULs, trims edges, and removes exact duplicates. Splitting happens by document **before** tokenizer training/chunking. Tokenizer and trainer use the same seed/validation fraction. Provenance fingerprints reject mismatched tokenizers or splits. Near duplicates and related documents require upstream deduplication/grouping to prevent semantic leakage. The in-memory loader and pure Python BPE trainer suit small corpora; large-scale work requires streaming and faster tokenization.

```bash
python -m kira.tokenizer.train --data data/private/my-corpus --output checkpoints/my-tokenizer.json
python -m kira.training.train --data data/private/my-corpus \
  --tokenizer checkpoints/my-tokenizer.json --output checkpoints/my-run
```

Use only data you have permission to train on. Private corpora belong in ignored `data/private/`. Original example text is dedicated to the public domain (CC0). No books or downloaded corpora are bundled.

## Training and checkpoints

Inputs `tokens[:-1]` predict `tokens[1:]`, shifted exactly once. Each document receives BOS/EOS. Chunking preserves every next-token target; right padding uses ignored target `-100`. Cross entropy is normalized by real target tokens across each accumulation group. Training performs backward, unscales FP16 gradients, measures/clips their norm, steps AdamW, steps a warmup/cosine scheduler, and clears gradients.

Configure batch size, accumulation, learning rate, weight decay, warmup, maximum steps/epochs, clipping, precision, evaluation interval and save interval in JSON. The first step/epoch limit reached stops training. Model vocabulary size follows the actual trained tokenizer size.

Checkpoints save weights, model/training configuration, embedded tokenizer/fingerprint, optimizer/scheduler/scaler state, step, epoch, next batch position, dataset fingerprint, processed target count, metrics, and Python/PyTorch device RNG states. Writes use a temporary file and atomic replacement.

- `latest.pt`: saved on validation improvement, periodic save, and clean completion.
- `best.pt`: lowest measured validation loss so far.
- `step-XXXXXXXX.pt`: retained at `save_interval`; manage disk retention yourself.
- `metrics.jsonl`: measured updates; resume restores checkpoint history.

```bash
# Controlled interruption without changing the planned schedule:
python -m kira.training.train --data data/example --stop-after 40
python -m kira.training.train --resume checkpoints/latest.pt
```

Resume restores the existing configuration/schedule; `--config` applies to new runs. Changed data is rejected. `--data NEW_LOCATION` permits identical relocated data. Resume in the same output directory to retain earlier best/periodic files. CPU exact resumption is tested with dropout and accumulation; device/library changes or nondeterministic kernels can change results. Crashes resume from the last checkpoint, not necessarily the last logged step. Load only trusted checkpoints; restricted `weights_only=True` loading reduces, but does not eliminate, risks from hostile files.

Logs include token-weighted loss, held-out perplexity/accuracy, target tokens processed, pre-clipping gradient norm, learning rate used, and synchronized training tokens/sec. Throughput includes forward/backward/update work and excludes evaluation/save I/O. No benchmark scores are claimed.

### Fine-tuning

Continuing an unfinished pretraining run is implemented. Instruction SFT, LoRA, DPO, MoE, and multimodal training are **not implemented**, and no controls claim otherwise. A new corpus currently requires a new run/split/tokenizer; `--resume` is not an instruction-tuning pipeline.

## Inference and evaluation

```bash
python -m kira.inference.generate --checkpoint checkpoints/best.pt \
  --prompt "hello" --max-new-tokens 48 --temperature 0.8 \
  --top-k 40 --top-p 0.95 --repetition-penalty 1.1 --seed 42
python -m kira.evaluation.evaluate --checkpoint checkpoints/latest.pt \
  --output runs/evaluation --prompt "hello" --prompt "the cat"
```

Temperature 0 selects greedy decoding. Top-k 0 and top-p 1 disable their respective filters. Sampling uses a per-request seeded CPU generator. Identical weights, inputs, software and device produce repeatable results. PAD/BOS/UNK are excluded; EOS stops generation and counts as a generated token. Output includes text, IDs, counts, elapsed time, throughput and stop reason. The base model performs text completion; it is not instruction trained.

Evaluation reconstructs the recorded validation split and rejects mismatched data. It writes measured loss/perplexity/accuracy and generation samples (`evaluation.json`) plus training/validation curves (`loss.svg`). Perplexity becomes null if exponentiation would overflow. These are run measurements, not benchmark scores.

## Web interface and tools

The dark interface has a collapsible desktop sidebar, keyboard-contained mobile drawer, responsive composer, saved conversations, conversation search, export/delete, local Font Awesome icons, and dedicated generation, memory, tools, model and training panels.

- **Messages:** assistant Markdown is sanitized with DOMPurify; raw HTML is displayed as text, remote images are omitted, unsafe links are removed, and links cannot control the opener. Fenced Python/JavaScript/JSON/Bash/CSS/HTML code is highlighted lazily with a copy control. User text remains literal text. Rendering preserves existing message DOM; the latest 80 messages are displayed initially, with a Show earlier messages control for older turns.
- **Actions:** Copy copies actual text. Edit loads a user message into the composer. Regenerate makes a new inference request. Both use current sampling settings, memory and attached context, and replace the selected turn plus later turns **only after successful generation**. A failed request preserves the old conversation and draft. Regenerate with unchanged seed/settings is intentionally reproducible.
- **Composer:** Enter sends, Shift+Enter inserts a newline, and IME composition does not accidentally send. Ctrl/Cmd+K creates a new chat when no modal is open. The scrollable conversation stays separate from the composer. Reduced-motion preferences disable animation.
- **Settings:** temperature, top-k/p, repetition penalty, seed, maximum new tokens and optional last-40-message history affect actual API requests. Temperature zero is greedy. Preferences stay in browser storage. Memory has a separate opt-in toggle per conversation, editable notes and a Clear action; turning it off excludes notes from the model request without deleting them.
- **Attachments:** UTF-8 text files only, up to 32 KiB each, eight context items / 48 KB combined. Each chip shows filename, type and actual byte size, and can be removed individually. Raw text is supplied to the model on generation; the app does not claim semantic understanding. Images, PDFs, audio and binary files are rejected. Attachments/search results are transient and are not added automatically when regenerating in a later session.
- **Status:** loaded/unloaded weights, backend offline, active training and generation reflect backend state. No fake reasoning trace or artificial progress is shown. Visible errors give actionable input/connection guidance; full Python logs stay at the backend.

 Model details display checkpoint measurements. The Train panel starts a separate Python process, polls real logged measurements, stops jobs, and loads saved weights. It never simulates browser training progress. Oversized prompts produce explicit context errors; shorten memory/files/history or train with a larger context.

Memory/files are input context, never weight updates. History and editable memory notes persist in browser localStorage; file contents and attached search snippets are transient. Previously saved notes start disabled unless the user explicitly enabled the memory toggle. Optional **Wikipedia web search** runs only on user request. Results appear separately and can be explicitly attached as model input. Search is not a replacement for generation; autonomous model-directed tool use is not implemented. Search requires internet; training/inference do not. Images, audio, fabricated reasoning traces and automatic chat training are unsupported.

The server binds to loopback by default, serves only frontend assets, bounds request sizes and permits one model operation at a time. Remote binds and cross-origin access enable bearer-token protection. Only explicitly configured origins are allowed; model checkpoints and raw datasets are never served as static files. Secrets never belong in frontend code. `.env` is ignored; `.env.example` contains no required variables.

## Structure

```text
frontend/         HTML, CSS, browser client
backend/          HTTP bridge, access protection and local training jobs
kira/model/       Transformer, GQA, RoPE, RMSNorm, SwiGLU
kira/tokenizer/   Byte BPE and tokenizer training CLI
kira/training/    Dataset, AdamW, scheduler, training, checkpoints
kira/inference/   Cached generation and sampling
kira/evaluation/  Held-out metrics, samples and curves
kira/tools/       Optional separate web search
configs/          JSON model/training settings
data/             Original examples; ignored private corpora
checkpoints/      Tracked README; ignored local tokenizer and weights
scripts/          Pipeline smoke test, audit, browser asset maintenance, CPU helper
tests/            Math, learning, resume, API and frontend security/state tests
```

## Verification

```bash
python -m pytest -q -s
python -m compileall -q kira backend tests
python -m pip check
npm ci --ignore-scripts  # Optional Node tooling for frontend development/tests
npm run check
npm test
bash scripts/smoke.sh runs/smoke
```

Tests verify tokenizer Unicode/special/unknown behavior, persisted vocabularies, RoPE, independent GQA reference math, causality, cache equivalence, logits, gradients, parameter updates, dataset shifting/splitting, save/load/exact resume, seed determinism, EOS, sampling and API boundaries, real HTTP training/load/generation/cancellation, CORS/authentication and static Pages paths. The overfit test trains for 100 updates and requires loss below 0.1 plus greedy `hello world` followed by EOS. That proves narrow learning, not general intelligence.

## Development weights versus an actual Kira checkpoint

The tiny configuration and synthetic corpus verify software behavior. Their weights are **development/test artifacts**, even though gradient updates are real. An actual Kira training effort needs an appropriately sized, licensed corpus, adequate compute and training duration, held-out evaluation, and a model card with measured results and limitations. This repository supplies the foundation and does not distribute a useful pretrained checkpoint. See [checkpoint storage and sharing](checkpoints/README.md).

## GitHub readiness and local checks

Automatic GitHub Actions checks are deferred in this version. Run the verification commands above locally using the reference CPU environment. They cover tokenizer round trips, model forward/backward passes, tiny overfitting, checkpoint resume and repository hygiene without a GPU, inference credentials, private dataset or pretrained weights. Adding a workflow later requires GitHub access that permits workflow updates. No hosted CI run is claimed.

Before committing:

```bash
python scripts/audit_repository.py
git status --short --untracked-files=all
git add --dry-run .
git diff --check
```

The audit examines existing tracked files and non-ignored new files, checking file sizes, generated/private artifacts, common secret patterns and developer-specific paths without printing secret values. It is a useful check, not a guarantee that every secret is detectable. Inspect the actual diff before publishing. `.gitignore` excludes weights everywhere, all checkpoint outputs, private/raw/processed corpora, virtual environments, caches, logs, generated results and local credentials. Only the tiny synthetic fixture is allowed under `data/`. Tests generate their own checkpoint fixtures; no weights are version-controlled.

The optional `package.json` provides startup, syntax checks, frontend tests and reproducible asset maintenance. All npm packages are development dependencies; browser runtime files are vendored under `frontend/vendor/` with version locks, licenses and a SHA256 manifest. No `node_modules/`, app bundles, caches or build directories are tracked. Run `npm ci --ignore-scripts` then `npm run vendor` only when refreshing these pinned assets. See [third-party notices](THIRD_PARTY_NOTICES.md). Python remains the required model runtime.

## Limitations

No useful pretrained checkpoint, large-corpus result, instruction-following guarantee, benchmark claim, vision/audio, distributed training, quantization or production serving system is supplied. Configuration scales the architecture, but tokenizer/data/serving implementations need further work for large workloads. CPU tests do not validate CUDA/MPS operation or mixed-precision performance. Keep measured quality separate from implemented architecture.

## License

Source code is available under the [MIT License](LICENSE). The original synthetic text in `data/example/tiny.txt` is separately dedicated to the public domain (CC0), as documented in `data/README.md`. Users are responsible for permissions on their own datasets and redistributed checkpoints.
