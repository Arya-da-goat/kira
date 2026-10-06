# Kira — a local, trainable text Transformer

Kira is a language model foundation implemented in Python and PyTorch, with its own architecture, trainable byte BPE tokenizer, next-token training, resumable checkpoints, local generation, evaluation, and browser interface. It uses **no external LLM inference service and needs no API key**.

**Architecture code does not equal intelligence. Useful language behavior comes from training the model on data.**

No pretrained language model is bundled. A fresh checkout has no weights. Training starts from random initialization. The original example corpus is only for exercising the pipeline; training on it does not create a useful assistant. A finished schedule is labeled **quality unverified**. The browser never invents replies or training measurements.

## Quick Start

Supported Python: **3.9+**, with **3.11 recommended and selected for CPU CI**. The reference CPU constraints target Python 3.9 and 3.11. Node is optional; there are no npm dependencies or build step. Run the commands from the repository root. Internet is needed to install packages; core training/inference then runs locally without credentials.

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

Open **http://127.0.0.1:3000**. Without a checkpoint the server serves the interface, reports missing weights, and returns HTTP 503 for generation. Python is required; there is no hosted inference or JavaScript model fallback. Restart the server to load a new checkpoint. `npm start` is an optional shortcut using `.venv/bin/python`.

The tiny configuration supports CPU development. Choose `--device cpu`, `cuda`, `mps`, or `auto`. CPU/MPS use FP32. CUDA auto selects BF16 when supported, otherwise FP16 with gradient scaling. CLI commands default to two CPU threads. Training prints device/precision and GPU name when available. Larger models need more memory for gradients, AdamW states, activations, and attention; scale gradually and measure on your hardware.

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

The dark sidebar/chat layout retains saved conversations, chat search, copy/export/delete, per-conversation memory, text attachments, and generation settings. Model details/curves display checkpoint measurements. Training runs in Python, never as simulated browser progress. Oversized prompts produce explicit context errors; shorten memory/files/history or train with a larger context.

Memory/files are transient input context, never weight updates. History/memory live in browser localStorage; attachments are transient. Optional **Wikipedia web search** runs only on user request. Results appear separately and can be explicitly attached as model input. Search is not a replacement for generation; autonomous model-directed tool use is not implemented. Search requires internet; training/inference do not. Images, audio, fabricated reasoning traces and automatic chat training are unsupported.

The server binds to loopback by default, serves only frontend assets, bounds request sizes and permits one model generation at a time. It has no authentication; keep it local. `--host 0.0.0.0` supports an isolated development preview. Secrets never belong in frontend code. `.env` is ignored; `.env.example` contains no required variables.

## Structure

```text
frontend/         HTML, CSS, browser client
backend/          HTTP bridge and transient prompt context
kira/model/       Transformer, GQA, RoPE, RMSNorm, SwiGLU
kira/tokenizer/   Byte BPE and tokenizer training CLI
kira/training/    Dataset, AdamW, scheduler, training, checkpoints
kira/inference/   Cached generation and sampling
kira/evaluation/  Held-out metrics, samples and curves
kira/tools/       Optional separate web search
configs/          JSON model/training settings
data/             Original examples; ignored private corpora
checkpoints/      Tracked README; ignored local tokenizer and weights
scripts/          Reproducible pipeline smoke test
tests/            Math, learning, resume and API tests
```

## Verification

```bash
python -m pytest -q -s
python -m compileall -q kira backend tests
python -m pip check
npm run check
bash scripts/smoke.sh runs/smoke
```

Tests verify tokenizer Unicode/special/unknown behavior, persisted vocabularies, RoPE, independent GQA reference math, causality, cache equivalence, logits, gradients, parameter updates, dataset shifting/splitting, save/load/exact resume, seed determinism, EOS, sampling and API boundaries. The overfit test trains for 100 updates and requires loss below 0.1 plus greedy `hello world` followed by EOS. That proves narrow learning, not general intelligence.

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

The repository contains an optional dependency-free `package.json` for startup and JavaScript syntax shortcuts. Python remains the required runtime.

## Limitations

No useful pretrained checkpoint, large-corpus result, instruction-following guarantee, benchmark claim, vision/audio, distributed training, quantization or production serving system is supplied. Configuration scales the architecture, but tokenizer/data/serving implementations need further work for large workloads. CPU tests do not validate CUDA/MPS operation or mixed-precision performance. Keep measured quality separate from implemented architecture.

## License

Source code is available under the [MIT License](LICENSE). The original synthetic text in `data/example/tiny.txt` is separately dedicated to the public domain (CC0), as documented in `data/README.md`. Users are responsible for permissions on their own datasets and redistributed checkpoints.
