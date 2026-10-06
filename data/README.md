# Dataset preparation

`example/tiny.txt` is original synthetic test text dedicated to the public domain (CC0). It exercises the tokenizer and training pipeline; it is not a language pretraining corpus. This tiny fixture is the only dataset included in Git.

## Where data belongs

Keep legally obtained corpora in `data/private/`, `data/raw/` or `data/processed/`. Create these directories when needed. Everything under `data/` is ignored except this README and `example/tiny.txt`. Do not force-add private, personal or large datasets.

Run all commands below from the repository root with your Python environment active.

## Accepted formats

- UTF-8 `.txt`: documents separated by blank lines.
- `.json`: a string, `{"text": "..."}` object, or array of either.
- `.jsonl`: one JSON string or text object per line.
- A directory: supported files are read recursively in sorted order.

Example JSONL (each line is a separate document):

```jsonl
{"text": "A small seed grows in the garden."}
{"text": "A blue cup sits on the table."}
```

Prepare at least two distinct nonempty documents. Remove private information, check usage permissions, and group/deduplicate related source material before exporting. Split boundaries are **documents**, so do not split one source into related records that could leak into validation.

The loader normalizes line endings, removes NULs, trims edges and deduplicates exact cleaned documents. It assigns training/validation documents using the configured seed and validation fraction **before** BPE fitting or token chunking. Near-duplicate detection is not implemented; handle that upstream. Tokenizer provenance and dataset fingerprints reject inconsistent training/resume inputs.

## Train on your dataset

Place your prepared files inside `data/private/my-corpus/`, then:

```bash
python -m kira.tokenizer.train --data data/private/my-corpus \
  --config configs/tiny.json --output checkpoints/my-tokenizer.json
python -m kira.training.train --data data/private/my-corpus \
  --config configs/tiny.json --tokenizer checkpoints/my-tokenizer.json \
  --output checkpoints/my-run
python -m kira.evaluation.evaluate --checkpoint checkpoints/my-run/latest.pt \
  --data data/private/my-corpus --output runs/my-evaluation
```

Use the same configuration for tokenizer fitting and training. `configs/tiny.json` is a CPU development configuration; choose scale and training duration based on your data/hardware, then measure quality. Do not reuse a tokenizer trained on validation documents. The current loader and BPE trainer keep the corpus in memory and are intended for small datasets.

## Train from the web interface

Open **Train** in Kira, paste text or upload a `.txt`, `.json`, or `.jsonl` file
(up to 32 KiB), choose settings, and start training. This sends your dataset to
your connected Python backend, which saves it in ignored `runs/web/RUN_ID/`.
The same document cleaning, deduplication and held-out split apply. Each new
web run fits its own tokenizer and starts random model weights. Use the CLI
for larger datasets or resuming an existing checkpoint. GitHub Pages hosts
only the frontend; see the main README's mobile setup instructions.
