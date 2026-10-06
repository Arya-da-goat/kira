# Local checkpoints (not version-controlled)

This directory contains documentation in Git. Model weights, tokenizer files, training logs and nested run directories are ignored. No checkpoint is needed to run the automated tests: they create tiny temporary models and clean up through pytest's temporary directories.

## Create your own weights

Run from the repository root after installing the Python environment:

```bash
python -m kira.tokenizer.train --data data/example
python -m kira.training.train --data data/example
python -m kira.inference.generate --checkpoint checkpoints/latest.pt --prompt hello
python -m backend.server --checkpoint checkpoints/latest.pt
```

These commands produce a **development/test checkpoint**, not a useful pretrained assistant. For actual Kira training, use a legally obtained, adequately sized corpus, a suitable configuration and held-out evaluation. A completed schedule does not establish quality.

Training saves `latest.pt`, `best.pt`, periodic `step-XXXXXXXX.pt` files, `tokenizer.json`, and `metrics.jsonl`. Checkpoints embed tokenizer data, model configuration, weights, optimizer/scheduler/scaler states, RNG state and training progress. See the root README for resume commands.

## Existing checkpoints

There is no official pretrained checkpoint download. Place your own trusted, compatible Kira checkpoint at `checkpoints/latest.pt`, or choose another file with `--checkpoint`. Arbitrary third-party model formats are not compatible. Never load an untrusted checkpoint.

For a relocated checkpoint, run commands from the repository root and pass `--data data/private/my-corpus` to resume/evaluate when the saved dataset location differs. Identical dataset content and splits are required. Newly created runs preserve relative dataset paths when given relative paths.

Share weights separately from source control, with a model card covering training data permissions, configuration, steps, measured evaluation, limitations and a checksum. Do not force-add weights to Git. This applies to small test weights too; tests generate their own fixtures.

Browser-started training saves checkpoints in ignored `runs/web/RUN_ID/`.
Use **Train → Load into chat** to load a saved checkpoint without restarting
the backend. Stopping a worker preserves only checkpoints already written;
the last measured step can be newer than the last saved step. The UI reports
the actual loaded step. These files also work with the inference/evaluation
CLI. Keep backups outside Git, especially before deleting a Codespace.
