#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root after activating the Python environment.
run_dir="${1:-runs/smoke}"
if [ -e "$run_dir" ]; then
  echo "Choose a fresh output directory: $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir"
python -m kira.tokenizer.train --data data/example --output "$run_dir/tokenizer.json"
python -m kira.training.train --data data/example --tokenizer "$run_dir/tokenizer.json" --output "$run_dir" --stop-after 40
python -m kira.training.train --resume "$run_dir/latest.pt"
python -m kira.inference.generate --checkpoint "$run_dir/latest.pt" --prompt hello --seed 42
python -m kira.evaluation.evaluate --checkpoint "$run_dir/latest.pt" --output "$run_dir/evaluation"
