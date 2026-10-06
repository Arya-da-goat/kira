#!/usr/bin/env bash
# CPU quick start for a Linux Codespace. Reuses a working project environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import torch, fastapi, uvicorn; import kira' >/dev/null 2>&1; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/python -m pip install -r requirements.txt -c constraints-cpu.txt
fi
exec .venv/bin/python -m backend.server --host 0.0.0.0 --device cpu \
  --allow-origin "${1:-https://arya-da-goat.github.io}" --port "${2:-3000}"
