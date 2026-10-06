"""One bounded local training process; generated data/weights stay outside Git."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator
from backend.training_worker import write_json
from kira.training.dataset import read_documents, split_documents

ROOT = Path(__file__).resolve().parents[1]


class TrainingRequest(BaseModel):
    text: str = Field(min_length=1, max_length=64000)
    @field_validator('text')
    @classmethod
    def limit_utf8(cls, value):
        if len(value.encode('utf-8')) > 32768:
            raise ValueError('Web datasets are limited to 32 KiB; use the Python CLI for larger data')
        return value

    format: Literal['txt', 'json', 'jsonl'] = 'txt'
    steps: int = Field(default=100, ge=2, le=2000)
    batch_size: int = Field(default=4, ge=1, le=8)
    learning_rate: float = Field(default=0.001, ge=0.00001, le=0.01, allow_inf_nan=False)
    sequence_length: int = Field(default=128, ge=32, le=256)
    vocab_size: int = Field(default=384, ge=260, le=512)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class TrainingJobs:
    """Call under the server's model lock to serialize training and generation."""
    def __init__(self, directory, device, threads=2):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device, self.threads = str(device), threads
        self.process = None
        self.active_id = None
        # Workers are stopped on normal shutdown. A crash leaves an interrupted run;
        # never claim a former process finished or load a checkpoint during a write.
        for path in self.directory.glob('*/job.json'):
            job = json.loads(path.read_text())
            if job['status'] == 'running':
                job['status'] = 'interrupted'
                write_json(path, job)

    def path(self, run_id):
        if not re.fullmatch(r'[a-f0-9]{32}', run_id):
            raise ValueError('Invalid run ID')
        path = self.directory / run_id
        if not (path / 'job.json').is_file():
            raise FileNotFoundError('Training run not found')
        return path

    def _finish(self, status):
        path = self.path(self.active_id) / 'job.json'
        job = json.loads(path.read_text())
        job['status'] = status
        job['ended_at'] = datetime.now(timezone.utc).isoformat()
        write_json(path, job)
        self.process = self.active_id = None

    def busy(self):
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                return True
            self._finish('completed' if code == 0 else 'failed')
        return False

    def start(self, request):
        if self.busy():
            raise RuntimeError('A training run is already active')
        run_id = uuid4().hex
        directory = self.directory / run_id
        directory.mkdir()
        data = directory / ('dataset.' + request.format)
        data.write_text(request.text, encoding='utf-8')
        try:
            training, validation = split_documents(read_documents(data), 0.2, request.seed)
        except (ValueError, RecursionError) as exc:
            # No model or checkpoint exists yet. Remove only this invalid upload.
            data.unlink()
            directory.rmdir()
            raise ValueError('Use valid text/JSON with at least two distinct documents. ' + str(exc)[:200]) from exc
        config = json.loads((ROOT / 'configs/tiny.json').read_text())
        config['model'].update(vocab_size=request.vocab_size, max_seq_len=request.sequence_length)
        config['training'].update(max_steps=request.steps, max_epochs=request.steps,
                                  batch_size=request.batch_size, learning_rate=request.learning_rate,
                                  seed=request.seed, warmup_steps=min(10, request.steps - 1),
                                  eval_interval=min(10, request.steps), save_interval=max(10, request.steps // 4))
        write_json(directory / 'config.json', config)
        job = {'id': run_id, 'status': 'running', 'format': request.format,
               'created_at': datetime.now(timezone.utc).isoformat(), 'max_steps': request.steps,
               'train_documents': len(training), 'validation_documents': len(validation)}
        write_json(directory / 'job.json', job)
        try:
            with (directory / 'worker.log').open('wb') as log:
                self.process = subprocess.Popen(
                    [sys.executable, '-m', 'backend.training_worker', '--run-dir', str(directory),
                     '--device', self.device, '--threads', str(self.threads)],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        except OSError:
            job['status'] = 'failed'
            write_json(directory / 'job.json', job)
            raise
        self.active_id = run_id
        return self.snapshot(run_id)

    def snapshot(self, run_id):
        self.busy()
        directory = self.path(run_id)
        result = json.loads((directory / 'job.json').read_text())
        phase = directory / 'phase.json'
        result.update(json.loads(phase.read_text()) if phase.exists() else {'phase': 'Starting Python worker'})
        history = []
        metrics = directory / 'metrics.jsonl'
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # The last record can still be in flight.
        result['history'] = history
        result['checkpoint_available'] = result['status'] in ('completed', 'cancelled', 'failed') and (directory / 'latest.pt').exists()
        return result

    def list(self):
        self.busy()
        files = sorted(self.directory.glob('*/job.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        return [self.snapshot(path.parent.name) for path in files[:20]]

    def stop(self, run_id):
        self.path(run_id)
        if self.busy() and self.active_id == run_id:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self._finish('cancelled')
        return self.snapshot(run_id)

    def close(self):
        if self.busy():
            self.stop(self.active_id)
