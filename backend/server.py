"""Local HTTP bridge to a single explicitly selected Kira checkpoint."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import os
import secrets
from dataclasses import asdict
from pathlib import Path
import threading
from typing import Annotated, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints
import uvicorn
from backend.context import build_prompt
from backend.security import BackendAccess
from backend.training_jobs import TrainingJobs, TrainingRequest
from kira.hardware import select_device, describe
from kira.inference.generate import generate
from kira.training.checkpoint import load_model, training_status
from kira.tools.search import search_web

ROOT = Path(__file__).resolve().parents[1]


class Message(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(max_length=16000)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=16000)
    messages: list[Message] = Field(default_factory=list, max_length=40)
    memory: str = Field(default='', max_length=8000)
    context: list[Annotated[str, StringConstraints(max_length=32768)]] = Field(default_factory=list, max_length=8)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0, le=5, allow_inf_nan=False)
    top_k: int = Field(default=40, ge=0, le=100000)
    top_p: float = Field(default=0.95, gt=0, le=1, allow_inf_nan=False)
    repetition_penalty: float = Field(default=1, gt=0, le=10, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**63 - 1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)


class BodyLimit:
    """Bound request bodies before JSON parsing, including chunked requests."""
    def __init__(self, app, limit=128 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body = message.get('body', b'')
            chunks.append(body)
            size += len(body)
            if size > self.limit:
                await send({'type': 'http.response.start', 'status': 413,
                            'headers': [(b'content-type', b'application/json')]})
                await send({'type': 'http.response.body', 'body': b'{"detail":"Request body too large"}'})
                return
            if not message.get('more_body', False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': b''.join(chunks), 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(checkpoint_path='checkpoints/latest.pt', device_name='auto', *,
               runs_dir='runs/web', access_token=None, allowed_origins=(), threads=2):
    if allowed_origins and not access_token:
        raise ValueError('Cross-origin access requires a backend access token')
    device = select_device(device_name)
    jobs = TrainingJobs(runs_dir, device, threads)

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.close()

    app = FastAPI(title='Kira local model', docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(BodyLimit)
    app.add_middleware(BackendAccess, token=access_token, origins=allowed_origins)
    # CORS is outermost so preflights work and authentication failures remain readable.
    app.add_middleware(CORSMiddleware, allow_origins=list(allowed_origins),
                       allow_methods=['GET', 'POST'], allow_headers=['Authorization', 'Content-Type'])
    model = tokenizer = checkpoint = None
    error: Optional[str] = None
    if Path(checkpoint_path).exists():
        try:
            model, tokenizer, checkpoint = load_model(checkpoint_path, device)
        except Exception:
            error = 'Checkpoint could not be loaded. Check the file locally and restart the server.'
    else:
        error = 'No weights loaded. Open Train, run a tiny training job, then load its checkpoint into chat.'
    lock = threading.Lock()

    @app.get('/api/status')
    def status():
        # Never read a mixture of old/new model state during checkpoint replacement.
        if not lock.acquire(blocking=False):
            return {'ready': False, 'busy': True, 'training_active': False,
                    'detail': 'The model is generating or loading weights. Try again shortly.'}
        try:
            active = jobs.busy()
            operational = {'training_active': active, 'busy': False}
            if model is None:
                return dict(operational, ready=False, training_status='untrained / no loaded weights', detail=error)
            return dict(operational, ready=True, model='Kira dense text Transformer',
                        training_status=training_status(checkpoint['state']),
                        training_step=checkpoint['state']['step'],
                        parameters=sum(p.numel() for p in model.parameters()),
                        config=asdict(model.config), hardware=describe(device, 'fp32'),
                        latest_metrics=checkpoint['state']['history'][-1] if checkpoint['state']['history'] else None)
        finally:
            lock.release()

    @app.get('/api/metrics')
    def metrics():
        return {'history': checkpoint['state']['history'] if checkpoint else []}

    @app.get('/api/training/example')
    def example():
        return {'text': (ROOT / 'data/example/tiny.txt').read_text(), 'format': 'txt'}

    # All job state transitions and model replacement share the inference lock.
    def acquire():
        if not lock.acquire(blocking=False):
            raise HTTPException(429, 'Model is busy; retry shortly')

    @app.get('/api/training')
    def training_runs():
        acquire()
        try:
            return {'runs': jobs.list()}
        finally:
            lock.release()

    @app.post('/api/training')
    def start_training(request: TrainingRequest):
        acquire()
        try:
            return jobs.start(request)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            lock.release()

    @app.post('/api/training/{run_id}/stop')
    def stop_training(run_id: str):
        acquire()
        try:
            return jobs.stop(run_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404, str(exc)) from exc
        finally:
            lock.release()

    @app.post('/api/training/{run_id}/load')
    def load_training(run_id: str):
        nonlocal model, tokenizer, checkpoint, error
        acquire()
        try:
            if jobs.busy():
                raise HTTPException(409, 'Stop or finish training before loading weights')
            snapshot = jobs.snapshot(run_id)
            if not snapshot['checkpoint_available']:
                raise HTTPException(409, 'This run has no saved checkpoint available')
            # Only checkpoints created by this server; never an arbitrary uploaded pickle/path.
            loaded = load_model(jobs.path(run_id) / 'latest.pt', device)
            model, tokenizer, checkpoint = loaded
            error = None
            return {'loaded': True, 'step': checkpoint['state']['step']}
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404, str(exc)) from exc
        finally:
            lock.release()

    @app.post('/api/chat')
    def chat(request: ChatRequest):
        if model is None:
            raise HTTPException(503, error)
        if not lock.acquire(blocking=False):
            raise HTTPException(429, 'Model is busy generating. Retry after the current request.')
        try:
            if jobs.busy():
                raise HTTPException(409, 'Training is running. Stop or finish it before generating.')
            result = generate(model, tokenizer, build_prompt(request), request.max_new_tokens,
                              request.temperature, request.top_k, request.top_p,
                              request.repetition_penalty, request.seed)
            return dict(result, model='Kira', training_status=training_status(checkpoint['state']))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            lock.release()

    @app.post('/api/tools/search')
    def search(request: SearchRequest):
        try:
            return {'results': search_web(request.query), 'provider': 'Wikipedia search'}
        except Exception as exc:
            raise HTTPException(502, 'Search is unavailable. Local inference still works.') from exc

    # Serve only public assets, never repository files, checkpoints, or .env.
    @app.get('/')
    def index():
        return FileResponse(ROOT / 'frontend/index.html')

    @app.get('/{asset:path}')
    def frontend_asset(asset: str):
        public = (ROOT / 'frontend').resolve()
        path = (public / asset).resolve()
        if not path.is_relative_to(public) or not path.is_file() or path.suffix not in {'.js', '.css', '.svg'}:
            raise HTTPException(404, 'Asset not found')
        media = {'.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml'}
        return FileResponse(path, media_type=media[path.suffix])

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/latest.pt')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=3000, type=int)
    parser.add_argument('--threads', default=2, type=int)
    parser.add_argument('--allow-origin', action='append', default=[],
                        help='Exact Pages origin, e.g. https://arya-da-goat.github.io (no path)')
    parser.add_argument('--token-file', default='runs/backend-access.txt',
                        help='Private generated access token; never put this in Git')
    parser.add_argument('--runs-dir', default='runs/web')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    from urllib.parse import urlsplit
    for origin in args.allow_origin:
        parsed = urlsplit(origin)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or origin != f'{parsed.scheme}://{parsed.netloc}':
            parser.error('--allow-origin must be an exact HTTP(S) origin without a path')
    token = None
    if args.host not in ('127.0.0.1', 'localhost', '::1') or args.allow_origin:
        token_path = Path(args.token_file)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            token = token_path.read_text().strip()
            if len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
                parser.error('Token file must contain a private token of at least 32 ASCII characters')
        else:
            token = secrets.token_urlsafe(32)
            with os.fdopen(fd, 'w') as file:
                file.write(token)
        print(f'Backend access token (keep private; paste into Connect): {token}', flush=True)
    torch.set_num_threads(args.threads)
    app = create_app(args.checkpoint, args.device, runs_dir=args.runs_dir,
                     access_token=token, allowed_origins=args.allow_origin, threads=args.threads)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
