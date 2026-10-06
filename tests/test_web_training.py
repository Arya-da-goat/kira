"""Real CPU web lifecycle and the boundary exposed to a static Pages client."""
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from backend.server import create_app

ORIGIN = 'https://arya-da-goat.github.io'


def test_browser_access_boundary(tmp_path):
    credential = 'synthetic-test-only-' * 3
    with pytest.raises(ValueError, match='requires'):
        create_app(runs_dir=tmp_path, allowed_origins=[ORIGIN])
    app = create_app(tmp_path / 'absent.pt', 'cpu', runs_dir=tmp_path / 'jobs',
                     access_token=credential, allowed_origins=[ORIGIN])
    headers = {'Origin': ORIGIN, 'Authorization': 'Bearer ' + credential}
    with TestClient(app) as client:
        # Static assets are public; every API (including compute and training data) is protected.
        for asset in ('/', '/app.js', '/lab.js', '/chart.js', '/styles.css'):
            assert client.get(asset).status_code == 200
        for endpoint in ('/api/status', '/api/metrics', '/api/training', '/api/training/example'):
            response = client.get(endpoint, headers={'Origin': ORIGIN})
            assert response.status_code == 401
            assert response.headers['access-control-allow-origin'] == ORIGIN
        assert client.post('/api/training', json={'text': 'a\n\nb'}).status_code == 401
        assert client.get('/api/status', headers=headers).status_code == 200
        assert client.get('/api/status', headers=dict(headers, Origin='https://untrusted.example')).status_code == 403
        preflight = client.options('/api/training', headers={'Origin': ORIGIN,
            'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'authorization,content-type'})
        assert preflight.status_code == 200
        assert preflight.headers['access-control-allow-origin'] == ORIGIN
        assert client.options('/api/training', headers={'Origin': 'https://untrusted.example',
            'Access-Control-Request-Method': 'POST'}).status_code == 400
    with TestClient(create_app(tmp_path / 'absent.pt', 'cpu', runs_dir=tmp_path / 'local')) as client:
        assert client.post('/api/training', json={'text': 'a\n\nb'}, headers={'Origin': 'https://untrusted.example'}).status_code == 403
        # Matching an attacker-controlled Host must not defeat the local-only origin guard.
        assert client.post('/api/training', json={'text': 'a\n\nb'}, headers={
            'Origin': 'http://evil.example', 'Host': 'evil.example'}).status_code == 403


def test_real_web_training_load_generate_and_stop(tmp_path, monkeypatch):
    for key in ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    missing = tmp_path / 'absent.pt'
    runs_dir = tmp_path / 'runs'
    with TestClient(create_app(missing, 'cpu', runs_dir=runs_dir, threads=1)) as client:
        assert not client.get('/api/status').json()['ready']
        data = client.get('/api/training/example').json()
        assert client.post('/api/training', json={'text': 'one document'}).status_code == 422
        assert client.post('/api/training', json={'text': 'é' * 20000}).status_code == 422
        assert client.post('/api/training', json={'text': 'a\n\nb', 'steps': 999999}).status_code == 422
        assert not list(runs_dir.iterdir())
        response = client.post('/api/training', json=dict(data, steps=4, sequence_length=32))
        assert response.status_code == 200, response.text
        run_id = response.json()['id']
        assert client.post('/api/training', json=data).status_code == 409
        assert client.post(f'/api/training/{run_id}/load', json={}).status_code == 409
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            run = client.get('/api/training').json()['runs'][0]
            if run['status'] != 'running':
                break
            time.sleep(0.05)
        assert run['status'] == 'completed', run
        assert run['checkpoint_available']
        assert len(run['history']) == 4
        assert run['history'][-1]['gradient_norm'] > 0
        assert run['history'][-1]['tokens_processed'] > 0
        assert run['history'][-1]['validation_loss'] > 0
        config = json.loads((runs_dir / run_id / 'config.json').read_text())
        assert config['model']['max_seq_len'] == 32 and config['training']['max_steps'] == 4
        checkpoint = runs_dir / run_id / 'latest.pt'
        before = checkpoint.read_bytes()
        assert client.post(f'/api/training/{run_id}/load', json={}).json()['step'] == 4
        assert client.get('/api/status').json()['ready']
        generated = client.post('/api/chat', json={'prompt': 'hi', 'max_new_tokens': 4}).json()
        assert generated['generated_tokens'] == len(generated['token_ids']) > 0
        assert checkpoint.read_bytes() == before
        long_run = client.post('/api/training', json=dict(data, steps=2000)).json()['id']
        assert client.post('/api/chat', json={'prompt': 'hi'}).status_code == 409
        stopped = client.post(f'/api/training/{long_run}/stop', json={}).json()
        assert stopped['status'] == 'cancelled'
        assert client.post('/api/training/not-a-run/load', json={}).status_code == 404
        assert client.get(f'/runs/web/{run_id}/dataset.txt').status_code == 404
    # History and genuine weights survive restart; loading is explicit, not a fake default model.
    with TestClient(create_app(missing, 'cpu', runs_dir=runs_dir)) as client:
        assert len(client.get('/api/training').json()['runs']) == 2
        assert client.post(f'/api/training/{run_id}/load', json={}).status_code == 200
        assert client.get('/api/status').json()['training_step'] == 4


def test_static_pages_entry_and_relative_assets():
    root = Path(__file__).resolve().parents[1]
    assert (root / '.nojekyll').exists()
    assert 'url=./frontend/' in (root / 'index.html').read_text()
    html = (root / 'frontend/index.html').read_text()
    assert 'href="./styles.css"' in html and 'src="./app.js"' in html
    assert 'src="/' not in html and 'href="/' not in html
