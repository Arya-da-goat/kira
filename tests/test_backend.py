from fastapi.testclient import TestClient
from backend.server import create_app, ChatRequest
from backend.context import build_prompt
from kira.training.train import train
from test_pipeline import prepare


def test_no_checkpoint_and_public_assets(tmp_path):
    with TestClient(create_app(tmp_path / 'missing.pt', 'cpu')) as client:
        assert client.get('/').status_code == 200
        assert client.get('/app.js').status_code == 200
        assert not client.get('/api/status').json()['ready']
        assert client.post('/api/chat', json={'prompt': 'hello'}).status_code == 503
        for path in ('/.env', '/checkpoints/latest.pt', '/pyproject.toml', '/data/example/tiny.txt'):
            assert client.get(path).status_code == 404
        assert client.post('/api/chat', content=b'x' * 140000).status_code == 413
        assert client.post('/api/chat', json={'prompt': 'a', 'temperature': -1}).status_code == 422


def test_context_and_search_tool(tmp_path, monkeypatch):
    request = ChatRequest(prompt='now', messages=[{'role': 'user', 'content': 'before'}],
                          memory='remember', context=['document'])
    assert all(word in build_prompt(request) for word in ('now', 'before', 'remember', 'document'))
    assert build_prompt(ChatRequest(prompt='raw completion')) == 'raw completion'
    monkeypatch.setattr('backend.server.search_web', lambda query: [{'title': query, 'snippet': 'fixture', 'url': 'https://en.wikipedia.org/wiki/Test'}])
    with TestClient(create_app(tmp_path / 'missing.pt', 'cpu')) as client:
        result = client.post('/api/tools/search', json={'query': 'test'})
        assert result.status_code == 200 and result.json()['results'][0]['title'] == 'test'


def test_checkpoint_api_generates_without_training(tmp_path, monkeypatch):
    for key in ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    config, data, tokenizer = prepare(tmp_path)
    output = tmp_path / 'run'
    train(config, data, tokenizer, output, device_name='cpu', stop_after=2)
    checkpoint = output / 'latest.pt'
    before = checkpoint.read_bytes()
    with TestClient(create_app(checkpoint, 'cpu')) as client:
        status = client.get('/api/status').json()
        assert status['ready'] and status['training_step'] == 2
        assert status['training_status'] == 'partially trained'
        result = client.post('/api/chat', json={'prompt': 'hi', 'max_new_tokens': 5, 'seed': 7})
        assert result.status_code == 200
        payload = result.json()
        assert payload['generated_tokens'] == len(payload['token_ids']) > 0
        assert payload['tokens_per_second'] > 0
        assert client.post('/api/chat', json={'prompt': 'x' * 1000}).status_code == 422
        assert len(client.get('/api/metrics').json()['history']) == 2
    assert checkpoint.read_bytes() == before
