import json
from dataclasses import asdict
import pytest
import torch
from torch.utils.data import DataLoader
from kira.config import ModelConfig, TrainingConfig
from kira.tokenizer.tokenizer import Tokenizer
from kira.training.dataset import read_documents, split_documents, data_fingerprint, TokenDataset
from kira.training.train import train
from kira.training.checkpoint import load_model, read_checkpoint
from kira.inference.generate import generate
from kira.inference.sampling import sample_token
from kira.evaluation.metrics import evaluate


def prepare(tmp_path):
    data = tmp_path / 'data.json'
    documents = [f'hello world {i}. this is original tiny example {i}.' for i in range(12)]
    data.write_text(json.dumps(documents))
    tc = TrainingConfig(batch_size=2, gradient_accumulation_steps=2, max_steps=8,
                        max_epochs=20, warmup_steps=2, eval_interval=2, save_interval=4)
    mc = ModelConfig(d_model=32, num_layers=1, num_attention_heads=4, num_kv_heads=2,
                     intermediate_size=64, max_seq_len=24, dropout=0.1)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'model': asdict(mc), 'training': asdict(tc)}))
    training, validation = split_documents(documents, tc.validation_fraction, tc.seed)
    tokenizer = Tokenizer.train(training, 280)
    tokenizer_path = tmp_path / 'tokenizer.json'
    tokenizer.save(tokenizer_path)
    (tmp_path / 'tokenizer.json.provenance.json').write_text(json.dumps({
        'data_fingerprint': data_fingerprint(training, validation),
        'tokenizer_fingerprint': tokenizer.fingerprint}))
    return config, data, tokenizer_path


def test_dataset_formats_split_and_shift(tmp_path):
    (tmp_path / 'one.txt').write_text('hello\r\nworld\n\nsecond text\n\nhello\nworld')
    (tmp_path / 'two.json').write_text(json.dumps([{'text': 'third text'}, 'fourth text']))
    (tmp_path / 'three.jsonl').write_text('"fifth text"\n{"text": "sixth text"}\n')
    docs = read_documents(tmp_path)
    assert len(docs) == 6
    train_docs, val_docs = split_documents(docs, .3, 42)
    assert set(train_docs).isdisjoint(val_docs)
    assert (train_docs, val_docs) == split_documents(docs, .3, 42)
    tokenizer = Tokenizer()
    dataset = TokenDataset(['hello world'], tokenizer, 4)
    targets = torch.cat([y[y != -100] for _, y in dataset]).tolist()
    assert targets == tokenizer.encode('hello world', eos=True)
    x, y = dataset[0]
    assert x[0].item() == tokenizer.bos_id
    assert torch.equal(x[1:], y[:-1])
    assert sum((y != -100).sum().item() for _, y in dataset) == len(targets)
    with pytest.raises(ValueError):
        split_documents(['same', 'same'], .2, 42)


@pytest.mark.parametrize("interruption", [3, 4])
def test_training_resume_exact_and_evaluate(tmp_path, interruption):
    config, data, tokenizer_path = prepare(tmp_path)
    full = tmp_path / 'full'
    partial = tmp_path / 'partial'
    train(config, data, tokenizer_path, full, device_name='cpu')
    train(config, data, tokenizer_path, partial, device_name='cpu', stop_after=interruption)
    saved = read_checkpoint(partial / 'latest.pt')
    assert saved['state']['step'] == interruption and not saved['state']['run_complete']
    assert saved['optimizer']['state'] and saved['scheduler'] and saved['rng']
    train(output=partial, resume=partial / 'latest.pt', device_name='cpu')
    original = read_checkpoint(full / 'latest.pt')
    resumed = read_checkpoint(partial / 'latest.pt')
    assert original['state']['tokens_processed'] == resumed['state']['tokens_processed']
    assert original['state']['step'] == resumed['state']['step'] == 8
    assert original['state']['run_complete']
    for key in original['model']:
        assert torch.equal(original['model'][key], resumed['model'][key]), key
    for a, b in zip(original['state']['history'], resumed['state']['history']):
        assert a['train_loss'] == b['train_loss']
        assert a['learning_rate'] == b['learning_rate']
    assert (partial / 'best.pt').exists() and (partial / 'step-00000004.pt').exists()
    model, tokenizer, _ = load_model(partial / 'latest.pt')
    loader = DataLoader(TokenDataset(['a short test'], tokenizer, 24), batch_size=1)
    metrics = evaluate(model, loader, torch.device('cpu'))
    assert metrics['perplexity'] > 0 and 0 <= metrics['token_accuracy'] <= 1
    result = generate(model, tokenizer, 'hi', max_new_tokens=5, seed=9)
    again = generate(model, tokenizer, 'hi', max_new_tokens=5, seed=9)
    assert result['token_ids'] == again['token_ids']
    assert result['generated_tokens'] == len(result['token_ids'])
    assert all(0 <= token < tokenizer.vocab_size for token in result['token_ids'])
    assert result['prompt_tokens'] == len(tokenizer.encode('hi', bos=True))
    with pytest.raises(ValueError, match='Shorten'):
        generate(model, tokenizer, '🦊' * 100)
    data.write_text(json.dumps(['changed data', 'different data']))
    with pytest.raises(ValueError, match='Dataset'):
        train(data=data, output=partial, resume=partial / 'latest.pt', device_name='cpu')


def test_eos_and_sampling():
    class EosModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))
            self.config = ModelConfig(vocab_size=260, max_seq_len=16)
        def forward(self, ids, **kwargs):
            logits = torch.zeros(1, ids.shape[1], 260)
            logits[..., 2] = 100
            return logits, []
    result = generate(EosModel(), Tokenizer(), 'hi', temperature=0)
    assert result['token_ids'] == [2] and result['text'] == '' and result['stop_reason'] == 'eos'
    assert sample_token(torch.arange(10).float(), [], top_k=1) == 9
    assert sample_token(torch.tensor([-1., 2., 10., 3.]), [], top_p=.01, forbidden=()) == 2
    assert sample_token(torch.tensor([1., 2., 3., 4.]), [3], temperature=0, repetition_penalty=10, forbidden=()) == 2
    for kwargs in ({'temperature': -1}, {'top_p': 0}, {'top_k': -1}, {'repetition_penalty': 0}):
        with pytest.raises(ValueError):
            sample_token(torch.ones(10), [], **kwargs)
