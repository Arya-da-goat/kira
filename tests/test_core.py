import pytest
import torch
from kira.config import ModelConfig
from kira.tokenizer.tokenizer import Tokenizer
from kira.model.transformer import KiraTransformer, next_token_loss
from kira.model.attention import CausalSelfAttention
from kira.model.rope import RotaryEmbedding


def small_config(**kwargs):
    return ModelConfig(**dict(dict(vocab_size=260, d_model=32, num_layers=2,
                                  num_attention_heads=4, num_kv_heads=2,
                                  intermediate_size=64, max_seq_len=32), **kwargs))


def test_tokenizer(tmp_path):
    tokenizer = Tokenizer.train(['hello world hello world', '世界 café 🌍'], 280)
    assert tokenizer.vocab_size > 260
    for text in ('hello world', 'unseen 🦊 العربية', '<EOS>', '', '\x00\n'):
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode('', bos=True, eos=True) == [1, 2]
    assert tokenizer.decode([0, 1, 2], skip_special=False) == '<PAD><BOS><EOS>'
    assert tokenizer.decode([999999]) == '<UNK>'
    tokenizer.save(tmp_path / 'tokenizer.json')
    restored = Tokenizer.load(tmp_path / 'tokenizer.json')
    assert restored.fingerprint == tokenizer.fingerprint
    assert restored.encode('hello world') == tokenizer.encode('hello world')
    assert restored.encode('hello') == Tokenizer.train(['hello world hello world', '世界 café 🌍'], 280).encode('hello')


def test_rope():
    rope = RotaryEmbedding(8, 32, 10000)
    x = torch.randn(2, 4, 12, 8)
    y = rope(x)
    assert y.shape == x.shape
    assert torch.equal(y, rope(x))
    assert torch.allclose(y[..., 0, :], x[..., 0, :])
    assert not torch.allclose(y[..., 1:, :], x[..., 1:, :])
    assert torch.allclose(x.square().sum(-1), y.square().sum(-1), atol=1e-5)
    assert torch.equal(rope(x[..., 5:8, :], 5), y[..., 5:8, :])
    with pytest.raises(ValueError):
        rope(x, 30)


@pytest.mark.parametrize('kv_heads', [1, 2, 4])
def test_attention_causal_gqa_and_cache(kv_heads):
    torch.manual_seed(2)
    config = small_config(num_kv_heads=kv_heads)
    attention = CausalSelfAttention(config).eval()
    x = torch.randn(2, 8, 32)
    full, cache = attention(x, use_cache=True)
    assert full.shape == x.shape
    assert cache[0].shape == (2, kv_heads, 8, 8)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10
    assert torch.allclose(attention(changed)[0][:, :4], full[:, :4], atol=1e-6)
    _, prefix = attention(x[:, :5], use_cache=True)
    suffix, _ = attention(x[:, 5:], cache=prefix, use_cache=True)
    assert torch.allclose(suffix, full[:, 5:], atol=1e-6)
    # Independent explicit grouped scaled-dot-product reference.
    q = attention.rope(attention.q_proj(x).view(2, 8, 4, 8).transpose(1, 2))
    k = attention.rope(attention.k_proj(x).view(2, 8, kv_heads, 8).transpose(1, 2))
    v = attention.v_proj(x).view(2, 8, kv_heads, 8).transpose(1, 2)
    outputs = []
    for head in range(4):
        group = head // (4 // kv_heads)
        scores = (q[:, head] @ k[:, group].transpose(-1, -2)) / (8 ** 0.5)
        scores = scores.masked_fill(torch.ones(8, 8, dtype=torch.bool).triu(1), float('-inf'))
        outputs.append(scores.softmax(-1) @ v[:, group])
    expected = attention.out_proj(torch.stack(outputs, 2).reshape(2, 8, 32))
    assert torch.allclose(full, expected, atol=1e-6)


def test_transformer_and_gradients():
    model = KiraTransformer(small_config())
    for length in (1, 7, 32):
        ids = torch.randint(0, 260, (2, length))
        logits, _ = model(ids)
        assert logits.shape == (2, length, 260)
    before = model.embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    next_token_loss(logits, ids).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.embedding.weight)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 33, dtype=torch.long))
    with pytest.raises(ValueError):
        small_config(num_kv_heads=3)


def test_tiny_overfit():
    torch.manual_seed(7)
    tokenizer = Tokenizer()
    sequence = torch.tensor([tokenizer.encode('hello world', bos=True, eos=True)] * 4)
    x, y = sequence[:, :-1], sequence[:, 1:]
    model = KiraTransformer(small_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    initial = next_token_loss(model(x)[0], y).item()
    for _ in range(100):
        loss = next_token_loss(model(x)[0], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    model.eval()
    final = next_token_loss(model(x)[0], y).item()
    assert final < initial * 0.05
    assert final < 0.1
    ids = [tokenizer.bos_id]
    for _ in range(20):
        token = model(torch.tensor([ids]))[0][0, -1].argmax().item()
        ids.append(token)
        if token == tokenizer.eos_id:
            break
    assert tokenizer.decode(ids) == 'hello world'
    assert ids[-1] == tokenizer.eos_id
    print(f'\nTiny overfit: initial loss={initial:.6f}, final loss={final:.6f}, greedy text={tokenizer.decode(ids)!r}')
