"""Comprehensive tests for Kira Frontier Architecture: Attention, RoPE, GQA, MoE, SFT, DPO, and Overfitting."""
import pytest
import torch
from kira.config import ModelConfig
from kira.tokenizer.tokenizer import Tokenizer
from kira.model.transformer import KiraTransformer, next_token_loss
from kira.model.attention import CausalSelfAttention
from kira.model.feed_forward import SwiGLU, SparseMoE
from kira.model.rope import RotaryEmbedding
from kira.training.sft import InstructionDataset, sft_collate_fn, compute_sft_loss
from kira.training.dpo import dpo_loss, get_batch_logps


def small_config(**kwargs):
    return ModelConfig(**dict(dict(vocab_size=265, d_model=32, num_layers=2,
                                   num_attention_heads=4, num_kv_heads=2,
                                   intermediate_size=64, max_seq_len=32), **kwargs))


def test_tokenizer(tmp_path):
    tokenizer = Tokenizer.train(['hello world hello world', '世界 café 🌍'], 280)
    assert tokenizer.vocab_size > 265
    for text in ('hello world', 'unseen 🦊 العربية', '<EOS>', '', '\x00\n'):
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode('', bos=True, eos=True) == [1, 2]
    assert tokenizer.decode([0, 1, 2], skip_special=False) == '<PAD><BOS><EOS>'
    assert tokenizer.decode([999999]) == '<UNK>'
    
    # Test chat template and SFT encoding
    messages = [
        {'role': 'system', 'content': 'You are Kira.'},
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello!'}
    ]
    chat_str = tokenizer.apply_chat_template(messages)
    assert '<|im_start|>system' in chat_str
    assert '<|im_start|>user' in chat_str
    assert '<|im_start|>assistant' in chat_str
    
    input_ids, labels = tokenizer.encode_chat(messages, max_length=64)
    assert len(input_ids) == len(labels)
    # Ensure system and user tokens are masked (-100) and assistant tokens have valid IDs
    assert labels[0] == -100  # BOS
    assert any(l != -100 for l in labels)  # Assistant tokens are unmasked

    tokenizer.save(tmp_path / 'tokenizer.json')
    restored = Tokenizer.load(tmp_path / 'tokenizer.json')
    assert restored.fingerprint == tokenizer.fingerprint
    assert restored.encode('hello world') == tokenizer.encode('hello world')


def test_rope_and_scaling():
    rope = RotaryEmbedding(8, 32, 10000)
    x = torch.randn(2, 4, 12, 8)
    y = rope(x)
    assert y.shape == x.shape
    assert torch.equal(y, rope(x))
    assert torch.allclose(y[..., 0, :], x[..., 0, :])
    assert not torch.allclose(y[..., 1:, :], x[..., 1:, :])
    # Norm preservation under rotation
    assert torch.allclose(x.square().sum(-1), y.square().sum(-1), atol=1e-5)
    # Sliced position offset equivalence
    assert torch.equal(rope(x[..., 5:8, :], 5), y[..., 5:8, :])
    with pytest.raises(ValueError):
        rope(x, 30)

    # Test RoPE context extension scaling (linear and NTK)
    linear_rope = RotaryEmbedding(8, 64, 10000, scaling_type='linear', scaling_factor=2.0)
    y_linear = linear_rope(x)
    assert y_linear.shape == x.shape
    assert not torch.equal(y, y_linear)

    ntk_rope = RotaryEmbedding(8, 64, 10000, scaling_type='ntk', scaling_factor=2.0)
    y_ntk = ntk_rope(x)
    assert y_ntk.shape == x.shape


@pytest.mark.parametrize('kv_heads', [1, 2, 4])
def test_attention_causal_gqa_and_cache(kv_heads):
    torch.manual_seed(2)
    config = small_config(num_kv_heads=kv_heads)
    attention = CausalSelfAttention(config).eval()
    x = torch.randn(2, 8, 32)
    full, cache = attention(x, use_cache=True)
    assert full.shape == x.shape
    assert cache[0].shape == (2, kv_heads, 8, 8)

    # Causal masking verification: changing future tokens does not affect earlier positions
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10
    assert torch.allclose(attention(changed)[0][:, :4], full[:, :4], atol=1e-6)

    # KV-cache equivalence: step-by-step cached attention matches full sequence
    _, prefix = attention(x[:, :5], use_cache=True)
    suffix, _ = attention(x[:, 5:], cache=prefix, use_cache=True)
    assert torch.allclose(suffix, full[:, 5:], atol=1e-6)


def test_moe_routing_and_gradients():
    torch.manual_seed(42)
    config = small_config(ffn_type='moe', num_experts=4, moe_top_k=2, moe_aux_loss_coeff=0.01)
    moe = SparseMoE(config)
    x = torch.randn(2, 8, 32, requires_grad=True)

    out, aux_loss = moe(x)
    assert out.shape == x.shape
    assert aux_loss > 0  # Load-balancing auxiliary loss is positive

    # Test backward pass and gradient flow through routing and expert weights
    loss = out.sum() + aux_loss
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert moe.router.weight.grad is not None and torch.isfinite(moe.router.weight.grad).all()
    for expert in moe.experts:
        assert expert.gate.weight.grad is not None


def test_transformer_and_gradients():
    for ffn_type in ('swiglu', 'moe'):
        model = KiraTransformer(small_config(ffn_type=ffn_type))
        for length in (1, 7, 32):
            ids = torch.randint(0, 265, (2, length))
            logits, _ = model(ids)
            assert logits.shape == (2, length, 265)
        before = model.embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        next_token_loss(logits, ids).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        assert not torch.equal(before, model.embedding.weight)


def test_sft_assistant_loss_masking():
    tokenizer = Tokenizer()
    messages = [
        {'role': 'user', 'content': 'ping'},
        {'role': 'assistant', 'content': 'pong'}
    ]
    dataset = InstructionDataset([messages], tokenizer, max_seq_len=32)
    x, y = dataset[0]
    model = KiraTransformer(small_config())
    loss, metrics = compute_sft_loss(model, x.unsqueeze(0), y.unsqueeze(0))
    assert loss > 0
    assert metrics['supervised_tokens'] > 0


def test_dpo_loss():
    # Synthetic log probabilities for chosen vs rejected responses
    policy_chosen = torch.tensor([-1.2, -0.8])
    policy_rejected = torch.tensor([-3.5, -4.0])
    ref_chosen = torch.tensor([-1.5, -1.0])
    ref_rejected = torch.tensor([-2.0, -1.8])

    loss, chosen_r, rejected_r = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )
    assert loss > 0
    # Chosen implicit reward should exceed rejected reward
    assert chosen_r > rejected_r


def test_parameter_and_memory_estimation():
    cfg = small_config(ffn_type='moe', num_experts=4, moe_top_k=2)
    breakdown = cfg.parameter_breakdown()
    assert breakdown['total_parameters'] > 0
    assert breakdown['embedding'] == cfg.vocab_size * cfg.d_model
    assert breakdown['tied_embeddings'] is True

    mem = cfg.estimate_memory_bytes(batch_size=2, seq_len=32)
    assert mem['weights_bytes'] > 0
    assert mem['kv_cache_bytes'] > 0
    assert mem['activation_bytes'] > 0


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
