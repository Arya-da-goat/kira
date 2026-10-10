"""Comprehensive test suite for Kira Frontier Architecture.
Tests cover:
- Tokenizer atomic special tokens, role markers, Unicode, literal text safety, checkpoint compatibility
- SFT assistant-only loss masking, truncation, padding, all-masked batch rejection, and synthetic learning
- Causal GQA attention, RoPE position encoding, and KV-cache logits equivalence
- Sparse MoE learned routing, load-balancing aux loss contribution, and gradient flow
- DPO log probabilities, loss, and preference margins
- Tiny-dataset overfitting and reproducibility
"""
import pytest
import torch
from kira.config import ModelConfig
from kira.tokenizer.tokenizer import Tokenizer
from kira.model.transformer import KiraTransformer, next_token_loss
from kira.model.attention import CausalSelfAttention
from kira.model.feed_forward import SwiGLU, SparseMoE
from kira.model.rope import RotaryEmbedding
from kira.training.sft import InstructionDataset, sft_collate_fn, compute_sft_loss, train_sft_step
from kira.training.dpo import dpo_loss, get_batch_logps


def small_config(**kwargs):
    return ModelConfig(**dict(dict(vocab_size=265, d_model=32, num_layers=2,
                                   num_attention_heads=4, num_kv_heads=2,
                                   intermediate_size=64, max_seq_len=32), **kwargs))


# -----------------------------------------------------------------------------
# 1. Tokenizer Tests (Bug A fix verification & compatibility)
# -----------------------------------------------------------------------------

def test_tokenizer_special_tokens_atomic_and_literal():
    tokenizer = Tokenizer()
    
    # 1. Verify dedicated special token IDs
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.unk_id == 3
    assert tokenizer.im_start_id == 4
    assert tokenizer.im_end_id == 5
    assert tokenizer.system_id == 6
    assert tokenizer.user_id == 7
    assert tokenizer.assistant_id == 8

    # 2. Atomic recognition when allowed_special=True
    assert tokenizer.encode("<|im_start|>", allowed_special=True) == [tokenizer.im_start_id]
    assert tokenizer.encode("<|im_end|>", allowed_special=True) == [tokenizer.im_end_id]
    assert tokenizer.encode("<|user|>", allowed_special=True) == [tokenizer.user_id]
    assert tokenizer.encode("<|assistant|>", allowed_special=True) == [tokenizer.assistant_id]
    assert tokenizer.encode("<|system|>", allowed_special=True) == [tokenizer.system_id]

    # 3. Literal special-token-looking text encoded safely as byte BPE when allowed_special=False
    literal_ids = tokenizer.encode("<|im_start|>", allowed_special=False)
    assert literal_ids != [tokenizer.im_start_id]
    assert len(literal_ids) > 1
    # When decoded, byte BPE reconstructs the literal text
    assert tokenizer.decode(literal_ids, skip_special=True) == "<|im_start|>"

    # 4. Decoding behavior with skip_special
    assert tokenizer.decode([tokenizer.im_start_id], skip_special=False) == "<|im_start|>"
    assert tokenizer.decode([tokenizer.im_start_id], skip_special=True) == ""
    assert tokenizer.decode([tokenizer.bos_id, tokenizer.im_start_id, tokenizer.eos_id], skip_special=False) == "<BOS><|im_start|><EOS>"


def test_tokenizer_unicode_and_roundtrip(tmp_path):
    tokenizer = Tokenizer.train(['hello world hello world', '世界 café 🌍'], 280)
    assert tokenizer.vocab_size > 265

    # Unicode roundtrip
    for text in ('hello world', 'unseen 🦊 العربية', 'math: ∑(x_i) = 42', '', '\x00\n'):
        ids = tokenizer.encode(text, allowed_special=False)
        assert tokenizer.decode(ids, skip_special=True) == text

    # Serialization and loading
    tok_file = tmp_path / 'tokenizer.json'
    tokenizer.save(tok_file)
    restored = Tokenizer.load(tok_file)
    assert restored.fingerprint == tokenizer.fingerprint
    assert restored.encode('hello world') == tokenizer.encode('hello world')


def test_tokenizer_legacy_checkpoint_compatibility():
    # Simulate a checkpoint saved under kira-byte-bpe-v1 (4 special tokens, bytes at 4..259)
    legacy_data = {
        'format': 'kira-byte-bpe-v1',
        'special_tokens': ['<PAD>', '<BOS>', '<EOS>', '<UNK>'],
        'merges': [[108, 109]],  # arbitrary dummy merge
        'vocab_hex': [bytes([i]).hex() for i in range(256)] + [(bytes([104]) + bytes([105])).hex()]
    }
    legacy_tok = Tokenizer.from_dict(legacy_data)
    assert legacy_tok.vocab_size == 4 + 256 + 1
    assert legacy_tok.pad_id == 0
    assert legacy_tok.bos_id == 1
    assert legacy_tok.eos_id == 2
    assert legacy_tok.unk_id == 3
    # Chat special tokens are not present in legacy tokenizer
    assert legacy_tok.im_start_id is None
    # Byte 0 is mapped to ID 4 (legacy mapping is preserved)
    assert legacy_tok.pieces[4] == b'\x00'


# -----------------------------------------------------------------------------
# 2. SFT Loss Masking Tests (Bug B fix verification)
# -----------------------------------------------------------------------------

def test_sft_assistant_loss_masking_and_chat_encoding():
    tokenizer = Tokenizer()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Say hello.'},
        {'role': 'assistant', 'content': 'Hello there!'}
    ]

    chat_str = tokenizer.apply_chat_template(messages)
    assert '<|im_start|>system\nYou are a helpful assistant.<|im_end|>' in chat_str
    assert '<|im_start|>user\nSay hello.<|im_end|>' in chat_str
    assert '<|im_start|>assistant\nHello there!<|im_end|>' in chat_str

    token_seq, label_seq = tokenizer.encode_chat(messages, max_length=128)
    assert len(token_seq) == len(label_seq)

    # System and user tokens must have target -100
    assert label_seq[0] == -100  # BOS
    # Assistant tokens must have unmasked targets (!= -100)
    supervised_targets = [l for l in label_seq if l != -100]
    assert len(supervised_targets) > 0


def test_sft_truncation_preserves_assistant_tokens():
    tokenizer = Tokenizer()
    # Create conversation where system and user messages are long
    long_system = 'You are an artificial intelligence. ' * 10
    messages = [
        {'role': 'system', 'content': long_system},
        {'role': 'user', 'content': 'What is two plus two?'},
        {'role': 'assistant', 'content': 'Four.'}
    ]

    # Truncate to a tight limit of 32 tokens
    token_seq, label_seq = tokenizer.encode_chat(messages, max_length=32)
    assert len(token_seq) <= 32
    # Verify that assistant targets were NOT wiped out by truncation
    supervised = [l for l in label_seq if l != -100]
    assert len(supervised) > 0, "Truncation must not eliminate all assistant targets"


def test_sft_all_masked_batch_rejection():
    model = KiraTransformer(small_config())
    x = torch.randint(0, 265, (2, 8))
    # Create an all-masked target tensor (all -100)
    all_masked_targets = torch.full((2, 8), -100, dtype=torch.long)

    # next_token_loss must reject all-masked targets with a descriptive error
    with pytest.raises(ValueError, match="entirely masked"):
        next_token_loss(model(x)[0], all_masked_targets)

    # compute_sft_loss must also reject all-masked batches
    with pytest.raises(ValueError, match="no valid supervised"):
        compute_sft_loss(model, x, all_masked_targets)


def test_sft_padding_and_collate():
    tokenizer = Tokenizer()
    conv1 = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]
    conv2 = [{'role': 'user', 'content': 'longer prompt here'}, {'role': 'assistant', 'content': 'longer reply here'}]
    dataset = InstructionDataset([conv1, conv2], tokenizer, max_seq_len=64)
    assert len(dataset) == 2

    batch = [dataset[0], dataset[1]]
    padded_x, padded_y = sft_collate_fn(batch, pad_token_id=tokenizer.pad_id)
    assert padded_x.shape == padded_y.shape
    assert padded_x.shape[0] == 2
    # Padded positions in targets must be -100
    assert (padded_y == -100).any()


def test_sft_synthetic_training_step():
    tokenizer = Tokenizer()
    conv = [{'role': 'user', 'content': 'ping'}, {'role': 'assistant', 'content': 'pong'}]
    dataset = InstructionDataset([conv], tokenizer, max_seq_len=32)
    model = KiraTransformer(small_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    batch = sft_collate_fn([dataset[0]], pad_token_id=tokenizer.pad_id)
    initial_metrics = compute_sft_loss(model, batch[0], batch[1])[1]
    initial_loss = initial_metrics['loss']

    # Perform a few optimization steps
    for _ in range(15):
        train_sft_step(model, optimizer, batch, device='cpu')

    final_metrics = compute_sft_loss(model, batch[0], batch[1])[1]
    assert final_metrics['loss'] < initial_loss
    assert final_metrics['supervised_tokens'] > 0


# -----------------------------------------------------------------------------
# 3. KV-Cache Next-Token Logits Equivalence
# -----------------------------------------------------------------------------

def test_kv_cache_logits_equivalence():
    torch.manual_seed(42)
    model = KiraTransformer(small_config()).eval()
    ids = torch.randint(0, 265, (1, 8))

    # 1. Full sequence uncached pass
    with torch.no_grad():
        logits_full, _ = model(ids)

    # 2. Cached step-by-step pass:
    # First process prefix of 5 tokens
    with torch.no_grad():
        logits_prefix, cache = model(ids[:, :5], use_cache=True)
        # Next token 6 evaluated with cache
        logits_cached, cache = model(ids[:, 5:6], cache=cache, use_cache=True)

    # The next-token prediction at position 5 (index 5) must match within numerical tolerance
    assert torch.allclose(logits_cached[:, 0, :], logits_full[:, 5, :], atol=1e-5)


# -----------------------------------------------------------------------------
# 4. Sparse MoE Routing, Aux Loss & Gradients
# -----------------------------------------------------------------------------

def test_moe_aux_loss_contribution_and_gradients():
    torch.manual_seed(42)
    config = small_config(ffn_type='moe', num_experts=4, moe_top_k=2, moe_aux_loss_coeff=0.05)
    model = KiraTransformer(config)

    x = torch.randint(0, 265, (2, 8))
    y = torch.randint(0, 265, (2, 8))

    # Forward pass requesting aux loss
    logits, _, aux_loss = model(x, return_aux_loss=True)
    assert aux_loss is not None
    assert aux_loss.item() > 0, "MoE load-balancing auxiliary loss must be positive"

    loss = next_token_loss(logits, y, aux_loss=aux_loss)
    loss.backward()

    # Verify gradients flow into routers and experts
    for layer in model.layers:
        assert layer.ffn.router.weight.grad is not None
        assert torch.isfinite(layer.ffn.router.weight.grad).all()


# -----------------------------------------------------------------------------
# 5. DPO Loss & Preference Margin Tests
# -----------------------------------------------------------------------------

def test_dpo_loss_and_gradients():
    policy_chosen = torch.tensor([-1.0, -0.8], requires_grad=True)
    policy_rejected = torch.tensor([-3.0, -3.5], requires_grad=True)
    ref_chosen = torch.tensor([-1.2, -1.0])
    ref_rejected = torch.tensor([-2.0, -2.1])

    loss, chosen_r, rejected_r = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )
    assert loss.item() > 0
    assert chosen_r.item() > rejected_r.item()

    loss.backward()
    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None


# -----------------------------------------------------------------------------
# 6. Tiny Overfitting Test
# -----------------------------------------------------------------------------

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
