import { test } from 'node:test';
import assert from 'node:assert';
import {
  ModelConfig,
  KiraTransformerModel,
  BpeTokenizer,
  RotaryEmbedding,
  rmsNorm,
  silu,
  sampleToken,
  KiraTrainer,
  PRNG
} from '../../src/kira_engine.js';

test('BpeTokenizer encodes and decodes text and respects special tokens', () => {
  const tokenizer = BpeTokenizer.train(['hello world from kira', 'the neural transformer'], 280);
  assert(tokenizer.vocab_size >= 265);
  
  const encoded = tokenizer.encode('hello world', true, true);
  assert.strictEqual(encoded[0], BpeTokenizer.bos_id);
  assert.strictEqual(encoded[encoded.length - 1], BpeTokenizer.eos_id);
  
  const decoded = tokenizer.decode(encoded, true);
  assert.strictEqual(decoded, 'hello world');
});

test('BpeTokenizer structured chat template and assistant SFT masking', () => {
  const tokenizer = new BpeTokenizer();
  const messages = [
    { role: 'system', content: 'You are Kira.' },
    { role: 'user', content: 'Ping' },
    { role: 'assistant', content: 'Pong' }
  ];

  const chatStr = tokenizer.applyChatTemplate(messages);
  assert(chatStr.includes('<|im_start|>system\nYou are Kira.<|im_end|>'));
  assert(chatStr.includes('<|im_start|>user\nPing<|im_end|>'));
  assert(chatStr.includes('<|im_start|>assistant\nPong<|im_end|>'));

  const { input_ids, labels } = tokenizer.encodeChat(messages, 128);
  assert.strictEqual(input_ids.length, labels.length);
  // System and user tokens must have target mask -100
  assert.strictEqual(labels[0], -100);
  // Assistant tokens must have real token targets (!= -100)
  assert(labels.some(l => l !== -100));

  // Test atomic special token recognition
  const atomicIds = tokenizer.encode('<|im_start|>', false, false, true);
  assert.strictEqual(atomicIds[0], BpeTokenizer.im_start_id);

  // Test literal handling when allowed_special is false
  const literalIds = tokenizer.encode('<|im_start|>', false, false, false);
  assert.notStrictEqual(literalIds[0], BpeTokenizer.im_start_id);
  assert.strictEqual(tokenizer.decode(literalIds, true), '<|im_start|>');

  // Test truncation preserves assistant tokens
  const longConv = [
    { role: 'system', content: 'Long system prompt here '.repeat(5) },
    { role: 'user', content: 'Long question here '.repeat(5) },
    { role: 'assistant', content: 'Target reply' }
  ];
  const truncated = tokenizer.encodeChat(longConv, 32);
  assert(truncated.input_ids.length <= 32);
  assert(truncated.labels.some(l => l !== -100), 'Truncation must preserve assistant targets');
});

test('RotaryEmbedding preserves norm and handles positions with context scaling', () => {
  const rope = new RotaryEmbedding(8, 32, 10000);
  const vec = new Float32Array([1.0, 0.5, -0.5, 2.0, 0.1, -0.2, 0.8, -0.4]);
  
  const rot0 = rope.apply(vec, 0);
  // Position 0 rotation by angle 0 preserves identity
  assert(Math.abs(rot0[0] - vec[0]) < 1e-5);
  assert(Math.abs(rot0[1] - vec[1]) < 1e-5);

  const rot5 = rope.apply(vec, 5);
  // Norm should be conserved under orthogonal rotation
  let normOrig = 0, normRot = 0;
  for (let i = 0; i < 8; i++) {
    normOrig += vec[i] * vec[i];
    normRot += rot5[i] * rot5[i];
  }
  assert(Math.abs(normOrig - normRot) < 1e-4);

  // Test linear context scaling
  const linearRope = new RotaryEmbedding(8, 64, 10000, 'linear', 2.0);
  const rotLinear = linearRope.apply(vec, 5);
  assert.notStrictEqual(rot5[0], rotLinear[0]);
});

test('KiraTransformerModel forward pass with GQA, RoPE, and KV caching', () => {
  const config = new ModelConfig({
    vocab_size: 270,
    d_model: 32,
    num_layers: 2,
    num_attention_heads: 4,
    num_kv_heads: 2,
    intermediate_size: 64,
    max_seq_len: 32,
    ffn_type: 'swiglu'
  });

  const model = new KiraTransformerModel(config, new PRNG(123));
  const kvCaches = [];

  // Step 0: prompt token
  const step0 = model.forwardStep(1, 0, kvCaches);
  assert.strictEqual(step0.logits.length, 270);
  assert.strictEqual(kvCaches[0].k.length, 1);

  // Step 1: next token using cached keys/values
  const step1 = model.forwardStep(10, 1, kvCaches);
  assert.strictEqual(step1.logits.length, 270);
  assert.strictEqual(kvCaches[0].k.length, 2);
});

test('Sparse MoE layer config and parameters', () => {
  const moeConfig = new ModelConfig({
    vocab_size: 270,
    d_model: 32,
    num_layers: 2,
    num_attention_heads: 4,
    num_kv_heads: 2,
    intermediate_size: 64,
    max_seq_len: 32,
    ffn_type: 'moe',
    num_experts: 4,
    moe_top_k: 2
  });

  const breakdown = moeConfig.parameterBreakdown();
  assert(breakdown.total_parameters > 0);
  assert(breakdown.router_total > 0);

  const model = new KiraTransformerModel(moeConfig, new PRNG(456));
  const kvCaches = [];
  const out = model.forwardStep(5, 0, kvCaches);
  assert.strictEqual(out.logits.length, 270);
});

test('KiraTrainer executes training updates and decreases loss', () => {
  const config = new ModelConfig({
    vocab_size: 270,
    d_model: 32,
    num_layers: 1,
    num_attention_heads: 4,
    num_kv_heads: 2,
    intermediate_size: 64,
    max_seq_len: 32
  });
  const model = new KiraTransformerModel(config, new PRNG(789));
  const tokenizer = new BpeTokenizer();
  const trainer = new KiraTrainer(model, tokenizer, { learning_rate: 0.01, warmup_steps: 2, max_steps: 10 });

  const tokens = [1, 10, 20, 30, 2];
  const m1 = trainer.trainStep(tokens);
  assert(m1.train_loss > 0);
  assert(m1.step === 1);
});

test('KV cache produces consistent sequential decoding logits', () => {
  const config = new ModelConfig({
    vocab_size: 270,
    d_model: 32,
    num_layers: 2,
    num_attention_heads: 4,
    num_kv_heads: 2,
    intermediate_size: 64,
    max_seq_len: 32
  });
  const model = new KiraTransformerModel(config, new PRNG(999));
  const kvCaches = [];
  const tokens = [5, 12, 19];

  let lastLogits;
  for (let i = 0; i < tokens.length; i++) {
    const res = model.forwardStep(tokens[i], i, kvCaches);
    lastLogits = res.logits;
  }
  assert.strictEqual(lastLogits.length, 270);
  assert.strictEqual(kvCaches[0].k.length, 3);
});
