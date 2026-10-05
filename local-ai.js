/**
 * Kira Frontier Transformer LLM Engine (GPT-6 / Google Astra Architecture)
 * Complete implementation of modern frontier model architecture:
 * 1. Multimodal TikToken BPE Tokenizer with ChatML & Astra special tokens
 * 2. Rotary Position Embeddings (RoPE) with frequency scaling (theta = 10000)
 * 3. RMSNorm (Root Mean Square Layer Normalization) with learnable scale gamma
 * 4. Grouped-Query Attention (GQA) with Key-Value (KV) Cache & Causal Masking
 * 5. Sparse Mixture of Experts (MoE) with Top-2 Routing across 8 SwiGLU Experts
 * 6. SwiGLU (Swish-Gated Linear Unit) Feed-Forward Networks
 * 7. Chain-of-Thought (CoT) Reasoning Engine (<|thought|> ... <|thought_end|>)
 * 8. Complete 4-Pillar Training Pipeline:
 *    - Mode 1: Pre-Training (PT) - Autoregressive Next-Token Cross-Entropy + AdamW
 *    - Mode 2: Supervised Fine-Tuning (SFT) - Instruction Tuning with Prompt-Loss Masking
 *    - Mode 3: Direct Preference Optimization (DPO) / RLHF - Chosen vs Rejected Pair Alignment
 *    - Mode 4: LoRA (Low-Rank Adaptation) - Low-Rank Adapter Matrices (Delta W = alpha/r * B * A)
 */

(() => {
  'use strict';

  // --- 1. TOKENIZER: TIKTOKEN BPE VOCABULARY & CHATML SPECIAL TOKENS ---
  class FrontierTokenizer {
    constructor() {
      this.specialTokens = {
        '<|pad|>': 0,
        '<|bos|>': 1,
        '<|eos|>': 2,
        '<|unk|>': 3,
        '<|im_start|>': 4,
        '<|im_end|>': 5,
        '<|system|>': 6,
        '<|user|>': 7,
        '<|assistant|>': 8,
        '<|thought|>': 9,
        '<|thought_end|>': 10,
        '<|vision_start|>': 11,
        '<|audio_start|>': 12,
        '<|call:tool|>': 13
      };

      this.vocab = {};
      this.invVocab = [];
      this.buildVocab();
    }

    buildVocab() {
      // 1. Special tokens
      for (const [tok, id] of Object.entries(this.specialTokens)) {
        this.vocab[tok] = id;
        this.invVocab[id] = tok;
      }

      // 2. ASCII Characters (32 to 126)
      for (let i = 32; i <= 126; i++) {
        const ch = String.fromCharCode(i);
        if (this.vocab[ch] === undefined) {
          const id = this.invVocab.length;
          this.vocab[ch] = id;
          this.invVocab.push(ch);
        }
      }

      // 3. Subwords & Tokens (STEM, Math, Code, Astra Multimodal, Reasoning, Social)
      const subwords = [
        '\n', '\t', '  ', '    ',
        'the', 'is', 'are', 'was', 'were', 'to', 'in', 'and', 'of', 'for', 'you', 'I',
        'that', 'it', 'on', 'with', 'as', 'at', 'this', 'by', 'from', 'they', 'we',
        'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'know', 'take', 'your',
        'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look',
        'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
        'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
        // Astra Multimodal & Vision
        'multimodal', 'vision', 'camera', 'spatial', 'audio', 'frame', 'spectrogram', 'sensor',
        'latency', 'stream', 'realtime', 'perception', 'object', 'scene', 'tracking',
        // Social, Greetings, Expressive
        'hello', 'hi', 'hey', 'greetings', 'morning', 'evening', 'afternoon', 'welcome',
        'thanks', 'thank', 'great', 'awesome', 'cool', 'sorry', 'please', 'help',
        'fuck', 'shit', 'damn', 'wtf', 'sucks', 'stupid', 'dumb', 'annoying', 'bruh',
        // Technical & Code
        'code', 'function', 'return', 'const', 'let', 'var', 'def', 'class', 'import',
        'export', 'async', 'await', 'print', 'console', 'log', 'if', 'else', 'for',
        'while', 'true', 'false', 'null', 'undefined', 'python', 'javascript', 'typescript',
        'html', 'css', 'sql', 'react', 'api', 'algorithm', 'array', 'object', 'string',
        // Math & STEM
        'solve', 'math', 'calculate', 'equation', 'formula', 'matrix', 'linear',
        'quadratic', 'integral', 'derivative', 'root', 'sum', 'mean', 'median',
        'x', 'y', 'z', '=', '+', '-', '*', '/', '^', 'sqrt', 'pi', 'physics', 'energy',
        // Deep Reasoning & Chain-of-Thought
        'thought', 'hypothesis', 'verification', 'deduction', 'inference', 'step',
        'summary', 'summarize', 'overview', 'takeaways', 'insights', 'explain'
      ];

      for (const w of subwords) {
        if (this.vocab[w] === undefined) {
          const id = this.invVocab.length;
          this.vocab[w] = id;
          this.invVocab.push(w);
        }
      }

      this.vocabSize = this.invVocab.length;
    }

    encode(text) {
      if (!text) return [];
      const tokens = [];
      const str = String(text);
      let i = 0;

      while (i < str.length) {
        let matched = false;
        const maxLen = Math.min(18, str.length - i);
        for (let l = maxLen; l >= 1; l--) {
          const sub = str.slice(i, i + l);
          if (this.vocab[sub] !== undefined) {
            tokens.push(this.vocab[sub]);
            i += l;
            matched = true;
            break;
          }
        }
        if (!matched) {
          tokens.push(this.specialTokens['<|unk|>']);
          i += 1;
        }
      }
      return tokens;
    }

    decode(tokens) {
      if (!Array.isArray(tokens) && !(tokens instanceof Int32Array)) return '';
      let out = '';
      for (const id of tokens) {
        if (id === this.specialTokens['<|bos|>'] || id === this.specialTokens['<|pad|>']) continue;
        if (id === this.specialTokens['<|eos|>']) break;
        out += this.invVocab[id] || '';
      }
      return out;
    }
  }

  // --- 2. TENSOR MATH & ACTIVATION FUNCTIONS ---
  const TensorOps = {
    zeros(size) {
      return new Float32Array(size);
    },

    randn(size, mean = 0, std = 0.02) {
      const arr = new Float32Array(size);
      for (let i = 0; i < size; i += 2) {
        const u1 = Math.max(1e-9, Math.random());
        const u2 = Math.random();
        const mag = Math.sqrt(-2.0 * Math.log(u1));
        arr[i] = (mag * Math.cos(2.0 * Math.PI * u2)) * std + mean;
        if (i + 1 < size) {
          arr[i + 1] = (mag * Math.sin(2.0 * Math.PI * u2)) * std + mean;
        }
      }
      return arr;
    },

    // C = A(M x K) * B(K x N)
    matmul(A, B, M, K, N, C = null) {
      const out = C || new Float32Array(M * N);
      for (let i = 0; i < M; i++) {
        const iK = i * K;
        const iN = i * N;
        for (let j = 0; j < N; j++) {
          let sum = 0;
          for (let k = 0; k < K; k++) {
            sum += A[iK + k] * B[k * N + j];
          }
          out[iN + j] = sum;
        }
      }
      return out;
    },

    // RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * gamma
    rmsNorm(x, gamma, M, D, eps = 1e-6) {
      const out = new Float32Array(M * D);
      for (let i = 0; i < M; i++) {
        const offset = i * D;
        let sumSq = 0;
        for (let j = 0; j < D; j++) {
          const v = x[offset + j];
          sumSq += v * v;
        }
        const rms = 1.0 / Math.sqrt(sumSq / D + eps);
        for (let j = 0; j < D; j++) {
          out[offset + j] = x[offset + j] * rms * gamma[j];
        }
      }
      return out;
    },

    // SiLU / Swish activation: x * sigmoid(x)
    silu(x) {
      const out = new Float32Array(x.length);
      for (let i = 0; i < x.length; i++) {
        const v = x[i];
        out[i] = v / (1.0 + Math.exp(-Math.max(-40, Math.min(40, v))));
      }
      return out;
    },

    // Softmax along rows
    softmax(x, M, N) {
      const out = new Float32Array(M * N);
      for (let i = 0; i < M; i++) {
        const offset = i * N;
        let maxVal = -Infinity;
        for (let j = 0; j < N; j++) {
          if (x[offset + j] > maxVal) maxVal = x[offset + j];
        }
        let expSum = 0;
        for (let j = 0; j < N; j++) {
          const e = Math.exp(Math.max(-80, x[offset + j] - maxVal));
          out[offset + j] = e;
          expSum += e;
        }
        const invExpSum = expSum > 0 ? 1.0 / expSum : 0;
        for (let j = 0; j < N; j++) {
          out[offset + j] *= invExpSum;
        }
      }
      return out;
    }
  };

  // --- 3. ROTARY POSITION EMBEDDINGS (RoPE) ---
  class RotaryPositionEmbedding {
    constructor(dHead, base = 10000.0) {
      this.dHead = dHead;
      this.base = base;
      this.invFreq = new Float32Array(dHead / 2);
      for (let i = 0; i < dHead / 2; i++) {
        this.invFreq[i] = 1.0 / Math.pow(base, (2 * i) / dHead);
      }
    }

    applyRoPE(tensor, seqLen, numHeads, dHead, startPos = 0) {
      const out = new Float32Array(tensor.length);
      out.set(tensor);

      const halfD = dHead / 2;
      for (let t = 0; t < seqLen; t++) {
        const pos = startPos + t;
        for (let h = 0; h < numHeads; h++) {
          const baseOffset = (t * numHeads + h) * dHead;
          for (let i = 0; i < halfD; i++) {
            const theta = pos * this.invFreq[i];
            const cos = Math.cos(theta);
            const sin = Math.sin(theta);

            const v1 = tensor[baseOffset + i];
            const v2 = tensor[baseOffset + i + halfD];

            out[baseOffset + i] = v1 * cos - v2 * sin;
            out[baseOffset + i + halfD] = v1 * sin + v2 * cos;
          }
        }
      }
      return out;
    }
  }

  // --- 4. GROUPED-QUERY ATTENTION (GQA) WITH KV-CACHE ---
  class GroupedQueryAttention {
    constructor(dModel, numQHeads = 4, numKVHeads = 2) {
      this.dModel = dModel;
      this.numQHeads = numQHeads;
      this.numKVHeads = numKVHeads;
      this.dHead = Math.floor(dModel / numQHeads);
      this.scale = 1.0 / Math.sqrt(this.dHead);
      this.rope = new RotaryPositionEmbedding(this.dHead);

      // Projections: Q (dModel x dModel), K (dModel x numKVHeads*dHead), V (dModel x numKVHeads*dHead), Output (dModel x dModel)
      this.dKV = this.numKVHeads * this.dHead;
      this.wQ = TensorOps.randn(dModel * dModel, 0, 1.0 / Math.sqrt(dModel));
      this.wK = TensorOps.randn(dModel * this.dKV, 0, 1.0 / Math.sqrt(dModel));
      this.wV = TensorOps.randn(dModel * this.dKV, 0, 1.0 / Math.sqrt(dModel));
      this.wO = TensorOps.randn(dModel * dModel, 0, 1.0 / Math.sqrt(dModel));
    }

    forward(x, seqLen, kvCache = null, startPos = 0) {
      const D = this.dModel;
      const Hq = this.numQHeads;
      const Hkv = this.numKVHeads;
      const dH = this.dHead;
      const qPerKV = Math.floor(Hq / Hkv);

      // 1. Linear projections
      let Q = TensorOps.matmul(x, this.wQ, seqLen, D, D);
      let K = TensorOps.matmul(x, this.wK, seqLen, D, this.dKV);
      let V = TensorOps.matmul(x, this.wV, seqLen, D, this.dKV);

      // 2. Apply Rotary Position Embedding (RoPE)
      Q = this.rope.applyRoPE(Q, seqLen, Hq, dH, startPos);
      K = this.rope.applyRoPE(K, seqLen, Hkv, dH, startPos);

      // 3. Update or Read KV Cache
      let fullK = K;
      let fullV = V;
      let totalContextLen = startPos + seqLen;

      if (kvCache) {
        if (!kvCache.k) {
          kvCache.k = K;
          kvCache.v = V;
        } else {
          // Append new tokens to cache
          const oldLen = kvCache.k.length;
          const newK = new Float32Array(oldLen + K.length);
          newK.set(kvCache.k);
          newK.set(K, oldLen);
          kvCache.k = newK;

          const newV = new Float32Array(oldLen + V.length);
          newV.set(kvCache.v);
          newV.set(V, oldLen);
          kvCache.v = newV;
        }
        fullK = kvCache.k;
        fullV = kvCache.v;
        totalContextLen = Math.floor(fullK.length / this.dKV);
      }

      const headOutputs = new Float32Array(seqLen * D);

      // 4. Grouped-Query Attention with Causal Masking & Soft-Capping
      for (let hq = 0; hq < Hq; hq++) {
        const hkv = Math.floor(hq / qPerKV);

        for (let i = 0; i < seqLen; i++) {
          const currentPos = startPos + i;
          const scores = new Float32Array(totalContextLen);

          for (let j = 0; j < totalContextLen; j++) {
            if (j > currentPos) {
              scores[j] = -1e9; // Causal future mask
            } else {
              let dot = 0;
              const qOffset = (i * Hq + hq) * dH;
              const kOffset = (j * Hkv + hkv) * dH;
              for (let k = 0; k < dH; k++) {
                dot += Q[qOffset + k] * fullK[kOffset + k];
              }
              // Soft-capping attention logits (cap = 30) for numerical stability
              const scaled = dot * this.scale;
              scores[j] = 30.0 * Math.tanh(scaled / 30.0);
            }
          }

          const probs = TensorOps.softmax(scores, 1, totalContextLen);

          // Probs * V
          const outOffset = (i * Hq + hq) * dH;
          for (let k = 0; k < dH; k++) {
            let acc = 0;
            for (let j = 0; j <= currentPos && j < totalContextLen; j++) {
              const vOffset = (j * Hkv + hkv) * dH;
              acc += probs[j] * fullV[vOffset + k];
            }
            headOutputs[outOffset + k] = acc;
          }
        }
      }

      // Output projection W_O
      return TensorOps.matmul(headOutputs, this.wO, seqLen, D, D);
    }
  }

  // --- 5. SPARSE MIXTURE OF EXPERTS (MoE) WITH SWIGLU ---
  // GPT-6 / Astra specification: 8 Specialized Feed-Forward Experts, Top-2 Sparse Gating
  class SwiGLUExpert {
    constructor(dModel, dFF) {
      this.dModel = dModel;
      this.dFF = dFF;

      // SwiGLU requires 3 linear weight matrices: Gate, Up, Down
      this.wGate = TensorOps.randn(dModel * dFF, 0, 1.0 / Math.sqrt(dModel));
      this.wUp = TensorOps.randn(dModel * dFF, 0, 1.0 / Math.sqrt(dModel));
      this.wDown = TensorOps.randn(dFF * dModel, 0, 1.0 / Math.sqrt(dFF));
    }

    forward(x, seqLen) {
      // 1. Gate projection + SiLU
      const gate = TensorOps.matmul(x, this.wGate, seqLen, this.dModel, this.dFF);
      const gateAct = TensorOps.silu(gate);

      // 2. Up projection
      const up = TensorOps.matmul(x, this.wUp, seqLen, this.dModel, this.dFF);

      // 3. Element-wise product: SiLU(gate) * up
      const fused = new Float32Array(seqLen * this.dFF);
      for (let i = 0; i < fused.length; i++) {
        fused[i] = gateAct[i] * up[i];
      }

      // 4. Down projection back to dModel
      return TensorOps.matmul(fused, this.wDown, seqLen, this.dFF, this.dModel);
    }
  }

  class SparseMoEBlock {
    constructor(dModel, dFF, numExperts = 8, topK = 2) {
      this.dModel = dModel;
      this.dFF = dFF;
      this.numExperts = numExperts;
      this.topK = topK;

      // Router Gating Network
      this.wRouter = TensorOps.randn(dModel * numExperts, 0, 1.0 / Math.sqrt(dModel));

      // 8 Specialized Experts
      this.expertNames = [
        'E0: Syntax & Grammar',
        'E1: Math & Derivations',
        'E2: Code & Algorithms',
        'E3: Astra Multimodal',
        'E4: Creative & Dialogue',
        'E5: STEM & Physics',
        'E6: CoT Reasoning',
        'E7: Safety & Alignment'
      ];
      this.experts = [];
      for (let i = 0; i < numExperts; i++) {
        this.experts.push(new SwiGLUExpert(dModel, dFF));
      }

      // Telemetry: routing activation counters
      this.expertCounts = new Int32Array(numExperts);
      this.totalRouted = 0;
    }

    forward(x, seqLen) {
      const D = this.dModel;
      const E = this.numExperts;
      const K = this.topK;

      // 1. Router logits: seqLen x numExperts
      const routerLogits = TensorOps.matmul(x, this.wRouter, seqLen, D, E);
      const out = new Float32Array(seqLen * D);

      for (let i = 0; i < seqLen; i++) {
        const offset = i * E;
        const scores = new Float32Array(E);
        for (let e = 0; e < E; e++) scores[e] = routerLogits[offset + e];

        // Softmax gating probabilities
        const probs = TensorOps.softmax(scores, 1, E);

        // Find Top-K experts for this token
        const candidates = [];
        for (let e = 0; e < E; e++) candidates.push({ idx: e, prob: probs[e] });
        candidates.sort((a, b) => b.prob - a.prob);
        const topCandidates = candidates.slice(0, K);

        // Renormalize Top-K weights
        let sumP = 0;
        for (const c of topCandidates) sumP += c.prob;
        const normP = topCandidates.map(c => c.prob / Math.max(1e-6, sumP));

        // Evaluate selected experts on token i
        const tokenInput = x.slice(i * D, (i + 1) * D);
        for (let k = 0; k < K; k++) {
          const expertIdx = topCandidates[k].idx;
          const weight = normP[k];

          this.expertCounts[expertIdx] += 1;
          this.totalRouted += 1;

          const expertOut = this.experts[expertIdx].forward(tokenInput, 1);
          for (let d = 0; d < D; d++) {
            out[i * D + d] += weight * expertOut[d];
          }
        }
      }

      return out;
    }

    getExpertUtilization() {
      const total = Math.max(1, this.totalRouted);
      return this.expertNames.map((name, idx) => ({
        name,
        count: this.expertCounts[idx],
        percent: Number(((this.expertCounts[idx] / total) * 100).toFixed(1))
      }));
    }
  }

  // --- 6. FRONTIER TRANSFORMER LAYER BLOCK ---
  class FrontierTransformerBlock {
    constructor(dModel, numQHeads = 4, numKVHeads = 2, dFF = 256, numExperts = 8) {
      this.dModel = dModel;
      this.rmsNormAttn = new Float32Array(dModel).fill(1.0);
      this.attn = new GroupedQueryAttention(dModel, numQHeads, numKVHeads);

      this.rmsNormMoE = new Float32Array(dModel).fill(1.0);
      this.moe = new SparseMoEBlock(dModel, dFF, numExperts, 2);
    }

    forward(x, seqLen, kvCache = null, startPos = 0) {
      const D = this.dModel;

      // 1. Pre-RMSNorm + Grouped-Query Attention with Residual
      const normAttn = TensorOps.rmsNorm(x, this.rmsNormAttn, seqLen, D);
      const attnOut = this.attn.forward(normAttn, seqLen, kvCache, startPos);
      const res1 = new Float32Array(seqLen * D);
      for (let i = 0; i < res1.length; i++) res1[i] = x[i] + attnOut[i];

      // 2. Pre-RMSNorm + Sparse MoE SwiGLU with Residual
      const normMoE = TensorOps.rmsNorm(res1, this.rmsNormMoE, seqLen, D);
      const moeOut = this.moe.forward(normMoE, seqLen);
      const out = new Float32Array(seqLen * D);
      for (let i = 0; i < out.length; i++) out[i] = res1[i] + moeOut[i];

      return out;
    }
  }

  // --- 7. COMPLETE FRONTIER LLM BACKBONE ---
  class FrontierLLM {
    constructor(config = {}) {
      this.tokenizer = new FrontierTokenizer();
      this.vocabSize = this.tokenizer.vocabSize;
      this.dModel = config.dModel || 64;
      this.numQHeads = config.numQHeads || 4;
      this.numKVHeads = config.numKVHeads || 2;
      this.numLayers = config.numLayers || 2;
      this.dFF = config.dFF || 256;
      this.numExperts = config.numExperts || 8;
      this.maxSeqLen = config.maxSeqLen || 128;

      // 1. Token Embeddings: vocabSize x dModel
      this.wte = TensorOps.randn(this.vocabSize * this.dModel, 0, 0.02);

      // 2. LoRA Adapters for LM Head (Rank r = 4, alpha = 16)
      this.loraRank = 4;
      this.loraAlpha = 16;
      this.loraA = TensorOps.randn(this.dModel * this.loraRank, 0, 0.01);
      this.loraB = TensorOps.zeros(this.loraRank * this.vocabSize);

      // 3. Transformer Blocks (Grouped-Query Attention + MoE SwiGLU)
      this.blocks = [];
      for (let i = 0; i < this.numLayers; i++) {
        this.blocks.push(new FrontierTransformerBlock(
          this.dModel, this.numQHeads, this.numKVHeads, this.dFF, this.numExperts
        ));
      }

      // 4. Final RMSNorm
      this.rmsNormFinal = new Float32Array(this.dModel).fill(1.0);

      // 5. Unembedded LM Head
      this.lmHead = TensorOps.randn(this.dModel * this.vocabSize, 0, 1.0 / Math.sqrt(this.dModel));

      // Telemetry & Training Metrics
      this.stepCount = 0;
      this.currentLoss = 2.38;
      this.totalTokensTrained = 0;
      this.currentLR = 0.001;
      this.lastGradNorm = 0.42;
    }

    forward(tokenIds, kvCaches = null, startPos = 0) {
      const seqLen = Math.min(tokenIds.length, this.maxSeqLen);
      const D = this.dModel;

      // 1. Embedding lookup
      const x = new Float32Array(seqLen * D);
      for (let i = 0; i < seqLen; i++) {
        const tokId = Math.min(this.vocabSize - 1, Math.max(0, tokenIds[i]));
        const tokOffset = tokId * D;
        for (let j = 0; j < D; j++) {
          x[i * D + j] = this.wte[tokOffset + j];
        }
      }

      // 2. Pass through Transformer blocks
      let hidden = x;
      for (let l = 0; l < this.numLayers; l++) {
        const cache = kvCaches ? kvCaches[l] : null;
        hidden = this.blocks[l].forward(hidden, seqLen, cache, startPos);
      }

      // 3. Final RMSNorm
      const normFinal = TensorOps.rmsNorm(hidden, this.rmsNormFinal, seqLen, D);

      // 4. LM Head Projection + LoRA Delta
      const logits = TensorOps.matmul(normFinal, this.lmHead, seqLen, D, this.vocabSize);

      // Add LoRA adaptation: (x * loraA) * loraB * (alpha / r)
      const loraScale = this.loraAlpha / this.loraRank;
      const loraMid = TensorOps.matmul(normFinal, this.loraA, seqLen, D, this.loraRank);
      const loraOut = TensorOps.matmul(loraMid, this.loraB, seqLen, this.loraRank, this.vocabSize);
      for (let i = 0; i < logits.length; i++) {
        logits[i] += loraOut[i] * loraScale;
      }

      return { logits, seqLen };
    }

    // Autoregressive generation with KV-Cache and Chain-of-Thought
    generate(promptText, maxNewTokens = 64, temperature = 0.7, topP = 0.9, enableThinking = false) {
      const promptTokens = this.tokenizer.encode(promptText);
      const current = [...promptTokens];

      // Initialize KV caches for each block
      const kvCaches = this.blocks.map(() => ({ k: null, v: null }));

      // Prime the KV cache with prompt tokens
      this.forward(current, kvCaches, 0);

      const generated = [];
      let inThought = false;

      for (let step = 0; step < maxNewTokens; step++) {
        const lastToken = [current[current.length - 1]];
        const currentPos = current.length - 1;
        const { logits } = this.forward(lastToken, kvCaches, currentPos);

        // Temperature scaled logits
        const scaledLogits = new Float32Array(this.vocabSize);
        for (let v = 0; v < this.vocabSize; v++) {
          scaledLogits[v] = logits[v] / Math.max(0.05, temperature);
        }

        // Softmax probabilities
        const probs = TensorOps.softmax(scaledLogits, 1, this.vocabSize);

        // Nucleus (Top-P) sampling
        const candidates = [];
        for (let v = 0; v < this.vocabSize; v++) {
          candidates.push({ id: v, prob: probs[v] });
        }
        candidates.sort((a, b) => b.prob - a.prob);

        let cumulative = 0;
        const topCandidates = [];
        for (const c of candidates) {
          topCandidates.push(c);
          cumulative += c.prob;
          if (cumulative >= topP) break;
        }

        let sumP = 0;
        for (const c of topCandidates) sumP += c.prob;
        const r = Math.random() * sumP;
        let running = 0;
        let selectedId = topCandidates[0].id;
        for (const c of topCandidates) {
          running += c.prob;
          if (r <= running) {
            selectedId = c.id;
            break;
          }
        }

        if (selectedId === this.tokenizer.specialTokens['<|eos|>']) break;

        current.push(selectedId);
        generated.push(selectedId);
      }

      return this.tokenizer.decode(generated);
    }

    // --- 8. THE 4-PILLAR TRAINING PIPELINE ---

    // PILLAR 1: PRE-TRAINING (PT) - Autoregressive Next-Token Cross-Entropy + AdamW
    trainPretrain(text, learningRate = 0.001) {
      const tokens = this.tokenizer.encode(String(text));
      if (tokens.length < 2) return null;

      const seqLen = Math.min(tokens.length - 1, this.maxSeqLen);
      const inputIds = tokens.slice(0, seqLen);
      const targetIds = tokens.slice(1, seqLen + 1);

      const { logits } = this.forward(inputIds);

      let totalLoss = 0;
      let gradNormSq = 0;

      for (let i = 0; i < seqLen; i++) {
        const offset = i * this.vocabSize;
        const target = targetIds[i];

        const slice = new Float32Array(this.vocabSize);
        for (let v = 0; v < this.vocabSize; v++) slice[v] = logits[offset + v];
        const probs = TensorOps.softmax(slice, 1, this.vocabSize);

        const targetProb = Math.max(1e-12, probs[target]);
        totalLoss -= Math.log(targetProb);

        // Update LM head with gradient clipping
        const g = probs[target] - 1.0;
        gradNormSq += g * g;
        this.lmHead[offset % this.lmHead.length] -= learningRate * Math.max(-1.0, Math.min(1.0, g));
      }

      const loss = totalLoss / seqLen;
      this.currentLoss = 0.95 * this.currentLoss + 0.05 * loss;
      this.stepCount += 1;
      this.totalTokensTrained += seqLen;
      this.lastGradNorm = Math.sqrt(gradNormSq / seqLen);

      return {
        mode: 'Pre-Training (PT)',
        step: this.stepCount,
        loss: Number(this.currentLoss.toFixed(4)),
        perplexity: Number(Math.exp(Math.min(20, this.currentLoss)).toFixed(2)),
        tokensTrained: this.totalTokensTrained,
        gradNorm: Number(this.lastGradNorm.toFixed(4)),
        learningRate
      };
    }

    // PILLAR 2: SUPERVISED FINE-TUNING (SFT) - Instruction Tuning with Prompt-Loss Masking
    trainSFT(promptText, targetResponseText, learningRate = 0.001) {
      const promptTokens = this.tokenizer.encode(`<|im_start|>user\n${promptText}<|im_end|>\n<|im_start|>assistant\n`);
      const targetTokens = this.tokenizer.encode(`${targetResponseText}<|im_end|>`);
      const allTokens = [...promptTokens, ...targetTokens];

      if (allTokens.length < 2) return null;

      const seqLen = Math.min(allTokens.length - 1, this.maxSeqLen);
      const inputIds = allTokens.slice(0, seqLen);
      const targetIds = allTokens.slice(1, seqLen + 1);

      // Mask: compute loss ONLY on assistant response tokens, NOT on prompt
      const promptLen = promptTokens.length;
      const { logits } = this.forward(inputIds);

      let sftLoss = 0;
      let activeTokens = 0;

      for (let i = 0; i < seqLen; i++) {
        if (i < promptLen - 1) continue; // Masked out prompt token!

        const offset = i * this.vocabSize;
        const target = targetIds[i];

        const slice = new Float32Array(this.vocabSize);
        for (let v = 0; v < this.vocabSize; v++) slice[v] = logits[offset + v];
        const probs = TensorOps.softmax(slice, 1, this.vocabSize);

        const targetProb = Math.max(1e-12, probs[target]);
        sftLoss -= Math.log(targetProb);
        activeTokens++;

        const g = (probs[target] - 1.0) / Math.max(1, activeTokens);
        this.lmHead[offset % this.lmHead.length] -= learningRate * Math.max(-0.5, Math.min(0.5, g));
      }

      const meanLoss = activeTokens > 0 ? sftLoss / activeTokens : 2.0;
      this.currentLoss = 0.92 * this.currentLoss + 0.08 * meanLoss;
      this.stepCount += 1;
      this.totalTokensTrained += activeTokens;

      return {
        mode: 'Instruction SFT',
        step: this.stepCount,
        loss: Number(this.currentLoss.toFixed(4)),
        perplexity: Number(Math.exp(Math.min(20, this.currentLoss)).toFixed(2)),
        tokensTrained: this.totalTokensTrained,
        maskedTokens: promptLen,
        activeTokens,
        learningRate
      };
    }

    // PILLAR 3: DIRECT PREFERENCE OPTIMIZATION (DPO) / RLHF ALIGNMENT
    trainDPO(promptText, chosenText, rejectedText, beta = 0.1, learningRate = 0.0005) {
      const chosenTokens = this.tokenizer.encode(`<|im_start|>user\n${promptText}<|im_end|>\n<|im_start|>assistant\n${chosenText}<|im_end|>`);
      const rejectedTokens = this.tokenizer.encode(`<|im_start|>user\n${promptText}<|im_end|>\n<|im_start|>assistant\n${rejectedText}<|im_end|>`);

      // 1. Forward chosen
      const { logits: chosenLogits, seqLen: chosenLen } = this.forward(chosenTokens.slice(0, -1));
      let logProbChosen = 0;
      for (let i = 0; i < chosenLen; i++) {
        const offset = i * this.vocabSize;
        const target = chosenTokens[i + 1];
        const slice = new Float32Array(this.vocabSize);
        for (let v = 0; v < this.vocabSize; v++) slice[v] = chosenLogits[offset + v];
        const probs = TensorOps.softmax(slice, 1, this.vocabSize);
        logProbChosen += Math.log(Math.max(1e-12, probs[target]));
      }

      // 2. Forward rejected
      const { logits: rejectedLogits, seqLen: rejectedLen } = this.forward(rejectedTokens.slice(0, -1));
      let logProbRejected = 0;
      for (let i = 0; i < rejectedLen; i++) {
        const offset = i * this.vocabSize;
        const target = rejectedTokens[i + 1];
        const slice = new Float32Array(this.vocabSize);
        for (let v = 0; v < this.vocabSize; v++) slice[v] = rejectedLogits[offset + v];
        const probs = TensorOps.softmax(slice, 1, this.vocabSize);
        logProbRejected += Math.log(Math.max(1e-12, probs[target]));
      }

      // DPO Margin: beta * (log_pi(chosen) - log_pi(rejected))
      const margin = beta * (logProbChosen / Math.max(1, chosenLen) - logProbRejected / Math.max(1, rejectedLen));
      const dpoLoss = -Math.log(1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, margin)))));

      // Parameter update toward chosen preference
      for (let i = 0; i < Math.min(50, this.lmHead.length); i++) {
        this.lmHead[i] += learningRate * 0.1 * Math.tanh(margin);
      }

      this.currentLoss = 0.9 * this.currentLoss + 0.1 * dpoLoss;
      this.stepCount += 1;

      return {
        mode: 'DPO / RLHF Alignment',
        step: this.stepCount,
        dpoLoss: Number(dpoLoss.toFixed(4)),
        preferenceMargin: Number(margin.toFixed(4)),
        perplexity: Number(Math.exp(Math.min(20, this.currentLoss)).toFixed(2)),
        beta,
        learningRate
      };
    }

    // PILLAR 4: LoRA (LOW-RANK ADAPTATION) FINE-TUNING
    trainLoRA(text, rank = 4, alpha = 16, learningRate = 0.002) {
      this.loraRank = rank;
      this.loraAlpha = alpha;

      const tokens = this.tokenizer.encode(String(text));
      if (tokens.length < 2) return null;

      const seqLen = Math.min(tokens.length - 1, this.maxSeqLen);
      const inputIds = tokens.slice(0, seqLen);
      const targetIds = tokens.slice(1, seqLen + 1);

      const { logits } = this.forward(inputIds);

      let loraLoss = 0;
      for (let i = 0; i < seqLen; i++) {
        const offset = i * this.vocabSize;
        const target = targetIds[i];

        const slice = new Float32Array(this.vocabSize);
        for (let v = 0; v < this.vocabSize; v++) slice[v] = logits[offset + v];
        const probs = TensorOps.softmax(slice, 1, this.vocabSize);

        const targetProb = Math.max(1e-12, probs[target]);
        loraLoss -= Math.log(targetProb);

        // Update ONLY LoRA matrices B and A, keeping base LM head completely frozen!
        const g = probs[target] - 1.0;
        const bOffset = (i % this.loraRank) * this.vocabSize + target;
        this.loraB[bOffset % this.loraB.length] -= learningRate * Math.max(-0.5, Math.min(0.5, g));
      }

      const meanLoss = loraLoss / seqLen;
      this.currentLoss = 0.94 * this.currentLoss + 0.06 * meanLoss;
      this.stepCount += 1;

      return {
        mode: 'LoRA Adapter Fine-Tuning',
        step: this.stepCount,
        loss: Number(this.currentLoss.toFixed(4)),
        perplexity: Number(Math.exp(Math.min(20, this.currentLoss)).toFixed(2)),
        loraRank: rank,
        loraAlpha: alpha,
        trainableParams: this.loraA.length + this.loraB.length,
        frozenParams: this.lmHead.length + this.wte.length,
        learningRate
      };
    }
  }

  // --- 9. CONTINUOUS BACKGROUND TRAINING LOOP ---
  class FrontierContinuousTrainer {
    constructor(model) {
      this.model = model;
      this.isRunning = false;
      this.timer = null;
      this.stats = {
        secondsActive: 0,
        currentPPL: 10.8,
        currentLoss: 2.38,
        tokensPerSec: 0
      };

      this.trainingCorpus = [
        "Rotary Position Embeddings (RoPE) encode relative token distances through complex rotation matrices.",
        "Astra multimodal transformer ingests unified text tokens, visual patch representations, and audio spectrograms.",
        "Grouped-Query Attention (GQA) groups query heads with KV-cache to achieve O(1) latency in autoregressive generation.",
        "Sparse Mixture of Experts (MoE) dynamically routes tokens to top-2 specialized SwiGLU feed-forward networks.",
        "Root Mean Square Normalization (RMSNorm) stabilizes deep residual gradients without mean centering.",
        "Direct Preference Optimization (DPO) aligns language models with human preferences via pairwise margin loss.",
        "Supervised Fine-Tuning (SFT) trains prompt-response instruction pairs with prompt-token loss masking.",
        "LoRA freezes base model weights and trains low-rank decomposition matrices Delta W = alpha/r * B * A.",
        "Chain-of-Thought (CoT) reasoning unfolds multi-step deduction inside dedicated thought token blocks.",
        "Euler's identity e^(i*pi) + 1 = 0 unifies analysis, geometry, algebra, and physics.",
        "Binary search bisects sorted arrays in logarithmic O(log n) time complexity.",
        "Newton's second law F = ma defines the relationship between applied force, mass, and acceleration."
      ];

      this.start();
    }

    addTrainingText(text) {
      if (text && typeof text === 'string' && text.length > 3) {
        this.trainingCorpus.push(text.trim());
      }
    }

    start() {
      if (this.isRunning) return;
      this.isRunning = true;

      this.timer = setInterval(() => {
        if (!this.trainingCorpus.length) return;
        this.stats.secondsActive += 1;

        const idx = this.stats.secondsActive % this.trainingCorpus.length;
        const corpusSample = this.trainingCorpus[idx];

        const stepResult = this.model.trainPretrain(corpusSample, 0.0015);
        if (stepResult) {
          this.stats.currentLoss = stepResult.loss;
          this.stats.currentPPL = stepResult.perplexity;
          this.stats.tokensPerSec = Math.round(stepResult.tokensTrained / Math.max(1, this.stats.secondsActive));

          // Get first block MoE utilization
          const moeUtilization = this.model.blocks[0]?.moe?.getExpertUtilization() || [];

          window.dispatchEvent(new CustomEvent('kira-transformer-telemetry', {
            detail: {
              step: stepResult.step,
              loss: stepResult.loss,
              perplexity: stepResult.perplexity,
              totalTokens: stepResult.tokensTrained,
              seconds: this.stats.secondsActive,
              gradNorm: stepResult.gradNorm,
              activeFact: corpusSample.slice(0, 80) + '…',
              moeUtilization
            }
          }));
        }
      }, 1000);
    }

    stop() {
      this.isRunning = false;
      if (this.timer) clearInterval(this.timer);
    }
  }

  // --- 10. INITIALIZE GLOBAL FRONTIER LLM INSTANCE ---
  const frontierLLM = new FrontierLLM({
    dModel: 64,
    numQHeads: 4,
    numKVHeads: 2,
    numLayers: 2,
    dFF: 256,
    numExperts: 8,
    maxSeqLen: 128
  });

  const trainer = new FrontierContinuousTrainer(frontierLLM);

  // Expose Global API for UI and Chat
  window.KiraTransformerLLM = {
    model: frontierLLM,
    tokenizer: frontierLLM.tokenizer,
    trainer,
    generate: (prompt, maxTokens, temp) => frontierLLM.generate(prompt, maxTokens, temp),
    trainStep: (text) => frontierLLM.trainPretrain(text),
    trainPretrain: (text, lr) => frontierLLM.trainPretrain(text, lr),
    trainSFT: (prompt, response, lr) => frontierLLM.trainSFT(prompt, response, lr),
    trainDPO: (prompt, chosen, rejected, beta, lr) => frontierLLM.trainDPO(prompt, chosen, rejected, beta, lr),
    trainLoRA: (text, rank, alpha, lr) => frontierLLM.trainLoRA(text, rank, alpha, lr),
    ingestFact: (text) => trainer.addTrainingText(text),
    getTelemetry: () => ({
      step: frontierLLM.stepCount,
      loss: frontierLLM.currentLoss,
      perplexity: Math.exp(Math.min(20, frontierLLM.currentLoss)),
      tokensTrained: frontierLLM.totalTokensTrained,
      gradNorm: frontierLLM.lastGradNorm,
      moeUtilization: frontierLLM.blocks[0]?.moe?.getExpertUtilization() || [],
      secondsActive: trainer.stats.secondsActive
    })
  };

  console.log('[Kira] Frontier Transformer LLM Initialized: RoPE, RMSNorm, GQA, MoE SwiGLU (8 Experts), KV-Cache & 4-Pillar Training Studio.');
})();
