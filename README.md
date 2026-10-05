# Kira Frontier LLM (OpenAI GPT-6 & Google Astra Architecture)

Kira is an intelligent frontier chatspace and neural engine engineered to the architectural specifications of **OpenAI GPT-6** and **Google Project Astra**.

---

## 🌟 Key Architectural Features

1. **Multimodal Astra Perception Core & Tokenizer**:
   - Byte-Pair Encoding (BPE / TikToken byte-level vocabulary) with ChatML special tokens (`<|im_start|>`, `<|im_end|>`, `<|thought|>`, `<|vision_start|>`, `<|audio_start|>`).
   - Unified embedding stream fusing text tokens, visual patch representations, and audio spectrogram features.
   - Built-in Voice dictation (Speech-to-Text) and natural speech synthesis (Text-to-Speech).

2. **Rotary Position Embeddings (RoPE)**:
   - 2D orthogonal complex rotations on Query and Key projections ($\theta = 10000$), preserving relative token distances across long contexts.

3. **Pre-RMSNorm (Root Mean Square Normalization)**:
   - Replaces classical mean-centering LayerNorm with $\mathcal{O}(d)$ high-efficiency RMSNorm:
     $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum x_i^2 + \epsilon}} \odot \gamma$$

4. **Grouped-Query Attention (GQA) & KV-Cache**:
   - Groups $H_{kv}$ key-value heads with $H_q$ query heads to slash memory bandwidth.
   - Constant $\mathcal{O}(1)$ Key-Value (KV) cache for fast autoregressive token generation.
   - Soft-capped attention logits ($30.0 \times \tanh(S / 30.0)$) to prevent attention entropy collapse.

5. **Sparse Mixture of Experts (MoE) with SwiGLU**:
   - Router gating network routing each token dynamically to the **Top-2** most specialized experts out of 8 domain-specific SwiGLU networks (Syntax, Math, Code, Astra Multimodal, Creative Dialogue, STEM Physics, CoT Reasoning, and Safety Alignment).
   - Real-time expert utilization tracking and live telemetry gauges.

6. **Deep Chain-of-Thought (CoT) Reasoning Engine**:
   - Autonomous reflection and reasoning scratchpad displaying `💭 Thought for X.Xs` before emitting synthesized answers.

7. **4-Pillar Interactive Training Studio**:
   - **Pillar 1: Pre-Training (PT)**: Unsupervised next-token cross-entropy prediction with AdamW optimizer and gradient clipping ($||\mathbf{g}|| \le 1.0$).
   - **Pillar 2: Supervised Fine-Tuning (SFT)**: Instruction tuning with **Prompt-Loss Masking** (loss computed only on assistant target tokens, prompt masked out).
   - **Pillar 3: Direct Preference Optimization (DPO)**: Pairwise alignment with Chosen vs Rejected candidate completions and implicit reward margin loss.
   - **Pillar 4: LoRA (Low-Rank Adaptation)**: Freezes base weights and updates low-rank decomposition matrices ($\Delta W = \frac{\alpha}{r} B \cdot A$) with rank $r=4$.

---

## 🚀 How to Push Changes to GitHub Without Errors

If you encountered errors when pushing to GitHub, follow these exact steps. The repository includes a configured `.gitignore` that prevents large dependencies (`node_modules`) and secrets from blocking your push.

### Step 1: Initialize Git and Check Status
```bash
git init -b main
git status
```

### Step 2: Stage and Commit the Files
```bash
git add .
git commit -m "feat: Kira Frontier LLM (GPT-6 Astra architecture)"
```

### Step 3: Link to Your GitHub Repository
Create a new repository on [GitHub](https://github.com/new) (e.g. `kira-frontier-llm`). Then run:
```bash
# If using HTTPS:
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git

# OR if using SSH:
git remote add origin git@github.com:<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
```

### Step 4: Push to GitHub
```bash
git branch -M main
git push -u origin main
```

---

## 🛠️ Troubleshooting Common GitHub Push Errors

### 1. Error: `fatal: not a git repository`
- **Cause**: Git was not initialized in the project directory.
- **Fix**: Run `git init -b main`, then follow Steps 2–4 above.

### 2. Error: `remote: error: GH001: Large files detected` (or `exceeds file limit of 100.00 MB`)
- **Cause**: The `node_modules/` folder was accidentally staged.
- **Fix**: The included `.gitignore` already ignores `node_modules/`. If it was previously tracked, untrack it:
  ```bash
  git rm -r --cached node_modules
  git commit -m "chore: remove node_modules from git tracking"
  git push origin main
  ```

### 3. Error: `Updates were rejected because the remote contains work that you do not have locally`
- **Cause**: The GitHub repository was initialized with a README or license on GitHub.com.
- **Fix**: Rebase remote changes or force push to an empty repository:
  ```bash
  git pull origin main --rebase
  git push -u origin main
  # OR (for a brand new repository that you want to overwrite):
  git push -u origin main --force
  ```

### 4. Error: `Support for password authentication was removed`
- **Cause**: GitHub requires a Personal Access Token (PAT) or SSH key instead of your account password.
- **Fix**:
  1. Go to GitHub → **Settings** → **Developer Settings** → **Personal Access Tokens** → **Tokens (classic)**.
  2. Generate a token with the `repo` scope selected.
  3. When Git prompts for your password in the terminal, paste your Personal Access Token.
  4. Alternatively, use the GitHub CLI: `gh auth login`.

---

## 💻 Local Development

```bash
# 1. Install dependencies
npm install

# 2. Start development server
npm run dev

# 3. Open in browser
# Visit http://localhost:3000
```
