# 🤖 Transformers Fundamentals

> A from-scratch PyTorch implementation of the core Transformer building blocks, compared against a pretrained Hugging Face model, with controlled experiments on attention heads, positional encoding, and sequence length.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-ee4c2c)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-HF-yellow)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![Status](https://img.shields.io/badge/status-in%20progress-orange)

---

## 📑 Table of Contents

- [🎯 Goals](#-goals)
- [🗺️ Roadmap](#️-roadmap)
- [🗂️ Project Structure](#️-project-structure)
- [⚙️ Setup](#️-setup)
- [🚀 Usage](#-usage)
- [🧠 Part 1 — Self-Attention](#-part-1--self-attention)
- [🧩 Part 2 — Multi-Head Attention](#-part-2--multi-head-attention)
- [🏗️ Part 3 — Transformer Encoder Block](#️-part-3--transformer-encoder-block)
- [🤗 Part 4 — Hugging Face Model](#-part-4--hugging-face-model)
- [⚖️ Part 5 — Toy Model vs. Real Transformer](#️-part-5--toy-model-vs-real-transformer)
- [🧪 Part 6 — Experiments](#-part-6--experiments)
- [✅ Testing](#-testing)
- [📚 Key Concepts Cheat Sheet](#-key-concepts-cheat-sheet)
- [⚠️ Limitations](#️-limitations)
- [📖 References](#-references)

---

## 🎯 Goals

The goal is **understanding the Transformer architecture**, not just running a pretrained model.

- 🔧 Implement each component by hand, writing every `forward()` and tracking every tensor shape
- 🔢 Understand self-attention through a small numerical example
- 🤗 Run a pretrained Hugging Face classifier **without** `pipeline()` (using `AutoTokenizer` and `AutoModelForSequenceClassification`)
- 🔬 Run experiments that show *why* the design choices matter
- ⚖️ Honestly compare a toy implementation with a production-grade model

**Out of scope:** LangChain, LangGraph, agents, and building a ChatGPT-like system.

---

## 🗺️ Roadmap

| Phase | Topic | Status |
|:---:|---|:---:|
| 0 | Project skeleton and environment | ✅ Done |
| 1 | Attention math demo (numerical example) | 🚧 In progress |
| 2 | Embeddings, positional encoding, scaled dot-product attention | ⏳ Planned |
| 3 | Multi-head attention | ⏳ Planned |
| 4 | Encoder block and small model | ⏳ Planned |
| 5 | Training the from-scratch model | ⏳ Planned |
| 6 | Hugging Face inference and evaluation | ⏳ Planned |
| 7 | Three experiments | ⏳ Planned |
| 8 | Final comparison and write-up | ⏳ Planned |

---

## 🗂️ Project Structure

```text
transformers-project/
├── README.md
├── requirements.txt
├── pytest.ini
│
├── src/
│   ├── embeddings.py              # Token embedding (scaled by sqrt(d_model))
│   ├── positional_encoding.py     # Sinusoidal positional encoding
│   ├── attention.py               # Scaled dot-product attention (+ mask)
│   ├── multi_head_attention.py    # Multi-head attention
│   ├── feed_forward.py            # Position-wise feed-forward network
│   ├── transformer_block.py       # Encoder block (MHA + FFN + residual + LayerNorm)
│   └── model.py                   # Small Transformer classifier
│
├── experiments/
│   ├── heads.py                   # Experiment 1: number of heads
│   ├── positional_encoding.py     # Experiment 2: no positional encoding
│   └── sequence_length.py         # Experiment 3: cost vs. sequence length
│
├── reports/                       # Demo scripts, plots, result tables
├── train.py                       # Train the from-scratch model
├── evaluate.py                    # Evaluate own model and the HF model
│
└── tests/
    ├── test_attention.py
    ├── test_shapes.py
    └── test_transformer.py
```

---

## ⚙️ Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install PyTorch (CPU build) and the remaining dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Verify the installation
pytest
```

---

## 🚀 Usage

```bash
# Phase 1: step-by-step attention demo with numerical output
python reports/phase1_attention_demo.py

# Run all tests
pytest -v
```

> Commands for `train.py`, `evaluate.py`, and the experiments will be documented as those phases are completed.

---

## 🧠 Part 1 — Self-Attention

### The pipeline

```text
X  ──►  Q, K, V  ──►  QKᵀ  ──►  ÷ √dₖ  ──►  softmax  ──►  weights  ──►  weights · V
```

### The formula

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### 🔍 Why each piece exists

| Piece | Why it is there |
|---|---|
| **Q, K, V projections** | *Matching* (which tokens are related) and *content* (what information is passed on) are different jobs. Separate projections let the model learn asymmetric relations. |
| **QKᵀ** | Each query is compared with every key via a dot product, giving a `[T, T]` similarity matrix. The transpose makes shapes line up: `[T, dₖ] @ [dₖ, T] → [T, T]`. |
| **÷ √dₖ** | For random vectors with unit variance, the dot product has variance `dₖ`. Without scaling, scores grow with `dₖ`, softmax saturates (almost one-hot), and gradients vanish. |
| **Softmax (dim=-1)** | Turns each row of scores into positive weights that sum to 1, i.e. a distribution over the tokens each query attends to. |
| **· V** | Each output is a weighted average of all value vectors, so every token gathers context from the sequence. |

### 🔢 Numerical example

Simplification: no projections (`Q = K = V = X`) and `dₖ = 2`. Four tokens:

```text
t1 = [ 1,  0]    t2 = [ 0,  1]    t3 = [ 1,  1]    t4 = [ 1, -1]
```

**Step 1 — scores `QKᵀ`:**

```text
       t1  t2  t3  t4
t1  [   1,  0,  1,  1 ]
t2  [   0,  1,  1, -1 ]
t3  [   1,  1,  2,  0 ]
t4  [   1, -1,  0,  2 ]
```

**Step 2 — scaling by `√2 ≈ 1.414`.** Row t1 becomes `[0.707, 0, 0.707, 0.707]`.

**Step 3 — softmax on row t1:**

```text
weights(t1) = [0.286, 0.141, 0.286, 0.286]      # sums to 1
```

**Step 4 — weighted sum of V:**

```text
out(t1) = 0.286·[1,0] + 0.141·[0,1] + 0.286·[1,1] + 0.286·[1,-1]
        ≈ [0.859, 0.141]
```

💡 **Reading the result:** t1 attends least to t2 (orthogonal to it) and equally to the three tokens with a positive dot product.

### 🎚️ Why √dₖ matters (empirical)

`reports/phase1_attention_demo.py` (Part B) compares the average maximum softmax weight with and without scaling for `dₖ ∈ {4, 64, 512}`.

| dₖ | Score std | Avg max weight (unscaled) | Avg max weight (scaled) |
|:---:|:---:|:---:|:---:|
| 4 | _TBD_ | _TBD_ | _TBD_ |
| 64 | _TBD_ | _TBD_ | _TBD_ |
| 512 | _TBD_ | _TBD_ | _TBD_ |

> Fill in from the script output.

### 🔀 Self-attention ignores order

Permuting the input tokens only permutes the outputs (permutation equivariance, verified in Part C of the demo). Without extra information, the model cannot tell `"dog bites man"` from `"man bites dog"`. This is why **positional encoding** is required.

---

## 🧩 Part 2 — Multi-Head Attention

### Tensor shapes (B = batch, T = sequence length, D = model dim, H = heads, Dh = D / H)

| Stage | Shape |
|---|:---:|
| Input | `[B, T, D]` |
| After `W_q`, `W_k`, `W_v` | `[B, T, D]` |
| Split into heads | `[B, H, T, Dh]` |
| Scores / attention weights | `[B, H, T, T]` |
| Per-head output | `[B, H, T, Dh]` |
| Concatenate heads | `[B, T, D]` |
| After output projection `W_o` | `[B, T, D]` |

### ❓ Design questions

- **Why multiple heads?** Each head works in its own lower-dimensional subspace, so the model can attend to different kinds of relations in parallel (e.g. local syntax, long-range dependencies).
- **What does each head learn?** Not assigned by design; specialization emerges during training. Some heads end up redundant.
- **Why concatenate?** Each head produces a `Dh`-dimensional summary; concatenation restores the full `D`-dimensional representation.
- **Why a final projection `W_o`?** Concatenation only places head outputs side by side. `W_o` mixes information *across* heads.

---

## 🏗️ Part 3 — Transformer Encoder Block

```text
Input
  │
  ├──────────────┐
  ▼              │
Multi-Head Self-Attention
  │              │
  ▼              │
 (+) ◄───────────┘   Residual
  │
LayerNorm
  │
  ├──────────────┐
  ▼              │
Feed-Forward Network
  │              │
  ▼              │
 (+) ◄───────────┘   Residual
  │
LayerNorm
  │
Output  [B, T, D]
```

- 🔁 **Residual connections** provide a direct gradient path (`y = x + f(x)`), which makes deep stacks trainable and lets each layer learn a *correction* instead of a whole new representation.
- 📏 **LayerNorm** normalizes each token's features, stabilizing activation scale and training. It does not depend on batch statistics, so it works with variable-length sequences and small batches (unlike BatchNorm).
- 🧱 **Feed-forward network** is applied independently at each position (`Linear → activation → Linear`, usually with 4× expansion) and adds non-linear capacity.
- 📝 **Design choice:** _Post-LN or Pre-LN — to be decided in Phase 4 and justified here._

---

## 🤗 Part 4 — Hugging Face Model

- **Model:** _TBD (e.g. a small sentiment classifier)_
- **Dataset / split:** _TBD_
- **Approach:** `AutoTokenizer` + `AutoModelForSequenceClassification`, no `pipeline()`

Workflow:

1. ✂️ **Tokenize** raw text
2. 🔢 **Encode** (`input_ids`, `attention_mask`, padding, truncation)
3. 🧮 **Inference** (`model.eval()`, `torch.no_grad()`)
4. 🎯 **Prediction** (logits → softmax → label)
5. 📊 **Evaluation** (accuracy, F1)

### Results

| Model | Accuracy | F1 | Parameters |
|---|:---:|:---:|:---:|
| Own from-scratch model | _TBD_ | _TBD_ | _TBD_ |
| Hugging Face model | _TBD_ | _TBD_ | _TBD_ |

---

## ⚖️ Part 5 — Toy Model vs. Real Transformer

| Aspect | This project | Production-grade model |
|---|---|---|
| Scale | Small `D`, few layers | Hundreds of millions to billions of parameters |
| Pretraining | ❌ None | ✅ Large-scale self-supervised pretraining |
| Tokenization | Simple | Subword (BPE / WordPiece) |
| Positional info | Sinusoidal | Learned, rotary (RoPE), or relative |
| Attention | Plain implementation | Fused / memory-efficient kernels (e.g. FlashAttention) |
| Precision | float32 | Mixed precision (fp16 / bf16) |
| Training | Basic loop | Warmup, LR schedules, gradient clipping, distributed training |
| Architecture | Encoder only | Encoder, decoder, or encoder-decoder variants |
| Serving | None | Batching, caching, quantization, monitoring |

_To be expanded with concrete observations from the comparison._

---

## 🧪 Part 6 — Experiments

All experiments use fixed seeds and multiple runs per configuration.

### 🔬 Experiment 1 — Number of attention heads

**Setup:** `H ∈ {1, 2, 4, 8}` with `D` fixed (so `Dh = D / H` shrinks).

| Heads | Final loss | Accuracy | Time / epoch |
|:---:|:---:|:---:|:---:|
| 1 | _TBD_ | _TBD_ | _TBD_ |
| 2 | _TBD_ | _TBD_ | _TBD_ |
| 4 | _TBD_ | _TBD_ | _TBD_ |
| 8 | _TBD_ | _TBD_ | _TBD_ |

**Analysis:** _TBD_

### 🔬 Experiment 2 — Removing positional encoding

**Setup:** same model with and without positional encoding, on (a) the real task and (b) a synthetic order-sensitive task.

| Variant | Real task acc. | Order-sensitive task acc. |
|---|:---:|:---:|
| With positional encoding | _TBD_ | _TBD_ |
| Without positional encoding | _TBD_ | _TBD_ |

**Analysis:** _TBD_ (expected: self-attention is permutation equivariant, so order-dependent tasks should suffer most)

### 🔬 Experiment 3 — Sequence length and attention cost

**Setup:** measure forward/backward time and memory for increasing `T`.

| T | Time (ms) | Memory (MB) |
|:---:|:---:|:---:|
| 64 | _TBD_ | _TBD_ |
| 128 | _TBD_ | _TBD_ |
| 256 | _TBD_ | _TBD_ |
| 512 | _TBD_ | _TBD_ |

**Why O(T²)?** Every token attends to every other token, so each head builds a `[T, T]` score matrix. Both compute and memory therefore scale quadratically with `T`. Doubling the sequence length roughly quadruples the attention cost.

**Analysis:** _TBD_

---

## ✅ Testing

```bash
pytest -v
```

Planned checks:

- 📐 Attention weights sum to 1 along the key axis
- 🎭 Masked positions receive zero weight
- 🧪 Output matches `torch.nn.functional.scaled_dot_product_attention` (used **only** as a test reference)
- 📏 Shapes are correct across several `(B, T, D, H)` combinations
- 🔙 Gradients reach every parameter
- 🏋️ The model can overfit a tiny batch (sanity check)

---

## 📚 Key Concepts Cheat Sheet

| Question | Short answer |
|---|---|
| Why Q / K / V? | Separate roles: what to look for, what to match on, what to pass on |
| Why `QKᵀ`? | Pairwise dot-product similarity between all queries and keys |
| Why the transpose? | Makes the matrix product `[T,dₖ]·[dₖ,T]` valid |
| Why `√dₖ`? | Keeps score variance ≈ 1 so softmax does not saturate |
| What does softmax do? | Converts scores into a probability distribution per query |
| Why multiple heads? | Parallel attention in different subspaces |
| Why positional information? | Attention alone is order-blind |
| Self- vs. cross-attention? | Cross-attention takes Q from one sequence and K/V from another |
| Why residuals? | Gradient flow and easier optimization of deep stacks |
| Why LayerNorm? | Stable activations, independent of batch statistics |
| Encoder vs. decoder? | Decoder adds causal masking (and cross-attention in seq2seq) |
| BERT vs. GPT? | BERT = encoder stack, GPT = decoder stack |

---

## ⚠️ Limitations

- Small-scale models and datasets; results illustrate behavior and are not benchmarks
- No pretraining for the from-scratch model
- Encoder only (no decoder, no cross-attention implementation)
- CPU-only training limits model size and experiment breadth

---

## 📖 References

- Vaswani et al., *Attention Is All You Need* (2017)
- Ba, Kiros, Hinton, *Layer Normalization* (2016)
- He et al., *Deep Residual Learning for Image Recognition* (2015)
- Hugging Face Transformers documentation
