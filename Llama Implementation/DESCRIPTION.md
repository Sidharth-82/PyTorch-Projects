# 🦙 Llama 3.1-8B From Scratch: A Minimal PyTorch Implementation

## 📌 Overview

This project reimplements the **Llama 3.1-8B** language model **from scratch in PyTorch**, then loads Meta's official pretrained weights into the custom architecture to run real text generation.

The goal is educational: to understand every component of a modern decoder-only transformer by building it piece by piece — RMSNorm, Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA) with KV caching, and the SwiGLU feed-forward network — rather than relying on a high-level library.

Each building block is implemented and **shape-tested independently** before being assembled into the full model, which is then loaded with the official `consolidated.00.pth` checkpoint (≈8B parameters) and used to perform text completion.

---

## 🧠 Project Goals

* Build the Llama 3.1-8B architecture from first principles in PyTorch
* Implement and verify each transformer component in isolation
* Match Meta's official hyperparameters (Llama 3 paper, Table 3)
* Load real pretrained Llama 3.1-8B weights into the custom model
* Run end-to-end text completion with temperature + nucleus (top-p) sampling

---

## 🏗️ Model Architecture

```
Token IDs
    │
    ▼
Token Embeddings (VOCAB_SIZE × DIM)
    │
    ▼
┌─────────────────────────────┐
│  Transformer Block  × 32     │
│                              │
│   RMSNorm                    │
│      │                       │
│   Grouped-Query Attention    │
│   (RoPE + KV Cache)          │
│      │  (residual add)       │
│   RMSNorm                    │
│      │                       │
│   SwiGLU FeedForward         │
│      │  (residual add)       │
└─────────────────────────────┘
    │
    ▼
RMSNorm
    │
    ▼
Output Projection (DIM → VOCAB_SIZE)
    │
    ▼
Logits → Sampling → Next Token
```

---

## 🔩 Core Components

### 1️⃣ RMSNorm

Root Mean Square Layer Normalization replaces standard LayerNorm. It normalizes activations by their RMS value and applies a learnable per-dimension weight, with no mean subtraction or bias.

### 2️⃣ Rotary Position Embeddings (RoPE)

Positional information is injected by rotating query and key vectors in complex space:

* `precompute_freqs_cis` builds the rotation frequencies using `ROPE_THETA = 500000`
* `apply_rotary_emb` applies the rotations to queries and keys
* Relative positions are encoded directly into the attention dot product

### 3️⃣ Grouped-Query Attention (GQA)

Attention uses fewer key/value heads than query heads to save memory:

* 32 query heads, 8 key/value heads (`N_KV_HEAD_REP = 4`)
* KV heads are repeated to match query heads via `repeat_interleave`
* A **KV cache** stores past keys/values so each new token reuses prior computation
* A causal mask enforces autoregressive (left-to-right) attention
* Uses `F.scaled_dot_product_attention` for the core computation

### 4️⃣ SwiGLU FeedForward

The position-wise feed-forward network uses the SwiGLU activation:

```
FFN(x) = W2( SiLU(W1·x) ⊙ (W3·x) )
```

with hidden dimension `FFN_DIM = 14336`.

### 5️⃣ Transformer Block

Each block applies pre-normalization and residual connections:

```
h   = x + Attention(RMSNorm(x))
out = h + FeedForward(RMSNorm(h))
```

### 6️⃣ Tokenizer

A **Tiktoken-based BPE tokenizer** (Meta's Llama 3 tokenizer) with 256 reserved special tokens, including `<|begin_of_text|>`, `<|eot_id|>`, and chat header tokens. A `ChatFormat` helper builds instruction-tuned dialog prompts.

### 7️⃣ Generation

The `Llama` class handles checkpoint loading and autoregressive decoding:

* Loads the official `.pth` checkpoint into the custom `Transformer`
* `generate` performs token-by-token decoding with the KV cache
* Sampling via **temperature scaling** and **top-p (nucleus) sampling**
* Stops at EOS / end-of-turn tokens

---

## ⚙️ Model Configuration

| Parameter            | Value   | Source                          |
| -------------------- | ------- | ------------------------------- |
| Model Dimension      | 4096    | Llama 3 paper, Table 3          |
| FFN Dimension        | 14336   | Llama 3 paper, Table 3          |
| Layers               | 32      | Llama 3 paper, Table 3          |
| Attention Heads      | 32      | Llama 3 paper, Table 3          |
| KV Heads (GQA)       | 8       | Section 3.2 (GQA)               |
| Vocab Size           | 128256  | `params.json`                   |
| RoPE θ               | 500000  | Llama 3 paper, Table 3          |
| Norm ε               | 1e-5    | `params.json`                   |
| Max Batch Size       | 4       | Hardware-constrained            |
| Max Sequence Length  | 128     | Hardware-constrained            |

> Total parameters loaded: **~8.03 billion**

---

## 📂 Repository Structure

```
.
├── llama-implementation-from-scratch.ipynb
└── DESCRIPTION.md
```

---

## 📦 Requirements

The model weights and tokenizer are Meta's official Llama 3.1-8B release (e.g. via Kaggle's `llama3` dataset):

```
consolidated.00.pth     # model weights
params.json             # model config
tokenizer.model         # tiktoken BPE tokenizer
checklist.chk
```

---

## ▶️ Usage

### 1. Install Dependencies

```bash
pip install torch tiktoken blobfile
```

### 2. Run the Notebook

Open and execute:

```
llama-implementation-from-scratch.ipynb
```

The notebook will:

1. Define and shape-test each component (RMSNorm, RoPE, FFN, Attention, Block)
2. Assemble the full `Transformer`
3. Build the tokenizer
4. Load the pretrained Llama 3.1-8B checkpoint
5. Run text completion on sample prompts

### 3. Example

```python
prompts = [
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
]
results = generator.text_completion(
    prompts,
    max_gen_len=64,
    temperature=0.6,
    top_p=0.9,
)
```

---

## 🧩 Key Features

✅ Full Llama 3.1-8B architecture built from scratch in PyTorch
✅ RMSNorm, RoPE, Grouped-Query Attention, and SwiGLU implemented manually
✅ KV caching for efficient autoregressive generation
✅ Official Meta Tiktoken tokenizer with chat formatting
✅ Loads real pretrained 8B weights
✅ Temperature + top-p nucleus sampling

---

## 🚧 Current Limitations

* `MAX_SEQ_LEN` and `MAX_BATCH_SIZE` are small due to limited hardware
* Runs in float16 / bfloat16; checkpoint load is slow (~3 min on CPU mapping)
* Loaded with `strict=False`; outputs are raw base-model completions (not instruction-tuned)
* No fine-tuning or training loop — inference only

---

## 🔮 Future Improvements

* Longer context length and larger batch support
* Instruction-tuned chat inference using the `ChatFormat` pipeline
* Quantization for lower-memory inference
* Benchmarking against the reference Llama implementation
* Optional training / fine-tuning support

---

## 📚 References

* Llama 3 Herd of Models paper (architecture, Table 3, Section 3.2 GQA)
* Meta's official Llama 3 reference implementation
* RoFormer: Rotary Position Embedding (RoPE)
* GLU Variants Improve Transformer (SwiGLU)
