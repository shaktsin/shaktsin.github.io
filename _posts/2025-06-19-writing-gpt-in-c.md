---
layout: post
title: "Deconstruction Series #1: Rebuilding GPT-2 in Pure C"
date: 2025-06-19
description: "We tear down GPT-2 and rebuild it in C — no Python, just bare metal, pain, and performance."
---

Welcome to the **GPT-2 Deconstruction Series**, where we unravel OpenAI’s GPT-2 architecture from scratch — **no Python, no PyTorch, no dependencies**. Just raw logic, low-level memory management, and the elegance of C.
<!--more-->


In **Part 1**, we’ll lay the groundwork for running GPT-2 inference entirely in **pure C**. This includes:

- Understanding the GPT-2 architecture  
- Extracting and loading model weights  
- Building the Transformer block in C  
- Tokenizing input and generating predictions manually

> 🚨 This is a learning-oriented reimplementation. It's not meant to compete with optimized libraries like `ggml`, `transformers`, or `llama.cpp`. The goal is **understanding**.

---

## Architecture Summary: What is GPT-2?

GPT-2 is a stack of clever math layers that turns a sequence of words into predictions — one token at a time. Let’s walk through the key components in simple terms.


![GPT-2 Architecture](/assets/gpt-arch.png)

---

## Token + Position Embeddings

> **"Give meaning to numbers."**

Before GPT-2 can process words, it needs to turn them into numbers. Each word or subword (called a **token**) is mapped to a high-dimensional vector using a **token embedding**.

- The embedding table has size:  
  **(vocab_size, hidden_size)**  
  → For GPT-2 small: **(50257, 768)**  
  That means each token becomes a 768-length float vector.

- The model also needs to know **where** each word appears in the sentence. So we add **position embeddings** — like saying “this is the 1st word, this is the 2nd,” etc.

> Final input = token_embedding + position_embedding

---

## N Transformer Blocks

> **"Stacked layers of thinking."**

GPT-2 uses **12 blocks** (for GPT-2 Small). Each block contains two main sub-layers:

1. **Self-Attention**
2. **MLP (Feedforward Network)**

These layers are repeated again and again to refine the understanding of context.


## Self-Attention (`q`, `k`, `v`, `c_proj`)

> **"Look at other words before making a decision."**

This is where GPT-2 **pays attention** to previous tokens to figure out what’s important. It breaks the input into three parts:

- `q` = queries  
- `k` = keys  
- `v` = values  

Each is created by multiplying the input with a set of weights:

- Shapes:  
  - `q_weight`, `k_weight`, `v_weight`: **(hidden_size, hidden_size)**

### 🔣 Full Attention Formula (Mathematical Form)

Given input matrix \( X \in \mathbb{R}^{T \times d_{model}} \), where \( T \) is the sequence length and \( d_{model} \) is the hidden size:

1. **Linear Projections:**

\[
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
\]

Where:

- \( W^Q, W^K, W^V \in \mathbb{R}^{d_{model} \times d_{head}} \)
- Typically, \( d_{head} = \frac{d_{model}}{n_{heads}} \)

---

2. **Scaled Dot-Product Attention:**

\[
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_{head}}} + M \right)V
\]

Where:

- \( M \) is the **causal mask** that prevents attending to future tokens (values are \(-\infty\) above the diagonal)

---

3. **Concatenate Heads:**

\[
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
\]

Where:

- \( W^O \in \mathbb{R}^{d_{model} \times d_{model}} \)
- Each head is processed in parallel and then concatenated

---

4. **Final Projection:**

\[
\text{Output} = \text{MultiHead}(X)
\]

The final attention output is projected back using:

\[
\text{Output} \in \mathbb{R}^{T \times d_{model}} \quad \text{via} \quad c_{proj} \in \mathbb{R}^{d_{model} \times d_{model}}
\]

> These scores tell GPT-2 how much attention each word should pay to every previous word.
>
> The final attention output is projected back using `c_proj`.
>
> **Shape**: \( (d_{model}, d_{model}) \)
