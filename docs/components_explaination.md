# Where to tap the model — a guided tour for non-experts

This document explains the model we're training sparse autoencoders (SAEs) on,
how data flows through it, and what each "component" name means in the configs
(`resid_pre`, `resid_post`, `mamba_out`, `attn_out_prelinear`, `moe_out`,
`expert.<e>`, `shared_expert.<s>`, `attn_head.<h>`). No ML background assumed —
we build up from first principles.

---

## 1. What does a language model actually do?

A language model is a function that takes a sequence of tokens (sub-word chunks
of text) and produces, for each position, a probability distribution over which
token comes next.

```
"The cat sat on the"   →   P("mat") = 0.31
                            P("rug") = 0.18
                            P("chair") = 0.07
                            ...
```

The work happens inside a stack of **layers** (or "blocks"). Nemotron 3 Nano
30B-A3B has **52 layers**. Information passes through them one after another;
by the time it reaches the top, the model has refined a rich representation of
the input that it uses to predict the next token.

You can think of each layer as a small specialist that reads what the previous
layers have figured out, adds its own contribution, and passes the updated
notes forward.

---

## 2. The residual stream — the model's highway

This is the most important mental model in modern interpretability.

Every layer reads from and writes to a shared bus called the **residual
stream**. Imagine a long horizontal highway running through the whole model,
left-to-right (input → output), with one "lane" per token position. As text
flows through, each layer:

1. **Reads** the current state from the highway.
2. **Computes** something based on that state.
3. **Adds** the result back onto the highway.

```
        ┌─ block 0 ─┐    ┌─ block 1 ─┐    ┌─ block 2 ─┐
input → │           │ →  │           │ →  │           │ → ...
        │   reads   │    │   reads   │    │           │
        │   adds    │    │   adds    │    │           │
        └───────────┘    └───────────┘    └───────────┘
        ↑           ↑    ↑           ↑
   resid_pre   resid_post resid_pre  resid_post
   for layer 0 of layer 0 of layer 1 of layer 1
```

Two important facts:

- **Crucially: blocks add to the highway, they don't replace it.** This is
  what "residual" means — the connection from input to output is *still there*
  even after the layer's contribution is added. Old features survive; new ones
  accumulate on top.
- The highway has a fixed width of `d_model = 2688` real numbers per token at
  every position. After 52 layers of additions, every one of those 2688
  numbers per token is a soup of contributions from dozens of upstream
  computations. Untangling that soup is the whole point of training SAEs.

So our two simplest hook points are:

| Name           | Means                                              |
| -------------- | -------------------------------------------------- |
| `resid_pre[L]` | what the highway carries *just before* layer L acts |
| `resid_post[L]`| what the highway carries *just after* layer L acts  |

`resid_post` of layer L equals `resid_pre` of layer L+1 (they're the same point
on the highway, just labeled from different perspectives).

---

## 3. What does each layer do?

In a *vanilla* Transformer, every layer has two sub-pieces in this order:

```
[ Attention ]   →   [ Feed-Forward Network (FFN) ]
```

Attention lets each token "look at" other tokens to gather context. The FFN is
a per-token nonlinear transformation that processes that gathered context.

**Nemotron 3 Nano is not a vanilla Transformer.** It interleaves three kinds of
specialist blocks instead of two:

| Block kind | What it does | How many in the 52 |
| ---------- | ------------ | ------------------ |
| **Mamba-2 mixer** | Long-range token mixing using a *state-space model* (SSM). Cheaper than attention; constant-size state regardless of context length. | 23 |
| **GQA Attention** | Standard attention with grouped-query reduction (32 query heads, only 2 key/value heads — saves memory). | 6 |
| **MoE FFN** | A "Mixture of Experts" feed-forward layer with 128 small expert FFNs + 2 always-on shared experts. Each token uses only 6 of 128 routed experts. | 23 |

The order is hand-picked (see Figure 2 of the Nemotron 3 Nano paper):

```
×5 of:  [ Mamba | MoE | Mamba | MoE | Mamba | Attn | MoE ]
×3 of:  [ Mamba | MoE ]
×1 of:  [ Mamba | Attn | MoE ]
×4 of:  [ Mamba | MoE ]
```

Why this design? Mamba-2 is much cheaper than attention at long context, and
MoE makes FFN cheap *per token* by activating only a small slice of the
parameters. The result: 31.6 B total parameters but only 3.2 B active per
forward pass, with attention sprinkled in just enough to keep modeling quality
high.

---

## 4. Anatomy of each block kind

### 4.1 Mamba-2 block

Mamba-2 is a state-space model (SSM). For our purposes, you can think of it as
a learned IIR filter — a recurrent operation that maintains a 128-dimensional
hidden state per token and updates it as text flows past. It mixes tokens
*along the sequence axis* (so each output position is influenced by earlier
positions), much like attention does, but using an algorithm that's
asymptotically cheaper at long context.

```
   resid_pre ──► RMSNorm ──► Mamba-2 ──► (mamba_out)
                                              │
                                              ▼
                                        + back into
                                          resid_post
```

The hookable point we expose is **`mamba_out`** — the Mamba-2 block's
contribution *before* it gets added back into the residual stream. Think of it
as "what the Mamba-2 layer wants to write."

### 4.2 GQA Attention block

Attention works by, for each token, weighing how strongly it attends to every
other token in the sequence and pulling a weighted sum of their values. GQA
("grouped-query attention") is a memory-saving variant: 32 query heads share
just 2 key/value heads.

```
   resid_pre ──► RMSNorm ──► Q, K, V ──► softmax(QKᵀ)·V
                                              │
                                              ▼
                                       concat heads
                                              │
                                              ▼ (attn_out_prelinear, dim 32×128 = 4096)
                                          W_O linear
                                              │
                                              ▼
                                        + back into
                                          resid_post
```

Two hookable points here:

- **`attn_out_prelinear`** — the concatenation of all 32 head outputs *before*
  multiplying by the output projection W_O. This is the standard "attention
  output" SAE site used by Gemma Scope 2. Shape per token: 32 × 128 = 4096
  (notice: bigger than `d_model` = 2688).
- **`attn_head.<h>`** — just *one* head's output (the slice for head `h`,
  dimension 128). Use this if you want to study a single, narrowly-specialized
  head (e.g. the famous "induction heads" in mechanistic interpretability).

### 4.3 MoE FFN block (the most exotic one)

A normal FFN is one big two-layer network applied per token. **A Mixture of
Experts replaces it with many small networks (the "experts") and a learned
"router" that decides, per token, which experts to use.**

For Nemotron 3 Nano the MoE block has:

- 128 **routed experts** (small two-layer FFNs with intermediate dim 1856).
- 2 **shared experts** (small two-layer FFNs that fire for *every* token).
- A router that picks the **top 6 of 128** routed experts per token, weighting
  their outputs by router-assigned scores.

```
                       ┌──── routed expert 0
                       │              │ × score_0
                       ├──── routed expert 1
                       │              │ × score_1
   resid_pre ──► Router │     ...     │
                       │              │ × score_5
                       ├──── routed expert 127
                       │           (only top-6 selected per token)
                       └──── shared expert 0 ── always fires
                       └──── shared expert 1 ── always fires
                              ▲
                              │
                            sum
                              │
                              ▼ (moe_out)
                              │
                              ▼
                       + back into
                         resid_post
```

Hook points here:

- **`moe_out`** — the combined output (post-routing, post-weighting). The
  layer's full FFN-equivalent contribution. Treat it like a normal FFN output.
- **`expert.<e>`** — the output of *one specific* routed expert (e.g.
  `expert.42`). Crucially: **this only "fires" for the small fraction of
  tokens routed to that expert**, ≈ 6/128 ≈ 4.7 % of tokens on average.
  When training an SAE here, the cache stores activations *only for those
  tokens*, so the cache for one expert is tiny.
- **`shared_expert.<s>`** — output of one of the always-on shared experts
  (`s ∈ {0, 1}`). These fire for every token, so caches are full-size.

---

## 5. Putting it all together — the journey of one token

Imagine the token "cat" entering the model at position 5 of an input sequence.

```
1. Embedding lookup turns "cat" into a 2688-dim vector. This goes onto the
   highway at position 5.

2. Layer 0 (a Mamba-2 block): reads the highway, does state-space mixing
   along positions, writes its delta back. Highway updated.

3. Layer 1 (a MoE block): for each token, the router picks 6 out of 128
   experts. Each expert runs its little FFN. Their outputs are weighted-summed
   with the 2 shared experts' outputs and added back to the highway.

4. ... and so on for 52 layers, with attention sprinkled in 6 places ...

5. After layer 51, the highway state at position 5 is read out, projected to
   vocab-size logits, softmaxed, and we've predicted what comes after "cat".
```

When we train an SAE on `resid_post` at layer 25, we are saying: "for every
token, capture the highway state right after layer 25 finishes; learn a sparse
basis for those vectors so each token's state can be approximated as a sum of
just ~30 directions out of, say, 16 384 possible 'feature directions'."

When we train an SAE on `expert.42` at layer 12, we instead say: "of all the
tokens routed to expert 42 in layer 12, capture the expert's output for those
tokens; learn what set of features expert 42 has specialized in."

---

## 6. The component cheat sheet

| Component | One-line | When to pick it | Tensor shape |
| --- | --- | --- | --- |
| `resid_pre` | highway state *entering* layer L | layer-by-layer "what does the model know so far?" | (B, T, 2688) |
| `resid_post` | highway state *leaving* layer L | same as above; equivalent to `resid_pre` of L+1. **Best default for a first SAE.** | (B, T, 2688) |
| `mamba_out` | Mamba-2 block's contribution at layer L (only Mamba layers) | studying long-range mixing / SSM features | (B, T, 2688) |
| `attn_out_prelinear` | all 32 heads' outputs concatenated, *pre*-W_O projection (only Attn layers) | studying attention contributions; Gemma Scope 2 standard | (B, T, 4096) |
| `attn_head.<h>` | head `h`'s output alone, pre-W_O slice | studying one specific head's role | (B, T, 128) |
| `moe_out` | combined MoE FFN output (only MoE layers) | studying FFN-style features without caring which expert | (B, T, 2688) |
| `expert.<e>` | one routed expert's output (sparse: ~5 % of tokens) | studying expert specialization. Frontier work for hybrid Mamba+MoE LMs. | (~0.05·B·T, 2688) |
| `shared_expert.<s>` | one shared expert's output (every token) | studying always-on FFN-like features | (B, T, 2688) |

`B` is batch size, `T` is sequence length. A "token" is one (B, T) entry — so a
batch of 4 sequences of length 1024 produces 4 × 1024 = 4 096 token-activations
per forward pass.

---

## 7. Why train *different* SAEs on *different* components?

A common newcomer question: "isn't `resid_post` enough, since everything
eventually flows into the highway anyway?"

Two reasons it isn't:

1. **Different sites encode different things.** Attention contributes
   long-range relational features ("X refers to Y"); MoE contributes
   knowledge-like features ("Paris is in France"); Mamba-2 contributes
   smoother local-mixing features. An SAE trained on `resid_post` sees a
   *mixture* of all of these and may struggle to separate them. SAEs trained
   on `attn_out_prelinear` vs `moe_out` find different feature families.
2. **Per-expert SAEs probe specialization.** Training an SAE per expert
   reveals what each expert has chosen to specialize in (math? code? German?
   syntax?), which a residual-stream SAE alone cannot see, because by the time
   you're on the highway the experts' contributions have been weighted-summed
   and mixed with the shared-expert contribution.

In practice: start with `resid_post` at a mid layer (this project's dev
default is layer 25), get the pipeline working end-to-end, then expand
outward — to `attn_out_prelinear`, then `moe_out`, then individual experts.

---

## 8. References

- *Gemma Scope 2 Technical Paper* (Google DeepMind, Sept 2025) — defines the
  three-sites-per-layer hooking convention this project follows.
- *Nemotron 3 Nano Technical Report* (NVIDIA, Dec 2025, arXiv 2512.20848) —
  the architecture this document describes.
- *Mamba-2: Transformers Are SSMs* (Dao & Gu, 2024) — the SSM mixer used here.
- *DeepSeek-MoE* (Dai et al., 2024) — the granular-MoE design with shared
  experts that inspired Nemotron's MoE layers.
- *Towards Monosemanticity* (Anthropic, 2023) — the foundational SAE-on-LM
  paper that established the residual-stream-SAE recipe.
