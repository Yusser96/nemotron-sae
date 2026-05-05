# nemotron-sae

Sparse-autoencoder (SAE) and transcoder training pipeline for
[`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16),
NVIDIA's 30 B-parameter hybrid **Mamba-2 + GQA-Attention + MoE** language model.

The pipeline is configurable per `(layer, component)` so each can be trained as an
independent parallel job, supports a **dev** preset (smoke test, 100 documents) and
a **prod** preset (full sweep), and uses a sharded on-disk activation cache so
a single 30 B-model forward pass can feed an unbounded number of SAE training
runs without reloading the LM.

## What it implements

- **JumpReLU SAE** with the exact recipe from
  [Gemma Scope 2](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf):
  quadratic L0 penalty around a target `L0*`, learnable per-latent threshold,
  rectangular-kernel STE for the Heaviside / L0 derivatives (bandwidth `ε = 0.001`),
  decoder columns renormalized to unit norm after every Adam step,
  decoder gradient projected orthogonal to those columns,
  pre-encoder bias subtracted, `W_enc` tied at init then untied,
  Adam `(β₁, β₂) = (0, 0.999)`, peak LR `7e-5`, batch 4096 tokens,
  cosine warmup `0.1 LR → LR` over 1 000 steps,
  L0 coefficient λ linearly warmed up over 50 000 steps.

- **Hookable components** (per layer, where applicable):
  `resid_pre`, `resid_post`, `mamba_out`, `attn_out_prelinear`,
  `attn_head.<h>`, `moe_out`, `expert.<e>`, `shared_expert.<s>`.
  Module paths are introspected at load time (`model_topology.json`), since the
  model uses `trust_remote_code` and the modeling file is the source of truth.

- **Shard cache**: safetensors files of shuffled activations + a JSON manifest.
  `ShardWriter` flushes when the buffer crosses `shard_size_bytes`; `ShardReader`
  memory-maps shards and stitches them across batch boundaries.
  `ActivationBuffer` keeps a rolling `n_batches_in_buffer × batch_size` buffer
  that refills when half-empty.

- **Eval**: L0, FVU, dead-feature %, ΔCE (cross-entropy increase when the SAE
  reconstruction is patched into the LM forward pass).

- **Job launcher**: materializes the cartesian product of
  `layers × components × archs × widths × L0 targets` into shell commands and
  runs them in parallel via a `ProcessPoolExecutor`, pinning each job to one GPU
  via `CUDA_VISIBLE_DEVICES`. `--executor dry` just prints the commands.

## Quick start

```bash
# 1. Set up the venv (auto-detects macOS vs Linux/CUDA)
bash scripts/setup_venv.sh
source .venv/bin/activate
source env.sh    # HF_TOKEN, OPENAI_API_KEY (gitignored)

# 2. Run the unit-test suite (works on CPU / macOS — covers the full pipeline
#    end-to-end on synthetic activations; no LM needed)
pytest tests/ -v

# 3. On a CUDA host: run the dev smoke test (cache → train → eval, ~30 min)
bash scripts/dev_smoke_test.sh
```

The dev smoke test caches activations from `nvidia/Nemotron-CC-v2.1` (same data
distribution as production, but only the first 100 documents) at layer 25
component `resid_post`, trains a tiny `d_sae=4096` JumpReLU SAE for 1 000 steps,
and evaluates it.

## Per-(layer, component) jobs

The two phases are decoupled CLIs:

```bash
# Phase A: extract activations to a sharded cache (one-time per layer/component)
python -m sae_pipeline.cli.cache_activations \
    --config configs/dev.yaml --layer 25 --component resid_post

# Phase B: train an SAE on a cache (parameterized over arch / width / L0)
python -m sae_pipeline.cli.train_sae \
    --config configs/dev.yaml --layer 25 --component resid_post \
    --arch jumprelu --width 4096 --l0 30

# Eval
python -m sae_pipeline.cli.evaluate \
    --config configs/dev.yaml --layer 25 --component resid_post \
    --arch jumprelu --width 4096 --l0 30
```

Components: `resid_pre`, `resid_post`, `mamba_out`, `attn_out_prelinear`,
`moe_out`, `expert.<e>`, `shared_expert.<s>`, `attn_head.<h>`.

## Sweep launcher

```bash
# Print the full prod sweep (15 cache jobs + 270 train+eval jobs = 285 total)
python -m sae_pipeline.cli.launch --config configs/prod.yaml --executor dry

# Run them locally, one process per visible GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m sae_pipeline.cli.launch --config configs/prod.yaml --executor local
```

## Configs

`configs/dev.yaml` and `configs/prod.yaml` differ only in:

| Knob               | Dev                | Prod                          |
| ------------------ | ------------------ | ----------------------------- |
| `n_documents`      | 100                | (uses `total_tokens` instead) |
| `total_tokens`     | —                  | 200 B                         |
| `seq_len`          | 1024               | 2048                          |
| `n_steps`          | 1 000              | 200 000                       |
| `d_sae`            | 4096               | sweep `[16k, 64k, 256k]`      |
| `l0_target`        | 30                 | sweep `[10, 50, 100]`         |
| `target.layer`     | 25                 | `[5, 12, 25, 38, 51]`         |
| `target.component` | `resid_post`       | `[resid_post, moe_out, attn_out_prelinear]` |

The same code path serves both — only the budget knobs differ — so any
data-induced bug surfaces in dev too.

## Architecture (verified from the Nemotron 3 Nano technical report)

| Property                      | Value                                  |
| ----------------------------- | -------------------------------------- |
| Total layers                  | 52 (23 Mamba-2 + 6 Attn + 23 MoE)      |
| `d_model`                     | 2688                                   |
| Q-heads / KV-heads / head dim | 32 / 2 / 128                           |
| Mamba state / heads / head dim| 128 / 64 / 64 (8 groups)               |
| MoE experts (routed / shared) | 128 / 2                                |
| MoE active per token          | 6 routed                               |
| Expert intermediate dim       | 1856                                   |
| Total / active params         | 31.6 B / 3.2 B (3.6 B w/ embeddings)   |
| Loader requirements           | `transformers ≥ 4.57.3`, `mamba_ssm`, `causal-conv1d`, `trust_remote_code=True` |

## Library stack

- **Model load**: `transformers` + `mamba_ssm` + `causal-conv1d` (the model
  has real Mamba-2 mixer layers, so the SSM kernels are required).
- **Hooking**: raw PyTorch forward hooks via the `extractor` module
  (works with any `nn.Module`, including `trust_remote_code` custom architectures).
  `nnsight` is also installed as a fallback / alternative.
- **SAE math**: hand-written, mirroring Gemma Scope 2 conventions.
  The activation function is pluggable; `topk`, `batchtopk`, and `matryoshka`
  are reserved as future drop-ins.
- **Cache**: `safetensors` shards + a JSON manifest.
- **Data**: HF `datasets` streaming from
  [`nvidia/Nemotron-CC-v2.1`](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1).

## Memory & GPU efficiency

- LM weights live on GPU only (`device_map="auto"`); BF16 by default,
  FP8 sibling repo or 4-bit `bitsandbytes` for ≤ 24 GB GPUs.
- Activation buffer is bounded:
  `n_batches_in_buffer × batch_size × d_model × 4 B` ≈ 360 MB at defaults.
- Shards are memory-mapped, never fully loaded.
- For per-expert hooks, only the routed-token activations are kept
  (≈ 6/128 ≈ 4.7 % occupancy), so per-expert caches are naturally tiny.
- One SAE training process per `(layer, component, arch, width, L0)` combination,
  pinned to a single GPU; the launcher spreads jobs across visible GPUs.

## References

- McDougall, Conmy, Kramár, Lieberum, Rajamanoharan, Nanda. *Gemma Scope 2 — Technical Paper.* Google DeepMind, Sept 2025. ([PDF](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf))
- Lieberum et al. *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.* arXiv:2408.05147, ICLR 2025.
- Rajamanoharan et al. *Improving SAEs by training with JumpReLU activation.* 2024.
- He et al. *Llama Scope.* arXiv:2410.20526, 2024.
- Dunefsky et al. *Transcoders find interpretable LLM feature circuits.* NeurIPS 2024.
- Gao et al. *Scaling and evaluating sparse autoencoders* (TopK). arXiv:2406.04093.
- NVIDIA. *Nemotron 3 Nano — Technical Report.* arXiv:2512.20848, Dec 2025.

## Project layout

```
.
├── configs/                # dev.yaml, prod.yaml
├── scripts/                # setup_venv.sh, dev_smoke_test.sh
├── src/sae_pipeline/
│   ├── config.py           # Pydantic schemas
│   ├── model/              # loader, topology
│   ├── hooks/              # ComponentSpec, forward-hook extractor
│   ├── data/               # streaming + tokenized packing
│   ├── cache/              # safetensors shard writer / reader / manifest
│   ├── sae/                # base, jumprelu, training loop
│   ├── eval/               # L0, FVU, dead %, ΔCE
│   └── cli/                # cache_activations, train_sae, evaluate, launch
└── tests/                  # 15 unit tests; full synthetic end-to-end pipeline
```
