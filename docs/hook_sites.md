# Hook sites

Each SAE trains on one **site**: a layer index plus a component name. Sites are
named by slug, for example `L26_resid_post` or `L12_attn_out_prelinear`
(`hooks/components.py`). `model/topology.py` maps them onto the module paths of
the `trust_remote_code` model; `python -m sae_pipeline.model.topology` writes
that map to JSON so naming drift shows up before any forward pass.

## Model layout

Nemotron 3 Nano 30B-A3B has 52 blocks over a residual stream of width
`d_model = 2688`. Each block is one of three kinds:

| Block | Count | Notes |
|---|---:|---|
| Mamba-2 mixer | 23 | state-space token mixing |
| GQA attention | 6 | 32 query heads × 128, 2 key/value heads |
| MoE feed-forward | 23 | 128 routed experts (top 6 per token) + 2 shared experts |

Every block reads the residual stream through an RMSNorm and adds its output
back, so `resid_post` of layer L is `resid_pre` of layer L+1.

## Components

| Component | Layers | Width | Captures |
|---|---|---:|---|
| `resid_pre` | all | 2688 | residual stream entering the block |
| `resid_post` | all | 2688 | residual stream leaving the block |
| `mamba_out` | Mamba | 2688 | mixer output before the residual add |
| `attn_out_prelinear` | attention | 4096 | concatenated head outputs before `W_O` |
| `attn_head.<h>` | attention | 128 | one head's slice of `attn_out_prelinear` |
| `moe_out` | MoE | 2688 | weighted sum of routed and shared experts |
| `shared_expert.<s>` | MoE | 2688 | one shared expert, every token |
| `expert.<e>` | MoE | 2688 | one routed expert, only the tokens routed to it |

Routed-expert sites produce a variable number of rows per forward pass.
`cache_activations` rejects them because the resumable cache counts rows per
forward pass; use `dump_activations` for diagnostic expert captures.
