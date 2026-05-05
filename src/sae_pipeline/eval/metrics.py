"""Core SAE evaluation metrics, mirroring Gemma Scope 2 §4.

- L0: average per-token active-latent count (the "true" non-differentiable L0)
- FVU: fraction of variance unexplained = MSE(x, x̂) / Var(x)
- Dead-feature %: latents that never fire over a sample of N tokens
- ΔCE: cross-entropy increase when SAE reconstruction is patched into the LM forward
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sae_pipeline.sae.base import SparseAutoencoder


@dataclass
class ReconMetrics:
    l0: float
    fvu: float
    dead_pct: float
    n_tokens: int


@torch.no_grad()
def reconstruction_metrics(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    dead_threshold_tokens: int = 50_000,
) -> ReconMetrics:
    """Compute L0, FVU, and dead-% on a tensor of activations (n_tokens, d_in)."""
    sae.eval()
    f = sae.encode(activations)
    x_hat = sae.decode(f)

    l0 = (f > 0).to(activations.dtype).sum(dim=-1).mean().item()

    mse = (activations - x_hat).pow(2).mean().item()
    var = activations.var(dim=0).mean().item()
    fvu = mse / max(var, 1e-12)

    # Dead-feature percentage: count latents that never fired across the sample.
    n = activations.shape[0]
    sample_n = min(n, dead_threshold_tokens)
    fired_any = (f[:sample_n] > 0).any(dim=0)
    dead_pct = 100.0 * (1.0 - fired_any.float().mean().item())

    return ReconMetrics(l0=l0, fvu=fvu, dead_pct=dead_pct, n_tokens=n)


@torch.no_grad()
def delta_ce(
    model,
    tokenizer,
    sae: SparseAutoencoder,
    sequences: torch.Tensor,           # (B, T) token ids
    hook_register,                     # callable: (model, sae) -> contextmanager
) -> dict[str, float]:
    """Run the model with and without the SAE patched in; return CE delta.

    `hook_register` should install a forward hook that replaces the activation at
    the target site with sae.decode(sae.encode(activation)). The exact wiring
    depends on the component, so it's passed in.
    """
    sae.eval()
    sequences = sequences.to(next(model.parameters()).device)

    # Baseline: clean forward pass
    out_clean = model(sequences, labels=sequences)
    ce_clean = float(out_clean.loss)

    # Patched: SAE in the loop
    with hook_register(model, sae):
        out_patched = model(sequences, labels=sequences)
        ce_patched = float(out_patched.loss)

    return {
        "ce_clean": ce_clean,
        "ce_patched": ce_patched,
        "delta_ce": ce_patched - ce_clean,
    }
