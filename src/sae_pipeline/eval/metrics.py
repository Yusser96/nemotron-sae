"""Core SAE evaluation metrics, mirroring Gemma Scope 2 §4.

- L0: average per-token active-latent count (the "true" non-differentiable L0)
- FVU: fraction of variance unexplained = MSE(x, x̂) / Var(x)
- Dead-feature %: latents that never fire over a sample of N tokens
- ΔCE: cross-entropy increase when SAE reconstruction is patched into the LM forward
- Distributions for plotting:
    * firing_frequency: shape (d_sae,) — per-latent firing fraction
    * l0_per_token:     shape (n_tokens,) — int latent count per token
    * recon_err_per_token: shape (n_tokens,) — squared L2 error per token
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from sae_pipeline.sae.base import SparseAutoencoder


@dataclass
class ReconMetrics:
    l0: float
    fvu: float
    dead_pct: float
    n_tokens: int


@dataclass
class ReconArrays:
    """Distributions used for plotting (saved to .npz alongside the JSON summary)."""
    firing_frequency: np.ndarray   # (d_sae,) float32
    l0_per_token: np.ndarray       # (n_tokens,) int32
    recon_err_per_token: np.ndarray  # (n_tokens,) float32


@torch.no_grad()
def reconstruction_metrics(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    dead_threshold_tokens: int = 50_000,
    return_arrays: bool = False,
) -> ReconMetrics | tuple[ReconMetrics, ReconArrays]:
    """Compute L0, FVU, and dead-% on a tensor of activations (n_tokens, d_in).

    With `return_arrays=True`, additionally return per-latent firing frequency,
    per-token L0, and per-token reconstruction error — all consumed by the
    plotting module to draw histograms.
    """
    sae.eval()
    f = sae.encode(activations)
    x_hat = sae.decode(f)

    active = (f > 0).to(activations.dtype)
    l0_per_token = active.sum(dim=-1)
    l0 = l0_per_token.mean().item()

    err_per_token = (activations - x_hat).pow(2).sum(dim=-1)
    mse = err_per_token.mean().item() / activations.shape[-1]
    var = activations.var(dim=0).mean().item()
    fvu = mse / max(var, 1e-12)

    # Dead-feature percentage: count latents that never fired across the sample.
    n = activations.shape[0]
    sample_n = min(n, dead_threshold_tokens)
    firing_frequency = active[:sample_n].mean(dim=0)
    fired_any = firing_frequency > 0
    dead_pct = 100.0 * (1.0 - fired_any.float().mean().item())

    metrics = ReconMetrics(l0=l0, fvu=fvu, dead_pct=dead_pct, n_tokens=n)
    if not return_arrays:
        return metrics
    arrays = ReconArrays(
        firing_frequency=firing_frequency.detach().cpu().float().numpy(),
        l0_per_token=l0_per_token.detach().cpu().to(torch.int32).numpy(),
        recon_err_per_token=err_per_token.detach().cpu().float().numpy(),
    )
    return metrics, arrays


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
