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
    active_features: int
    top1_activation_mass_pct: float
    max_firing_frequency: float
    features_over_5pct: int
    firing_gini: float
    firing_entropy_nats: float
    firing_entropy_normalised: float


@dataclass
class ReconArrays:
    """Distributions used for plotting (saved to .npz alongside the JSON summary)."""
    firing_frequency: np.ndarray   # (d_sae,) float32
    l0_per_token: np.ndarray       # (n_tokens,) int32
    recon_err_per_token: np.ndarray  # (n_tokens,) float32


@dataclass
class FiringConcentration:
    active_features: int
    top1_activation_mass_pct: float
    max_firing_frequency: float


def dead_pct_from_fired(fired_any: torch.Tensor) -> float:
    """Percentage of latents with no entry in a "has this ever fired" mask.

    Callers pass different masks on purpose: a cumulative all-time
    ``ever_fired`` flag and a windowed "fired within this sample" flag are
    not the same quantity, only the dead-percentage formula is shared.
    """
    return 100.0 * (1.0 - fired_any.float().mean().item())


def firing_concentration(frequency: torch.Tensor) -> FiringConcentration:
    """Active-latent count, top-1% activation mass and peak frequency.

    ``frequency`` is any non-negative per-latent tally where zero means the
    latent never fired (a fired-count, an EMA frequency, or a firing
    fraction all give the same result since only relative magnitude within
    the tensor matters).
    """
    fired_any = frequency > 0
    total = float(frequency.sum().item())
    top_k = max(1, frequency.numel() // 100)
    top1_mass = 0.0 if total <= 0 else 100.0 * float(
        torch.topk(frequency, top_k).values.sum().item() / total
    )
    return FiringConcentration(
        active_features=int(fired_any.sum().item()),
        top1_activation_mass_pct=top1_mass,
        max_firing_frequency=float(frequency.max().item()),
    )


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
    return streaming_reconstruction_metrics(
        sae,
        (activations,),
        dead_threshold_tokens=dead_threshold_tokens,
        return_arrays=return_arrays,
    )


def _gini(values: torch.Tensor) -> float:
    values = values.detach().to(torch.float64).flatten().clamp_min(0)
    if values.numel() == 0 or float(values.sum()) <= 0:
        return 0.0
    ordered = torch.sort(values).values
    n = ordered.numel()
    index = torch.arange(1, n + 1, device=ordered.device, dtype=ordered.dtype)
    numerator = (2 * index - n - 1) @ ordered
    return float((numerator / (n * ordered.sum())).item())


@torch.no_grad()
def streaming_reconstruction_metrics(
    sae: SparseAutoencoder,
    activation_batches,
    *,
    dead_threshold_tokens: int = 50_000,
    return_arrays: bool = False,
) -> ReconMetrics | tuple[ReconMetrics, ReconArrays]:
    """Evaluate without materialising the full latent matrix.

    ``activation_batches`` yields CPU or device tensors of shape ``(N, d_in)``.
    Reconstruction, support counts and moments are accumulated batch by batch,
    which makes million-token support evaluation practical for wide SAEs.
    """
    sae.eval()
    device = next(sae.parameters()).device
    n_total = 0
    support_seen = 0
    l0_sum = 0.0
    err_sum = 0.0
    x_sum = None
    x_sq_sum = None
    support_counts = torch.zeros(sae.d_sae, dtype=torch.float64, device=device)
    l0_parts: list[torch.Tensor] = []
    err_parts: list[torch.Tensor] = []

    for batch in activation_batches:
        if batch.numel() == 0:
            continue
        activations = batch.to(device, dtype=torch.float32, non_blocking=True)
        f = sae.encode(activations)
        x_hat = sae.decode(f)
        active = f > 0
        l0_per_token = active.sum(dim=-1)
        err_per_token = (activations - x_hat).pow(2).sum(dim=-1)

        n_batch = activations.shape[0]
        n_total += n_batch
        l0_sum += float(l0_per_token.sum().item())
        err_sum += float(err_per_token.sum().item())
        batch_sum = activations.sum(dim=0, dtype=torch.float64)
        batch_sq_sum = activations.square().sum(dim=0, dtype=torch.float64)
        x_sum = batch_sum if x_sum is None else x_sum + batch_sum
        x_sq_sum = batch_sq_sum if x_sq_sum is None else x_sq_sum + batch_sq_sum

        take = min(n_batch, max(0, dead_threshold_tokens - support_seen))
        if take:
            support_counts.add_(active[:take].sum(dim=0, dtype=torch.float64))
            support_seen += take
        if return_arrays:
            l0_parts.append(l0_per_token.detach().cpu().to(torch.int32))
            err_parts.append(err_per_token.detach().cpu().float())

    if n_total == 0 or x_sum is None or x_sq_sum is None:
        raise ValueError("Cannot evaluate an empty activation stream")
    n_float = float(n_total)
    firing_frequency = support_counts / max(1, support_seen)
    concentration = firing_concentration(firing_frequency)
    total_frequency = float(firing_frequency.sum().item())
    probabilities = firing_frequency / max(total_frequency, 1.0e-12)
    positive = probabilities > 0
    entropy_nats = float(
        (-(probabilities[positive] * probabilities[positive].log()).sum()).item()
    )
    entropy_normalised = entropy_nats / max(1.0e-12, float(torch.log(torch.tensor(float(sae.d_sae))).item()))
    # FVU = sum_t ||x_t - x_hat_t||^2 / sum_t ||x_t - x_bar||^2 (a ratio of
    # sums over the same token set, so n_total cancels -- do not renormalise
    # the numerator and denominator by different divisors here).
    variance_sum = x_sq_sum.sum() - x_sum.square().sum() / n_float
    fvu = err_sum / max(float(variance_sum.item()), 1.0e-12)
    metrics = ReconMetrics(
        l0=l0_sum / n_float,
        fvu=fvu,
        dead_pct=dead_pct_from_fired(firing_frequency > 0),
        n_tokens=n_total,
        active_features=concentration.active_features,
        top1_activation_mass_pct=concentration.top1_activation_mass_pct,
        max_firing_frequency=concentration.max_firing_frequency,
        features_over_5pct=int((firing_frequency > 0.05).sum().item()),
        firing_gini=_gini(firing_frequency),
        firing_entropy_nats=entropy_nats,
        firing_entropy_normalised=entropy_normalised,
    )
    if not return_arrays:
        return metrics
    arrays = ReconArrays(
        firing_frequency=firing_frequency.detach().cpu().float().numpy(),
        l0_per_token=torch.cat(l0_parts).numpy(),
        recon_err_per_token=torch.cat(err_parts).numpy(),
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
