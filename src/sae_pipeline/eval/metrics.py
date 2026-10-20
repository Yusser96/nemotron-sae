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

from dataclasses import dataclass

import numpy as np
import torch

from sae_pipeline.sae.jumprelu import inverse_simpson_from_mass
from sae_pipeline.sae.jumprelu import JumpReLUSAE


@dataclass
class ReconMetrics:
    """Summary metrics from evaluation over an activation stream.

    Units:
        l0: Mean active latents per token (count).
        fvu: Fraction of variance unexplained (ratio, lower is better).
        dead_pct: Inactive latent percentage (0.0 to 100.0).
        n_tokens: Total tokens evaluated.
        d_is: Inverse-Simpson effective dictionary size in [0, d_sae].
        u_is: Inverse-Simpson dictionary utilization fraction in [0.0, 1.0] (d_is / d_sae).
        d_50: Percentage of dictionary accounting for 50% of activation mass (0.0 to 100.0).
        d_90: Percentage of dictionary accounting for 90% of activation mass (0.0 to 100.0).
        d_99: Percentage of dictionary accounting for 99% of activation mass (0.0 to 100.0).
        q10..q99: Quantiles of firing frequency p_j among active latents (p_j > 0).
    """

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
    d_is: float = 0.0
    u_is: float = 0.0
    d_50: float = 0.0
    d_90: float = 0.0
    d_99: float = 0.0
    q10: float = 0.0
    q25: float = 0.0
    q50: float = 0.0
    q75: float = 0.0
    q90: float = 0.0
    q99: float = 0.0
    support_n_tokens: int = 0
    explained_variance: float = 0.0
    l0_p10: float = 0.0
    l0_p50: float = 0.0
    l0_p90: float = 0.0
    l0_p99: float = 0.0
    rare_nonzero_lt_1e5: int = 0
    rare_nonzero_lt_1e4: int = 0


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
    sae: JumpReLUSAE,
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
    sae: JumpReLUSAE,
    activation_batches,
    *,
    dead_threshold_tokens: int = 50_000,
    max_tokens: int | None = None,
    return_arrays: bool = False,
) -> ReconMetrics | tuple[ReconMetrics, ReconArrays]:
    """Evaluate without materialising the full latent matrix.

    ``activation_batches`` yields CPU or device tensors of shape ``(N, d_in)``.
    Reconstruction, support counts and moments are accumulated batch by batch,
    which makes million-token support evaluation practical for wide SAEs.
    ``max_tokens`` bounds reconstruction/L0 independently of the support window;
    the stream is read through the larger requested window.
    """
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if dead_threshold_tokens <= 0:
        raise ValueError("dead_threshold_tokens must be positive")
    sae.eval()
    device = next(sae.parameters()).device
    n_total = 0
    support_seen = 0
    l0_sum = 0.0
    err_sum = 0.0
    x_mean = None
    residual_sum = None
    variance_sum = torch.zeros((), dtype=torch.float64, device=device)
    support_counts = torch.zeros(sae.d_sae, dtype=torch.float64, device=device)
    l0_parts: list[torch.Tensor] = []
    err_parts: list[torch.Tensor] = []

    for batch in activation_batches:
        if batch.numel() == 0:
            continue
        activations = batch.to(device, dtype=torch.float32, non_blocking=True)
        if max_tokens is not None:
            remaining = max(max_tokens - n_total, dead_threshold_tokens - support_seen)
            if remaining <= 0:
                break
            activations = activations[:remaining]
        f = sae.encode(activations)
        active = f > 0
        take = min(activations.shape[0], max(0, dead_threshold_tokens - support_seen))
        if take:
            support_counts.add_(active[:take].sum(dim=0, dtype=torch.float64))
            support_seen += take
        n_batch = activations.shape[0] if max_tokens is None else min(activations.shape[0], max_tokens - n_total)
        if n_batch <= 0:
            continue
        activations = activations[:n_batch]
        x_hat = sae.decode(f[:n_batch])
        l0_per_token = active[:n_batch].sum(dim=-1)
        residual = activations - x_hat
        residual_batch_sum = residual.to(torch.float64).sum(dim=0)
        residual_sum = residual_batch_sum if residual_sum is None else residual_sum + residual_batch_sum
        err_per_token = residual.pow(2).sum(dim=-1)
        l0_sum += float(l0_per_token.sum().item())
        err_sum += float(err_per_token.sum().item())
        # Merge centred second moments in float64. E[x²] - E[x]² loses
        # small variances for activation streams with large nonzero means.
        batch_x = activations.to(torch.float64)
        batch_mean = batch_x.mean(dim=0)
        batch_variance_sum = (batch_x - batch_mean).square().sum()
        if x_mean is None:
            x_mean = batch_mean
            variance_sum = batch_variance_sum
        else:
            delta = batch_mean - x_mean
            combined_n = n_total + n_batch
            variance_sum += batch_variance_sum + delta.square().sum() * (
                n_total * n_batch / combined_n
            )
            x_mean += delta * (n_batch / combined_n)
        n_total += n_batch

        l0_parts.append(l0_per_token.detach().cpu().to(torch.int32))
        if return_arrays:
            err_parts.append(err_per_token.detach().cpu().float())

    if n_total == 0 or x_mean is None:
        raise ValueError("Cannot evaluate an empty activation stream")
    n_float = float(n_total)
    firing_frequency = support_counts / max(1, support_seen)
    concentration = firing_concentration(firing_frequency)
    total_frequency = float(firing_frequency.sum().item())
    d_is_tensor, u_is_tensor, probabilities = inverse_simpson_from_mass(
        firing_frequency, sae.d_sae
    )
    positive = probabilities > 0
    entropy_nats = float(
        (-(probabilities[positive] * probabilities[positive].log()).sum()).item()
    )
    entropy_normalised = entropy_nats / max(1.0e-12, float(torch.log(torch.tensor(float(sae.d_sae))).item()))

    # Inverse-Simpson effective dictionary size and utilization fraction
    d_is = float(d_is_tensor.item())
    u_is = float(u_is_tensor.item())

    # Concentration: latents for 50%, 90%, 99% mass as percentage of dictionary
    sorted_probs, _ = torch.sort(probabilities, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=0)
    k50 = int((cum_probs < 0.50).sum().item()) + 1 if total_frequency > 0 else 0
    k90 = int((cum_probs < 0.90).sum().item()) + 1 if total_frequency > 0 else 0
    k99 = int((cum_probs < 0.99).sum().item()) + 1 if total_frequency > 0 else 0
    d_50 = 100.0 * k50 / float(sae.d_sae)
    d_90 = 100.0 * k90 / float(sae.d_sae)
    d_99 = 100.0 * k99 / float(sae.d_sae)

    # Quantiles of active firing frequencies
    active_freqs = firing_frequency[firing_frequency > 0]
    if active_freqs.numel() > 0:
        q_tensor = torch.quantile(
            active_freqs.float(),
            torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.99], device=active_freqs.device),
        )
        q10, q25, q50, q75, q90, q99 = [float(v.item()) for v in q_tensor]
    else:
        q10 = q25 = q50 = q75 = q90 = q99 = 0.0

    # FVU = sum_t ||x_t - x_hat_t||^2 / sum_t ||x_t - x_bar||^2 (a ratio of
    # sums over the same token set, so n_total cancels -- do not renormalise
    # the numerator and denominator by different divisors here).
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
        d_is=d_is,
        u_is=u_is,
        d_50=d_50,
        d_90=d_90,
        d_99=d_99,
        q10=q10,
        q25=q25,
        q50=q50,
        q75=q75,
        q90=q90,
        q99=q99,
        support_n_tokens=support_seen,
        explained_variance=1.0 - max(0.0, err_sum - float(residual_sum.square().sum()) / n_float) / max(float(variance_sum), 1e-12),
        l0_p10=float(torch.quantile(torch.cat(l0_parts).float(), 0.10)),
        l0_p50=float(torch.quantile(torch.cat(l0_parts).float(), 0.50)),
        l0_p90=float(torch.quantile(torch.cat(l0_parts).float(), 0.90)),
        l0_p99=float(torch.quantile(torch.cat(l0_parts).float(), 0.99)),
        rare_nonzero_lt_1e5=int(((firing_frequency > 0) & (firing_frequency < 1e-5)).sum()),
        rare_nonzero_lt_1e4=int(((firing_frequency > 0) & (firing_frequency < 1e-4)).sum()),
    )
    if not return_arrays:
        return metrics
    try:
        arr_firing = firing_frequency.detach().cpu().float().numpy()
        arr_l0 = torch.cat(l0_parts).numpy() if l0_parts else np.zeros((0,), dtype=np.int32)
        arr_err = torch.cat(err_parts).numpy() if err_parts else np.zeros((0,), dtype=np.float32)
    except RuntimeError:
        arr_firing = np.array(firing_frequency.detach().cpu().float().tolist(), dtype=np.float32)
        arr_l0 = np.array(torch.cat(l0_parts).tolist(), dtype=np.int32) if l0_parts else np.zeros((0,), dtype=np.int32)
        arr_err = np.array(torch.cat(err_parts).tolist(), dtype=np.float32) if err_parts else np.zeros((0,), dtype=np.float32)
    arrays = ReconArrays(
        firing_frequency=arr_firing,
        l0_per_token=arr_l0,
        recon_err_per_token=arr_err,
    )
    return metrics, arrays


@torch.no_grad()
def delta_ce(
    model,
    tokenizer,
    sae: JumpReLUSAE,
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
