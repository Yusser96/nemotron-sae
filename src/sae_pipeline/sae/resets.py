"""Residual resets: replace rarely firing latents with high-error residual directions."""

from __future__ import annotations

from pathlib import Path

import torch

from sae_pipeline.cache.reader import ShardReader
from sae_pipeline.config import SAECfg
from sae_pipeline.sae.jumprelu import JumpReLUSAE


def active_subspace(
    cache_dir: str | Path,
    *,
    sample_tokens: int,
    rank: int,
    input_scale: float,
    centre: torch.Tensor,
    device: str,
    metadata_dir: Path,
) -> torch.Tensor:
    """Compute and cache leading centred activation directions."""
    from safetensors import safe_open

    path = metadata_dir / f"active_subspace_r{rank}.pt"
    if path.exists():
        basis = torch.load(path, map_location=device, weights_only=True)
        if tuple(basis.shape) == (centre.numel(), rank):
            return basis.to(device=device, dtype=torch.float32)

    manifest = ShardReader(cache_dir).manifest
    chunks: list[torch.Tensor] = []
    seen = 0
    for name in manifest.shard_paths:
        if seen >= sample_tokens:
            break
        with safe_open(str(Path(cache_dir) / name), framework="pt", device="cpu") as handle:
            part = handle.get_tensor("x")[: sample_tokens - seen]
        chunks.append(part.to(device=device, dtype=torch.float32) * input_scale - centre)
        seen += part.shape[0]
    if not chunks:
        raise ValueError(f"Cannot compute an active subspace from {cache_dir}")
    sample = torch.cat(chunks, dim=0)
    _, _, basis = torch.pca_lowrank(sample, q=rank, center=False)
    basis = basis[:, :rank].contiguous()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    torch.save(basis.detach().cpu(), path)
    return basis


@torch.no_grad()
def load_reset_calibration(
    sae: JumpReLUSAE,
    cache_dir: str | Path,
    *,
    sample_tokens: int,
    pool_size: int,
    normalization_scale: float,
    activation_center: torch.Tensor | None,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a bounded calibration sample and measure exact reset statistics.

    The returned tensors are the normalised calibration activations, a pool of
    highest-error residuals, and per-latent firing frequencies.  Resetting from
    one optimisation batch was too noisy for L26; this pass makes threshold
    calibration depend on the requested support window instead.
    """
    reader = ShardReader(cache_dir)
    chunks: list[torch.Tensor] = []
    seen = 0
    for shard in reader.iter_shards():
        if seen >= sample_tokens:
            break
        take = min(shard.shape[0], sample_tokens - seen)
        chunks.append(shard[:take].contiguous())
        seen += take
    if seen == 0:
        raise ValueError(f"Cannot calibrate a reset from empty cache {cache_dir}")
    calibration_x = torch.cat(chunks, dim=0).to(device=device, dtype=torch.float32)
    if normalization_scale != 1.0:
        calibration_x.mul_(normalization_scale)
    if activation_center is not None:
        calibration_x.sub_(activation_center)

    frequencies = torch.zeros(sae.d_sae, dtype=torch.float64, device=device)
    best_errors = torch.empty(0, dtype=torch.float32, device=device)
    best_residuals = torch.empty((0, sae.d_in), dtype=torch.float32, device=device)
    for start in range(0, calibration_x.shape[0], batch_size):
        x = calibration_x[start : start + batch_size]
        x_hat, f = sae(x)
        frequencies.add_((f > 0).sum(dim=0, dtype=torch.float64))
        errors = (x - x_hat).square().sum(dim=-1)
        k = min(pool_size, errors.numel())
        values, indices = torch.topk(errors, k=k, largest=True, sorted=False)
        residuals = (x - x_hat)[indices].detach()
        if best_errors.numel():
            values = torch.cat([best_errors, values])
            residuals = torch.cat([best_residuals, residuals])
        keep = min(pool_size, values.numel())
        keep_values, keep_indices = torch.topk(values, k=keep, largest=True, sorted=False)
        best_errors = keep_values
        best_residuals = residuals[keep_indices]
    return calibration_x, best_residuals, frequencies / calibration_x.shape[0]


@torch.no_grad()
def _zero_optimizer_slices(
    optim: torch.optim.Optimizer,
    parameter: torch.Tensor,
    indices: torch.Tensor,
    *,
    column_parameter: bool,
) -> None:
    """Reset Adam moments for selected SAE columns after feature replacement."""

    state = optim.state.get(parameter)
    if not state:
        return
    for value in state.values():
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
            continue
        if column_parameter:
            value[:, indices] = 0
        else:
            value[indices] = 0


def residual_reset_due(cfg: SAECfg, step: int) -> bool:
    """Whether a ``residual_reset`` strategy resamples features after ``step``."""
    return (
        cfg.feature_use_strategy in {"residual_reset", "frequency_residual_reset"}
        and step >= cfg.residual_reset_start_step
        and (step - cfg.residual_reset_start_step) % cfg.residual_reset_every_steps == 0
    )


def reset_l0_allowance(cfg: SAECfg, step: int, reset_step: int | None) -> float:
    """Extra target L0 for the features revived at ``reset_step``, annealed to zero."""
    if reset_step is None or step - reset_step >= cfg.residual_reset_anneal_steps:
        return 0.0
    return cfg.residual_reset_delta_l0 * (1.0 - (step - reset_step) / max(1, cfg.residual_reset_anneal_steps))


@torch.no_grad()
def resample_underused_features(
    sae: JumpReLUSAE,
    optim: torch.optim.Optimizer,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    feature_frequency: torch.Tensor,
    ever_fired: torch.Tensor,
    cfg: SAECfg,
    active_subspace: torch.Tensor | None = None,
    calibration_x: torch.Tensor | None = None,
    calibration_residuals: torch.Tensor | None = None,
    tokens_since_fired: torch.Tensor | None = None,
) -> int:
    """Replace low-frequency features with high-error residual directions.

    This is intentionally a bounded intervention: it touches only a fixed
    number of columns and resets the matching Adam moments.  The starting SAE
    is a dictionary initialisation, so preserving the identities of replaced
    latents is not a requirement; their firing history, including
    ``tokens_since_fired`` when AuxK tracks it, starts over.
    """

    candidates = torch.nonzero(
        feature_frequency < cfg.residual_reset_frequency_threshold,
        as_tuple=False,
    ).flatten()
    if candidates.numel() == 0:
        return 0
    available = (calibration_residuals.shape[0]
                 if calibration_x is not None and calibration_residuals is not None
                 else min(cfg.residual_reset_pool_size, x.shape[0]))
    n_reset = min(cfg.residual_reset_max_features, int(candidates.numel()), available)
    if n_reset == 0:
        return 0
    candidates = candidates[torch.randperm(candidates.numel(), device=x.device)[:n_reset]]

    residual = x - x_hat
    if calibration_x is not None and calibration_residuals is not None:
        if calibration_residuals.shape[0] < n_reset:
            raise ValueError("Reset residual pool is smaller than the requested reset")
        directions = calibration_residuals[
            torch.randperm(calibration_residuals.shape[0], device=x.device)[:n_reset]
        ]
    else:
        errors = residual.square().sum(dim=-1)
        pool_size = min(cfg.residual_reset_pool_size, x.shape[0])
        probabilities = errors.clamp_min(torch.finfo(errors.dtype).eps)
        probabilities = probabilities / probabilities.sum()
        sample_indices = torch.multinomial(probabilities, pool_size, replacement=False)
        directions = residual[sample_indices[:n_reset]]
    norms = directions.norm(dim=-1, keepdim=True)
    invalid = norms.squeeze(-1) <= 1.0e-8
    if invalid.any():
        directions[invalid] = torch.randn_like(directions[invalid])
        norms = directions.norm(dim=-1, keepdim=True)
    directions = directions / norms.clamp_min(1.0e-8)
    if active_subspace is not None:
        directions = directions @ active_subspace @ active_subspace.T
        norms = directions.norm(dim=-1, keepdim=True)
        invalid = norms.squeeze(-1) <= 1.0e-8
        if invalid.any():
            replacement = torch.randn(
                int(invalid.sum().item()), active_subspace.shape[1], device=x.device
            ) @ active_subspace.T
            directions[invalid] = replacement
            norms = directions.norm(dim=-1, keepdim=True)
        directions = directions / norms.clamp_min(1.0e-8)

    live = feature_frequency > 0
    live_norms = sae.W_enc[:, live].norm(dim=0)
    encoder_scale = (
        0.2 * live_norms.median().item() if live_norms.numel() else 0.2
    )
    encoder_scale = max(encoder_scale, 1.0e-3)

    sae.W_dec[:, candidates] = directions.T
    sae.W_enc[:, candidates] = directions.T * encoder_scale
    sae.b_enc[candidates] = 0

    calibration = x if calibration_x is None else calibration_x
    centred = calibration - sae.b_dec if sae.pre_encoder_bias else calibration
    projection = torch.relu(centred @ sae.W_enc[:, candidates] + sae.b_enc[candidates])
    target_frequency = (cfg.residual_reset_target_frequency
                        if cfg.residual_reset_target_frequency is not None
                        else min(1.0, cfg.residual_reset_delta_l0 / n_reset))
    quantile = torch.quantile(projection, 1.0 - target_frequency, dim=0)
    threshold = quantile.clamp_min(max(cfg.bandwidth, 1.0e-6))
    sae.log_theta[candidates] = threshold.log()

    _zero_optimizer_slices(optim, sae.W_dec, candidates, column_parameter=True)
    _zero_optimizer_slices(optim, sae.W_enc, candidates, column_parameter=True)
    _zero_optimizer_slices(optim, sae.b_enc, candidates, column_parameter=False)
    _zero_optimizer_slices(optim, sae.log_theta, candidates, column_parameter=False)
    ever_fired[candidates] = False
    feature_frequency[candidates] = 0
    if tokens_since_fired is not None:
        tokens_since_fired[candidates] = 0
    return n_reset
