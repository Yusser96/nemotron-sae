"""The JumpReLU SAE training objective, shared by every trainer.

Schedules map a global step to the coefficients of the loss (learning rate,
L0-penalty weight, Inverse-Simpson weight); :func:`training_loss` evaluates
the loss itself for one activation batch. The same function scores held-out
data, so early stopping can track exactly what training optimises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sae_pipeline.config import SAECfg
from sae_pipeline.sae.base import SparseAutoencoder
from sae_pipeline.sae.jumprelu import JumpReLUSAE, batch_inverse_simpson


def cosine_warmup(step: int, peak_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        # Linear warmup from peak_lr * 0.1 to peak_lr (Gemma Scope 2).
        return peak_lr * (0.1 + 0.9 * step / max(1, warmup_steps))
    # Cosine decay to 0.1 * peak_lr.
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def warmup_constant(step: int, peak_lr: float, warmup_steps: int) -> float:
    """Gemma Scope 2 schedule: 0.1*peak to peak, then no decay."""
    if step >= warmup_steps:
        return peak_lr
    return peak_lr * (0.1 + 0.9 * step / max(1, warmup_steps))


def cosine_warmup_constant(step: int, peak_lr: float, warmup_steps: int) -> float:
    """Cosine ramp from 0.1*peak to peak, followed by a constant rate."""
    if step >= warmup_steps:
        return peak_lr
    progress = step / max(1, warmup_steps)
    cosine_ramp = 0.5 * (1.0 - math.cos(math.pi * progress))
    return peak_lr * (0.1 + 0.9 * cosine_ramp)


def linear_warmup(step: int, peak: float, warmup_steps: int) -> float:
    if step >= warmup_steps:
        return peak
    return peak * (step / max(1, warmup_steps))


def inverse_simpson_warmup_steps(cfg: SAECfg) -> int:
    """The Inverse-Simpson weight ramps with the L0 penalty unless set separately."""
    if cfg.inverse_simpson_warmup_steps is None:
        return cfg.l0_warmup_steps
    return cfg.inverse_simpson_warmup_steps


@dataclass(frozen=True)
class LossCoefficients:
    lr: float
    lambda_l0: float
    lambda_is: float


def coefficients_at(cfg: SAECfg, step: int, *, n_steps: int | None = None) -> LossCoefficients:
    """Learning rate and penalty weights at ``step``.

    ``n_steps`` is only needed by ``cosine_decay``, the one schedule whose
    shape depends on the run length.
    """
    if cfg.lr_schedule == "warmup_constant":
        lr = warmup_constant(step, cfg.lr, cfg.warmup_steps)
    elif cfg.lr_schedule == "cosine_warmup_constant":
        lr = cosine_warmup_constant(step, cfg.lr, cfg.warmup_steps)
    else:
        if n_steps is None:
            raise ValueError(f"lr_schedule {cfg.lr_schedule!r} needs a fixed n_steps horizon")
        lr = cosine_warmup(step, cfg.lr, cfg.warmup_steps, n_steps)
    return LossCoefficients(
        lr=lr,
        lambda_l0=linear_warmup(step, peak=1.0, warmup_steps=cfg.l0_warmup_steps),
        lambda_is=linear_warmup(
            step, peak=cfg.inverse_simpson_max_scale, warmup_steps=inverse_simpson_warmup_steps(cfg)
        ),
    )


def quad_l0_loss(
    l0_diff: torch.Tensor,
    l0_target: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """Quadratic target-L0 penalty: ``(L0-L0*)² / (2 L0*)``."""
    return scale * (1.0 / (2.0 * max(l0_target, 1))) * (l0_diff - float(l0_target)).pow(2)


@dataclass
class LossTerms:
    """One batch's loss, its components, and the forward outputs."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    mse: torch.Tensor
    l0_penalty: torch.Tensor
    inverse_simpson_penalty: torch.Tensor
    d_is: float
    u_is: float
    x_hat: torch.Tensor
    f: torch.Tensor


def training_loss(
    sae: SparseAutoencoder,
    x: torch.Tensor,
    cfg: SAECfg,
    coefficients: LossCoefficients,
    *,
    target_l0: float,
) -> LossTerms:
    """Reconstruction + λ_L0·quadratic L0 penalty + λ_IS·Inverse-Simpson floor penalty."""
    x_hat, f = sae(x)
    squared_error = (x - x_hat).pow(2)
    if cfg.reconstruction_loss == "vector_sum":
        reconstruction = squared_error.sum(dim=-1).mean()
    else:
        reconstruction = squared_error.mean()
    l0_penalty = quad_l0_loss(sae.l0(x), target_l0)  # type: ignore[attr-defined]

    inverse_simpson_penalty = reconstruction.new_zeros(())
    d_is = u_is = 0.0
    if cfg.inverse_simpson_floor > 0 and cfg.inverse_simpson_max_scale > 0 and isinstance(sae, JumpReLUSAE):
        d_is_tensor, u_is_tensor = batch_inverse_simpson(sae.jump_relu_gate(x), sae.d_sae)
        d_is = float(d_is_tensor.detach().item())
        u_is = float(u_is_tensor.detach().item())
        gap = torch.clamp(
            (cfg.inverse_simpson_floor - u_is_tensor) / max(cfg.inverse_simpson_floor, 1.0e-8), min=0.0
        )
        inverse_simpson_penalty = coefficients.lambda_is * gap.square()

    total = reconstruction + coefficients.lambda_l0 * cfg.l0_penalty_scale * l0_penalty + inverse_simpson_penalty
    return LossTerms(
        total=total,
        reconstruction=reconstruction,
        mse=squared_error.mean(),
        l0_penalty=l0_penalty,
        inverse_simpson_penalty=inverse_simpson_penalty,
        d_is=d_is,
        u_is=u_is,
        x_hat=x_hat,
        f=f,
    )
