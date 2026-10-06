"""The JumpReLU SAE training objective, shared by every trainer.

Schedules map a global step to the coefficients of the loss (learning rate,
L0-penalty weight, Inverse-Simpson weight).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sae_pipeline.config import SAECfg


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
