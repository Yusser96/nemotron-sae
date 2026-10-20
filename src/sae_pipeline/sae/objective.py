"""The JumpReLU SAE training objective and update, shared by every trainer.

Schedules map a global step to the coefficients of the loss (learning rate,
L0-penalty weight, Inverse-Simpson weight); :func:`training_loss` evaluates
the loss itself for one activation batch, and :func:`optimizer_step` applies
one constrained Adam update from it. The same loss function scores held-out
data, so early stopping can track exactly what training optimises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sae_pipeline.config import SAECfg
from sae_pipeline.sae.jumprelu import JumpReLUSAE, inverse_simpson_from_mass


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


def auxk_loss(
    sae: JumpReLUSAE,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    dead: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Normalised error of reconstructing the main residual from the strongest dead latents.

    The candidates are the pre-gate activations, so the JumpReLU thresholds are
    bypassed (and get no gradient from this term); the decoder is shared and
    its bias is not re-added. The residual target is detached, so only the
    dead latents' encoder and decoder learn from it. The error is divided by
    the residual's variance, keeping the term's strength independent of how
    well the main path currently reconstructs. Zero while no latent is dead.
    """
    residual = (x - x_hat).detach()
    candidates = sae.gate_input(x) * dead
    values, indices = candidates.topk(min(k, sae.d_sae), dim=-1)
    residual_hat = torch.zeros_like(candidates).scatter(-1, indices, values) @ sae.W_dec.T
    error = (residual - residual_hat).pow(2).mean()
    variance = (residual - residual.mean(dim=0)).pow(2).mean()
    return dead.any() * error / variance.clamp_min(1.0e-8)


@dataclass
class LossTerms:
    """One batch's loss, its components, and the forward outputs."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    mse: torch.Tensor
    l0_penalty: torch.Tensor
    inverse_simpson_penalty: torch.Tensor
    auxk_penalty: torch.Tensor
    d_is: float
    u_is: float
    x_hat: torch.Tensor
    f: torch.Tensor


def training_loss(
    sae: JumpReLUSAE,
    x: torch.Tensor,
    cfg: SAECfg,
    coefficients: LossCoefficients,
    *,
    target_l0: float,
    usage: torch.Tensor | None = None,
    tokens_since_fired: torch.Tensor | None = None,
) -> LossTerms:
    """Reconstruction + λ_L0·quadratic L0 penalty + λ_IS·Inverse-Simpson floor penalty + α·AuxK.

    ``usage`` holds each feature's firing rate averaged over the last
    ``inverse_simpson_window_steps`` batches; it is updated in place and the
    floor is measured on it. Without it, each batch is measured on its own.

    ``tokens_since_fired`` counts the tokens since each latent last fired; it
    is updated in place and defines the dead latents AuxK trains. Without it
    (held-out scoring, which has no running count) AuxK is left out: it is an
    optimisation aid, and its changing dead set would make held-out objectives
    incomparable across evaluations.
    """
    x_hat, f = sae(x)
    squared_error = (x - x_hat).pow(2)
    if cfg.reconstruction_loss == "vector_sum":
        reconstruction = squared_error.sum(dim=-1).mean()
    else:
        reconstruction = squared_error.mean()
    l0_penalty = quad_l0_loss(sae.l0(x), target_l0)

    inverse_simpson_penalty = reconstruction.new_zeros(())
    d_is = u_is = 0.0
    if cfg.inverse_simpson_floor > 0 and cfg.inverse_simpson_max_scale > 0:
        firing = sae.jump_relu_gate(x).mean(dim=0)
        if usage is not None:
            with torch.no_grad():
                usage.lerp_(firing, 1.0 / cfg.inverse_simpson_window_steps)
            # The window's rates, with this batch's gradient.
            firing = usage + (firing - firing.detach())
        d_is_tensor, u_is_tensor, _ = inverse_simpson_from_mass(firing, sae.d_sae)
        d_is = float(d_is_tensor.detach().item())
        u_is = float(u_is_tensor.detach().item())
        gap = torch.clamp(
            (cfg.inverse_simpson_floor - u_is_tensor) / max(cfg.inverse_simpson_floor, 1.0e-8), min=0.0
        )
        inverse_simpson_penalty = coefficients.lambda_is * gap.square()

    auxk_penalty = reconstruction.new_zeros(())
    if cfg.auxk_scale > 0 and tokens_since_fired is not None:
        # A fraction of the current reconstruction loss, whatever its reduction or input scale.
        auxk_penalty = (
            cfg.auxk_scale
            * reconstruction.detach()
            * auxk_loss(sae, x, x_hat, tokens_since_fired >= cfg.auxk_dead_tokens, cfg.auxk_k)
        )
        with torch.no_grad():
            tokens_since_fired.add_(x.shape[0]).masked_fill_((f > 0).any(dim=0), 0)

    total = (
        reconstruction
        + coefficients.lambda_l0 * cfg.l0_penalty_scale * l0_penalty
        + inverse_simpson_penalty
        + auxk_penalty
    )
    return LossTerms(
        total=total,
        reconstruction=reconstruction,
        mse=squared_error.mean(),
        l0_penalty=l0_penalty,
        inverse_simpson_penalty=inverse_simpson_penalty,
        auxk_penalty=auxk_penalty,
        d_is=d_is,
        u_is=u_is,
        x_hat=x_hat,
        f=f,
    )


@dataclass
class GradientStats:
    ste_window_pct: float
    theta_grad_norm: float
    encoder_grad_norm: float


def optimizer_step(
    sae: JumpReLUSAE,
    optim: torch.optim.Optimizer,
    loss: torch.Tensor,
    x: torch.Tensor,
    cfg: SAECfg,
    *,
    step: int,
    lr: float,
) -> GradientStats:
    """Backpropagate ``loss`` and apply one Adam step with the decoder constraints."""
    for group in optim.param_groups:
        group["lr"] = lr
    optim.zero_grad(set_to_none=True)
    loss.backward()
    ste_window_pct = theta_grad_norm = encoder_grad_norm = 0.0
    # These values are consumed only by periodic StepLog rows. Avoid an extra
    # encoder GEMM and three host synchronizations on every unlogged update.
    if step % cfg.log_every == 0:
        with torch.no_grad():
            z = sae.gate_input(x)
            ste_window_pct = 100.0 * (
                (z - sae.theta).abs() < (cfg.bandwidth / 2.0)
            ).float().mean().item()
            theta_grad = getattr(sae, "log_theta").grad
            encoder_grad = getattr(sae, "W_enc").grad
            theta_grad_norm = 0.0 if theta_grad is None else theta_grad.norm().item()
            encoder_grad_norm = 0.0 if encoder_grad is None else encoder_grad.norm().item()
    if cfg.gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.gradient_clip_norm, error_if_nonfinite=True)
    if step <= cfg.decoder_freeze_steps:
        sae.W_dec.grad = None
    if cfg.decoder_unit_norm and step > cfg.decoder_freeze_steps:
        sae.project_decoder_grad()
    optim.step()
    if cfg.decoder_unit_norm:
        sae.renormalize_decoder()
    return GradientStats(
        ste_window_pct=ste_window_pct,
        theta_grad_norm=theta_grad_norm,
        encoder_grad_norm=encoder_grad_norm,
    )
