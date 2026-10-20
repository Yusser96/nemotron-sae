"""Training-log rows and the firing statistics behind them."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sae_pipeline.eval.metrics import dead_pct_from_fired, firing_concentration
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.objective import GradientStats, LossTerms


@dataclass
class StepLog:
    step: int
    loss: float
    mse: float
    l0_penalty: float
    hard_l0: float
    lr: float
    lambda_l0: float
    dead_pct: float
    ste_window_pct: float
    theta_grad_norm: float
    encoder_grad_norm: float
    reconstruction_loss: float
    feature_use_penalty: float
    active_features: int
    top1_activation_mass_pct: float
    max_firing_frequency: float
    feature_resets: int
    target_l0: float
    d_is: float = 0.0
    u_is: float = 0.0
    inverse_simpson_penalty: float = 0.0
    auxk_penalty: float = 0.0
    # The weighted AuxK term over the reconstruction loss, and why the dead
    # latents (no main-path firing for auxk_dead_tokens) are inactive on this batch.
    auxk_ratio: float = 0.0
    dead_latents: int = 0
    dead_below_threshold: int = 0
    dead_nonpositive: int = 0


@torch.no_grad()
def record_firing(
    f: torch.Tensor,
    *,
    feature_frequency: torch.Tensor,
    ever_fired: torch.Tensor,
    ema_decay: float,
) -> None:
    """Fold one batch of latent activations into the firing statistics."""
    fired = f.detach() > 0
    feature_frequency.mul_(ema_decay).add_(fired.float().mean(dim=0), alpha=1.0 - ema_decay)
    ever_fired |= fired.any(dim=0)


@torch.no_grad()
def step_log(
    sae: JumpReLUSAE,
    x: torch.Tensor,
    *,
    step: int,
    loss: torch.Tensor,
    terms: LossTerms,
    gradient_stats: GradientStats,
    lr: float,
    lambda_l0: float,
    target_l0: float,
    ever_fired: torch.Tensor,
    feature_frequency: torch.Tensor,
    feature_use_penalty: float = 0.0,
    feature_resets: int = 0,
    dead: torch.Tensor | None = None,
) -> StepLog:
    """Assemble one training-log row; costs a few device syncs, so call it sparingly.

    ``dead`` marks AuxK's dead latents. Each is either positive on some token of
    ``x`` yet under its JumpReLU threshold, or not positive on any: AuxK can only
    help the first kind, since the ReLU leaves the second without gradient.
    """
    concentration = firing_concentration(feature_frequency)
    dead_latents = dead_below_threshold = 0
    if dead is not None:
        dead_latents = int(dead.sum())
        dead_below_threshold = int((dead & (sae.gate_input(x) > 0).any(dim=0)).sum())
    return StepLog(
        step=step,
        loss=float(loss.detach()),
        mse=float(terms.mse.detach()),
        l0_penalty=float(terms.l0_penalty.detach()),
        hard_l0=sae.hard_l0(x).item(),
        lr=lr,
        lambda_l0=lambda_l0,
        dead_pct=dead_pct_from_fired(ever_fired),
        ste_window_pct=gradient_stats.ste_window_pct,
        theta_grad_norm=gradient_stats.theta_grad_norm,
        encoder_grad_norm=gradient_stats.encoder_grad_norm,
        reconstruction_loss=float(terms.reconstruction.detach()),
        feature_use_penalty=feature_use_penalty,
        active_features=concentration.active_features,
        top1_activation_mass_pct=concentration.top1_activation_mass_pct,
        max_firing_frequency=concentration.max_firing_frequency,
        feature_resets=feature_resets,
        target_l0=target_l0,
        d_is=terms.d_is,
        u_is=terms.u_is,
        inverse_simpson_penalty=float(terms.inverse_simpson_penalty.detach()),
        auxk_penalty=float(terms.auxk_penalty.detach()),
        auxk_ratio=float(terms.auxk_penalty.detach() / terms.reconstruction.detach()),
        dead_latents=dead_latents,
        dead_below_threshold=dead_below_threshold,
        dead_nonpositive=dead_latents - dead_below_threshold,
    )
