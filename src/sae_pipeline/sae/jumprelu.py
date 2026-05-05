"""JumpReLU SAE — Gemma Scope 2 default.

Activation:
    f(x) = JumpReLU_θ(z) = z ⊙ H(z − θ)        with z = W_enc x + b_enc
Reconstruction:
    x̂ = W_dec f + b_dec
Loss (quadratic L0 penalty around target L0*):
    L = ‖x − x̂‖² + λ · (2 / L0*) · (‖f‖₀ − L0*)²

Both the Heaviside H and the L0 norm are non-differentiable; we use straight-through
estimators with a kernel bandwidth ε (Rajamanoharan et al. 2024b; Gemma Scope 2 §2.3).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sae_pipeline.sae.base import SparseAutoencoder, init_encoder_from_decoder


class _JumpReLUSTE(torch.autograd.Function):
    """Heaviside H(z − θ) with rectangular-kernel STE for both z and θ.

    Forward returns the gate (1 if z > θ else 0).
    Backward distributes (1 / ε) inside the |z - θ| < ε/2 window.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        return (z > theta).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (z, theta) = ctx.saved_tensors
        eps = ctx.bandwidth
        # Rectangular kernel of width eps centered at z=theta.
        in_window = (z - theta).abs() < (eps / 2.0)
        kernel = in_window.to(z.dtype) / eps
        grad_z = grad_out * kernel
        # ∂H/∂θ = -∂H/∂z (since H depends on z-θ); contract over batch.
        grad_theta = -(grad_out * kernel).sum(dim=0)
        return grad_z, grad_theta, None


def jump_relu(z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
    gate = _JumpReLUSTE.apply(z, theta, bandwidth)
    return z * gate


class _L0STE(torch.autograd.Function):
    """L0 norm with rectangular-kernel STE on the threshold parameter.

    Forward returns ‖f‖₀ counted per-sample then mean-reduced.
    Backward routes (-1 / ε) into θ inside the |z - θ| < ε/2 window.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        active = (z > theta).to(z.dtype)
        return active.sum(dim=-1).mean()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (z, theta) = ctx.saved_tensors
        eps = ctx.bandwidth
        in_window = (z - theta).abs() < (eps / 2.0)
        # ∂L0/∂θ = -1/ε inside the window; mean over batch dim.
        n_batch = z.shape[0]
        grad_theta = -in_window.to(z.dtype).sum(dim=0) / (eps * n_batch)
        return None, grad_out * grad_theta, None


def l0_norm_diff(z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return _L0STE.apply(z, theta, bandwidth)


class JumpReLUSAE(SparseAutoencoder):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        bandwidth: float = 0.001,
        threshold_init: float = 0.001,
        pre_encoder_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.bandwidth = bandwidth
        self.pre_encoder_bias = pre_encoder_bias

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # Per-latent learnable JumpReLU threshold; init small positive.
        self.theta = nn.Parameter(torch.full((d_sae,), threshold_init))

        # He-uniform W_dec, then renormalize columns to unit norm.
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))
        # Tie at init, untie afterwards.
        init_encoder_from_decoder(self.W_dec, self.W_enc)

    def _decoder_weight(self) -> nn.Parameter:
        return self.W_dec

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_encoder_bias:
            x = x - self.b_dec
        return x @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_pre(x)
        return jump_relu(z, self.theta, self.bandwidth)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec.T + self.b_dec

    def l0(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable L0 (via STE on theta)."""
        z = self.encode_pre(x)
        return l0_norm_diff(z, self.theta, self.bandwidth)

    @torch.no_grad()
    def hard_l0(self, x: torch.Tensor) -> torch.Tensor:
        """Non-differentiable, true L0 used for monitoring."""
        z = self.encode_pre(x)
        return ((z > self.theta).to(z.dtype).sum(dim=-1)).mean()
