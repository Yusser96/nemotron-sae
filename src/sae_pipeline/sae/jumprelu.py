"""JumpReLU sparse autoencoder with a log-parameterised positive threshold.

Activation:
    f(x) = z ⊙ H(z − θ)        with z = ReLU(W_enc x + b_enc), θ = exp(log_theta)
Reconstruction:
    x̂ = W_dec f + b_dec
Loss (quadratic L0 penalty around target L0*):
    L = ‖x − x̂‖² + λ · (2 / L0*) · (‖f‖₀ − L0*)²

Both the Heaviside H and the L0 norm are non-differentiable; a
rectangular-kernel straight-through estimator (bandwidth ε) is applied to the
log-threshold gradient only (Rajamanoharan et al. 2024b; Gemma Scope 2 §2.3).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sae_pipeline.sae.base import SparseAutoencoder, init_encoder_from_decoder


class _JumpReLU(torch.autograd.Function):
    """Fused JumpReLU gate with a rectangular-kernel straight-through gradient.

    The encoder gradient is the gate itself. The rectangular kernel is used
    only for the log-threshold gradient, so it cannot leak into the encoder
    path through a gate-multiply decomposition.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, log_theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
        theta = log_theta.exp()
        active = z > theta
        ctx.save_for_backward(z, theta, active)
        ctx.bandwidth = bandwidth
        return z * active.to(z.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        z, theta, active = ctx.saved_tensors
        eps = ctx.bandwidth
        in_window = (z - theta).abs() < (eps / 2.0)
        grad_z = grad_out * active.to(grad_out.dtype)
        grad_log_theta = -(
            grad_out * (theta / eps) * in_window.to(grad_out.dtype)
        ).sum(dim=0)
        return grad_z, grad_log_theta, None


class _L0(torch.autograd.Function):
    """Mean active-latent count with an STE on log-thresholds only."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, log_theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
        theta = log_theta.exp()
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        return (z > theta).to(z.dtype).sum(dim=-1).mean()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        z, theta = ctx.saved_tensors
        eps = ctx.bandwidth
        in_window = (z - theta).abs() < (eps / 2.0)
        grad_log_theta = -(theta * in_window.to(z.dtype)).sum(dim=0) / (eps * z.shape[0])
        return None, grad_out * grad_log_theta, None


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
        if threshold_init <= 0:
            raise ValueError("threshold_init must be positive")
        self.d_in = d_in
        self.d_sae = d_sae
        self.bandwidth = bandwidth
        self.pre_encoder_bias = pre_encoder_bias

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # Positivity is structural and threshold updates are multiplicative.
        self.log_theta = nn.Parameter(torch.full((d_sae,), math.log(threshold_init)))

        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))
        init_encoder_from_decoder(self.W_dec, self.W_enc)

    @property
    def theta(self) -> torch.Tensor:
        """Positive JumpReLU threshold, exposed for metrics and diagnostics."""
        return self.log_theta.exp()

    def _decoder_weight(self) -> nn.Parameter:
        return self.W_dec

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_encoder_bias:
            x = x - self.b_dec
        return x @ self.W_enc + self.b_enc

    def gate_input(self, x: torch.Tensor) -> torch.Tensor:
        """Non-negative pre-gate activation required by the JumpReLU recipe."""
        return torch.relu(self.encode_pre(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return _JumpReLU.apply(self.gate_input(x), self.log_theta, self.bandwidth)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec.T + self.b_dec

    def l0(self, x: torch.Tensor) -> torch.Tensor:
        return _L0.apply(self.gate_input(x), self.log_theta, self.bandwidth)

    @torch.no_grad()
    def hard_l0(self, x: torch.Tensor) -> torch.Tensor:
        return (self.gate_input(x) > self.theta).to(x.dtype).sum(dim=-1).mean()

    @torch.no_grad()
    def unfold_raw_export(self, input_scale: float) -> None:
        """Convert exported raw-activation weights to normalised training coordinates."""
        if input_scale <= 0:
            raise ValueError("input_scale must be positive")
        raw_w_enc = self.W_enc.detach().clone()
        raw_b_dec = self.b_dec.detach().clone()
        self.b_enc.add_(raw_b_dec @ raw_w_enc)
        self.W_enc.div_(input_scale)
        self.W_dec.mul_(input_scale)
        self.b_dec.mul_(input_scale)
        self.pre_encoder_bias = True
        self.renormalize_decoder_preserving_function()

    @torch.no_grad()
    def renormalize_decoder_preserving_function(self) -> None:
        """Make decoder columns unit norm while preserving SAE reconstruction."""
        norms = self.W_dec.norm(dim=0).clamp_min(1e-8)
        self.W_dec.div_(norms.unsqueeze(0))
        self.W_enc.mul_(norms.unsqueeze(0))
        self.b_enc.mul_(norms)
        self.log_theta.add_(norms.log())

    @torch.no_grad()
    def raw_export_state_dict(
        self,
        input_scale: float,
        input_center: torch.Tensor | None = None,
        output_center: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fold normalised training parameters back into raw-activation weights."""
        if input_scale <= 0:
            raise ValueError("input_scale must be positive")
        raw_w_enc = self.W_enc * input_scale
        input_center = (
            torch.zeros_like(self.b_dec) if input_center is None else input_center
        )
        output_center = (
            torch.zeros_like(self.b_dec) if output_center is None else output_center
        )
        if input_center.shape != self.b_dec.shape or output_center.shape != self.b_dec.shape:
            raise ValueError("input_center and output_center must have shape (d_in,)")
        raw_b_dec = (self.b_dec + output_center) / input_scale
        raw_b_enc = self.b_enc - (self.b_dec + input_center) @ self.W_enc
        return {
            "W_enc": raw_w_enc.detach().cpu().contiguous(),
            "W_dec": (self.W_dec / input_scale).T.detach().cpu().contiguous(),
            "b_enc": raw_b_enc.detach().cpu().contiguous(),
            "b_dec": raw_b_dec.detach().cpu().contiguous(),
            "threshold": self.theta.detach().cpu().contiguous(),
        }
