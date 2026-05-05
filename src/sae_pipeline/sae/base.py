"""Abstract SAE interface common to JumpReLU / TopK / BatchTopK / Matryoshka.

All variants share the encoder/decoder shape and the conventions:
- decoder columns unit-norm after every step
- pre-encoder bias subtracted from input
- W_enc initialized as W_dec.T at start, untied afterwards
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module, ABC):
    d_in: int
    d_sae: int

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Return W_enc x + b_enc (no activation function)."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return f(x) = activation_fn(W_enc x + b_enc)."""

    @abstractmethod
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Return x̂ = W_dec f + b_dec."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    @torch.no_grad()
    def renormalize_decoder(self) -> None:
        """Rescale W_dec columns to unit norm and project gradient direction
        component out of the decoder gradient before the optimizer step."""
        w = self._decoder_weight()
        norms = w.norm(dim=0, keepdim=True).clamp_min(1e-8)
        w.div_(norms)

    @torch.no_grad()
    def project_decoder_grad(self) -> None:
        """Project the decoder gradient onto the tangent space of the unit sphere
        (i.e. orthogonal to each column). Per Bricken et al. and Gemma Scope 2."""
        w = self._decoder_weight()
        if w.grad is None:
            return
        # For each column, subtract projection onto the column itself.
        # grad_col -= (grad_col · col) * col
        dot = (w.grad * w).sum(dim=0, keepdim=True)
        w.grad.sub_(dot * w)

    def _decoder_weight(self) -> nn.Parameter:
        raise NotImplementedError


def init_encoder_from_decoder(W_dec: nn.Parameter, W_enc: nn.Parameter) -> None:
    """Tied initialization: W_enc and W_dec are stored with identical shape
    (d_in, d_sae) where each column is a dictionary direction; encoder shares the
    same columns at init then trains untied.
    """
    if W_enc.shape != W_dec.shape:
        raise ValueError(
            f"Tied init expects matching shapes; got W_enc {tuple(W_enc.shape)} "
            f"vs W_dec {tuple(W_dec.shape)}"
        )
    with torch.no_grad():
        W_enc.copy_(W_dec)
