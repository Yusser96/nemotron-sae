"""Compare activations from two extraction backends on the same tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActivationAgreement:
    n_tokens: int
    cosine_mean: float
    cosine_p01: float
    cosine_min: float
    max_abs_error: float
    relative_l2: float  # ||candidate - reference|| / ||reference|| over all tokens

    def passes(self, *, min_cosine_mean: float, min_cosine_p01: float, max_relative_l2: float) -> bool:
        return (
            self.cosine_mean >= min_cosine_mean
            and self.cosine_p01 >= min_cosine_p01
            and self.relative_l2 <= max_relative_l2
        )


def compare_activations(reference: torch.Tensor, candidate: torch.Tensor) -> ActivationAgreement:
    """Per-token agreement of two ``(tokens, d)`` activation matrices, in float64."""
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: reference {tuple(reference.shape)} vs candidate {tuple(candidate.shape)}")
    if reference.ndim != 2 or reference.shape[0] == 0 or reference.shape[1] == 0:
        raise ValueError("Expected non-empty (tokens, width) activation matrices")
    ref = reference.to(torch.float64)
    cand = candidate.to(torch.float64)
    cosine = torch.nn.functional.cosine_similarity(ref, cand, dim=-1, eps=1e-12)
    # Identical zero vectors agree; a zero/nonzero pair retains cosine zero.
    both_zero = (ref == 0).all(dim=-1) & (cand == 0).all(dim=-1)
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    return ActivationAgreement(
        n_tokens=ref.shape[0],
        cosine_mean=float(cosine.mean()),
        cosine_p01=float(torch.quantile(cosine, 0.01)),
        cosine_min=float(cosine.min()),
        max_abs_error=float((cand - ref).abs().max()),
        relative_l2=float((cand - ref).norm() / ref.norm().clamp_min(1e-12)),
    )
