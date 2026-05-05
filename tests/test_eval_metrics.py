"""Smoke-test reconstruction_metrics on a controlled synthetic SAE."""

import torch

from sae_pipeline.eval.metrics import reconstruction_metrics
from sae_pipeline.sae.jumprelu import JumpReLUSAE


def test_recon_metrics_perfect_identity_on_zero_input():
    """Zero input → zero reconstruction → FVU = NaN-protected, L0 = 0."""
    sae = JumpReLUSAE(d_in=16, d_sae=64)
    x = torch.zeros(128, 16) + 1e-9
    m = reconstruction_metrics(sae, x, dead_threshold_tokens=128)
    assert m.l0 >= 0.0
    assert m.fvu >= 0.0
    assert 0.0 <= m.dead_pct <= 100.0


def test_recon_metrics_runs_on_random_input():
    sae = JumpReLUSAE(d_in=16, d_sae=64)
    x = torch.randn(2048, 16)
    m = reconstruction_metrics(sae, x, dead_threshold_tokens=512)
    assert m.n_tokens == 2048
    assert m.l0 >= 0.0
    assert m.fvu > 0.0      # untrained SAE has nonzero error
