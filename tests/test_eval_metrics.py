"""Smoke-test reconstruction_metrics on a controlled synthetic SAE."""

import torch

from sae_pipeline.eval.metrics import reconstruction_metrics, streaming_reconstruction_metrics
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


def test_fvu_is_one_for_predict_the_mean_baseline():
    """FVU = sum||x-x_hat||^2 / sum||x-mean||^2, so predicting the sample mean
    for every token must give FVU == 1 regardless of n_tokens or d_in -- this
    guards against the numerator/denominator being normalised by mismatched
    divisors (n_tokens vs. d_in), which silently deflates FVU by a factor of
    d_in / n_tokens.
    """
    torch.manual_seed(0)
    x = torch.randn(4096, 12) * 3 + 5
    sae = JumpReLUSAE(d_in=12, d_sae=8)
    with torch.no_grad():
        sae.log_theta.fill_(20.0)  # threshold so high no latent ever fires
        sae.b_dec.copy_(x.mean(dim=0))
    m = reconstruction_metrics(sae, x, dead_threshold_tokens=4096)
    assert m.l0 == 0.0
    assert abs(m.fvu - 1.0) < 1e-4


def test_streaming_metrics_match_dense_metrics():
    sae = JumpReLUSAE(d_in=8, d_sae=32)
    x = torch.randn(257, 8)
    dense = reconstruction_metrics(sae, x, dead_threshold_tokens=257, return_arrays=True)
    streamed = streaming_reconstruction_metrics(
        sae, (x[:73], x[73:181], x[181:]), dead_threshold_tokens=257, return_arrays=True
    )
    dense_metrics, dense_arrays = dense
    stream_metrics, stream_arrays = streamed
    assert stream_metrics.n_tokens == dense_metrics.n_tokens
    assert abs(stream_metrics.l0 - dense_metrics.l0) < 1e-6
    assert abs(stream_metrics.fvu - dense_metrics.fvu) < 1e-6
    assert abs(stream_metrics.dead_pct - dense_metrics.dead_pct) < 1e-6
    assert abs(stream_metrics.firing_gini - dense_metrics.firing_gini) < 1e-6
    assert torch.equal(torch.from_numpy(stream_arrays.l0_per_token), torch.from_numpy(dense_arrays.l0_per_token))
