"""Reconstruction and feature-use metrics on SAEs with known outputs."""

import numpy as np
import torch

from sae_pipeline.eval.metrics import (
    reconstruction_metrics,
    streaming_reconstruction_metrics,
)
from sae_pipeline.sae.jumprelu import JumpReLUSAE


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
    assert m.dead_pct == 100.0
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
    assert np.array_equal(stream_arrays.l0_per_token, dense_arrays.l0_per_token)


def test_fvu_handles_small_variance_around_a_large_mean():
    x = torch.tensor([[9999.0], [10000.0], [10001.0], [10002.0]])
    sae = JumpReLUSAE(d_in=1, d_sae=2)
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.b_dec.fill_(10000.5)
    metrics = streaming_reconstruction_metrics(sae, (x[:1], x[1:3], x[3:]))
    assert abs(metrics.fvu - 1.0) < 1e-10
    assert metrics.support_n_tokens == 4


def test_support_window_can_exceed_reconstruction_window():
    sae = JumpReLUSAE(d_in=2, d_sae=2)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(2)); sae.W_dec.copy_(torch.eye(2))
        sae.b_enc.zero_(); sae.b_dec.zero_(); sae.log_theta.fill_(-5)
    x = torch.tensor([[1., 0.], [2., 0.], [0., 1.], [0., 2.], [0., 3.]])
    metrics, arrays = streaming_reconstruction_metrics(sae, x.split(3), max_tokens=2,
                                                      dead_threshold_tokens=4, return_arrays=True)
    reference = reconstruction_metrics(sae, x[:2])
    assert metrics.n_tokens == 2 and metrics.support_n_tokens == 4
    assert metrics.fvu == reference.fvu and metrics.l0 == reference.l0
    assert metrics.dead_pct == 0 and reference.dead_pct == 50
    assert len(arrays.recon_err_per_token) == 2
    np.testing.assert_allclose(arrays.firing_frequency, [0.5, 0.5])
