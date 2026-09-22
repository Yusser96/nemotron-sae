"""Sanity-check JumpReLU SAE: shapes, gradients, sparsity, decoder norm invariance."""

import math

import torch

from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.config import SAECfg
from sae_pipeline.sae.train import (
    _resample_underused_features,
    _scheduled_l0_target,
    quad_l0_loss,
)


def test_jumprelu_forward_shape():
    d_in, d_sae = 32, 256
    sae = JumpReLUSAE(d_in=d_in, d_sae=d_sae)
    x = torch.randn(64, d_in)
    x_hat, f = sae(x)
    assert x_hat.shape == (64, d_in)
    assert f.shape == (64, d_sae)


def test_jumprelu_decoder_unit_norm_after_renorm():
    sae = JumpReLUSAE(d_in=16, d_sae=128)
    sae.W_dec.data.mul_(3.7)  # break unit norm
    sae.renormalize_decoder()
    norms = sae.W_dec.data.norm(dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_jumprelu_l0_decreases_with_higher_threshold():
    sae = JumpReLUSAE(d_in=16, d_sae=128)
    x = torch.randn(256, 16)
    sae.log_theta.data.fill_(-8.0)   # almost everything fires
    l0_low = sae.hard_l0(x).item()
    sae.log_theta.data.fill_(math.log(5.0))    # almost nothing fires
    l0_high = sae.hard_l0(x).item()
    assert l0_low > l0_high


def test_jumprelu_backward_runs():
    """One step of gradient descent should reduce reconstruction loss on a fixed batch."""
    sae = JumpReLUSAE(d_in=16, d_sae=64, bandwidth=0.01)
    optim = torch.optim.Adam(sae.parameters(), lr=1e-2, betas=(0.0, 0.999))
    x = torch.randn(64, 16)

    loss_history = []
    for _ in range(50):
        x_hat, _ = sae(x)
        loss = (x - x_hat).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        sae.project_decoder_grad()
        optim.step()
        sae.renormalize_decoder()
        loss_history.append(loss.item())

    assert loss_history[-1] < loss_history[0], (loss_history[0], loss_history[-1])


def test_vector_sum_reconstruction_is_coordinate_mean_times_width():
    x = torch.randn(8, 32)
    coordinate_mean = (x.square()).mean()
    vector_sum = x.square().sum(dim=-1).mean()
    torch.testing.assert_close(vector_sum, coordinate_mean * x.shape[-1])


def test_l0_penalty_scale_is_explicit():
    l0 = torch.tensor(12.0)
    base = quad_l0_loss(l0, 10)
    torch.testing.assert_close(quad_l0_loss(l0, 10, scale=4.0), base * 4.0)


def test_gradual_l0_schedule_reaches_final_target():
    cfg = SAECfg(l0_target_start=16.7, l0_target_warmup_steps=15_000)
    assert _scheduled_l0_target(0, 10, cfg) == 16.7
    assert _scheduled_l0_target(15_000, 10, cfg) == 10.0
    assert 10.0 < _scheduled_l0_target(7_500, 10, cfg) < 16.7


def test_centred_coordinates_preserve_initial_function():
    torch.manual_seed(0)
    sae = JumpReLUSAE(d_in=5, d_sae=11)
    x = torch.randn(32, 5)
    centre = torch.randn(5)
    before, features_before = sae(x)
    with torch.no_grad():
        sae.b_dec.sub_(centre)
    after_centred, features_after = sae(x - centre)
    torch.testing.assert_close(features_before, features_after, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(before - centre, after_centred, atol=1e-5, rtol=1e-5)


def test_residual_reset_replaces_columns_and_clears_adam_moments():
    torch.manual_seed(0)
    sae = JumpReLUSAE(d_in=8, d_sae=16, bandwidth=0.01)
    optim = torch.optim.Adam(sae.parameters(), lr=1e-3)
    x = torch.randn(64, 8)
    recon, _ = sae(x)
    recon.square().mean().backward()
    optim.step()
    old_decoder = sae.W_dec.detach().clone()
    frequency = torch.zeros(16)
    ever_fired = torch.ones(16, dtype=torch.bool)
    count = _resample_underused_features(
        sae, optim, x, recon.detach(), frequency, ever_fired,
        SAECfg(residual_reset_max_features=4, residual_reset_pool_size=16),
    )
    assert count == 4
    assert not torch.equal(old_decoder, sae.W_dec)
    assert int((~ever_fired).sum()) == count
    assert torch.all(frequency == 0)
    zero_columns = optim.state[sae.W_dec]["exp_avg"].abs().sum(dim=0) == 0
    assert int(zero_columns.sum()) >= count


def test_jumprelu_l0_grad_flows_to_theta():
    sae = JumpReLUSAE(d_in=16, d_sae=64, bandwidth=0.05)
    x = torch.randn(64, 16)
    # Encourage a lot of latents to be near the threshold edge so STE has signal.
    sae.log_theta.data.fill_(math.log(0.01))
    l0 = sae.l0(x)
    l0.backward()
    assert sae.log_theta.grad is not None
    assert sae.log_theta.grad.abs().sum().item() > 0.0


def test_jumprelu_pre_encoder_bias_is_configurable():
    x = torch.ones(4, 3)
    with_bias = JumpReLUSAE(d_in=3, d_sae=5, pre_encoder_bias=True)
    without_bias = JumpReLUSAE(d_in=3, d_sae=5, pre_encoder_bias=False)
    without_bias.load_state_dict(with_bias.state_dict())
    with_bias.b_dec.data.fill_(1.0)
    torch.testing.assert_close(
        with_bias.encode_pre(x), (x - with_bias.b_dec) @ with_bias.W_enc + with_bias.b_enc
    )
    torch.testing.assert_close(
        without_bias.encode_pre(x), x @ without_bias.W_enc + without_bias.b_enc
    )


def test_normalised_coordinates_and_raw_export_are_functionally_equivalent():
    raw = JumpReLUSAE(d_in=3, d_sae=7, pre_encoder_bias=False)
    raw.b_dec.data.normal_()
    raw.b_enc.data.normal_()
    raw.log_theta.data.fill_(math.log(0.05))
    x_raw = torch.randn(11, 3)
    raw_recon, raw_features = raw(x_raw)

    canonical = JumpReLUSAE(d_in=3, d_sae=7, pre_encoder_bias=False)
    canonical.load_state_dict(raw.state_dict())
    scale = 0.2
    canonical.unfold_raw_export(scale)
    canonical_recon, canonical_features = canonical(x_raw * scale)

    torch.testing.assert_close(canonical_features > 0, raw_features > 0)
    torch.testing.assert_close(canonical_recon, raw_recon * scale, atol=1e-5, rtol=1e-5)
