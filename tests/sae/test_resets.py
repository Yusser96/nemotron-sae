import pytest
import torch

from sae_pipeline.config import SAECfg
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.resets import resample_underused_features, reset_l0_allowance, residual_reset_due


@pytest.mark.parametrize("unused,pool,expected", [(1, 8, 1), (4, 2, 2)])
def test_reset_handles_small_candidate_and_direction_pools(unused, pool, expected):
    sae = JumpReLUSAE(3, 4)
    x = torch.randn(8, 3)
    frequency = torch.ones(4)
    frequency[:unused] = 0
    count = resample_underused_features(
        sae, torch.optim.Adam(sae.parameters()), x, torch.zeros_like(x),
        frequency, torch.ones(4, dtype=torch.bool),
        SAECfg(residual_reset_pool_size=pool, residual_reset_max_features=4),
    )
    assert count == expected
    assert torch.isfinite(sae.theta).all()


def test_explicit_reset_frequency_controls_calibration():
    torch.manual_seed(9)
    sae = JumpReLUSAE(3, 4)
    x = torch.randn(1000, 3)
    resample_underused_features(
        sae, torch.optim.Adam(sae.parameters()), x, torch.zeros_like(x),
        torch.zeros(4), torch.zeros(4, dtype=torch.bool),
        SAECfg(residual_reset_target_frequency=0.1, bandwidth=1e-6),
    )
    assert torch.allclose((sae.encode(x) > 0).float().mean(0), torch.full((4,), 0.1), atol=0.002)


def test_unused_frequency_setting_is_rejected():
    with pytest.raises(ValueError, match="was unused"):
        SAECfg(dead_freq_threshold=0.2)


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
    count = resample_underused_features(
        sae, optim, x, recon.detach(), frequency, ever_fired,
        SAECfg(residual_reset_max_features=4, residual_reset_pool_size=16),
    )
    assert count == 4
    assert not torch.equal(old_decoder, sae.W_dec)
    assert int((~ever_fired).sum()) == count
    assert torch.all(frequency == 0)
    zero_columns = optim.state[sae.W_dec]["exp_avg"].abs().sum(dim=0) == 0
    assert int(zero_columns.sum()) >= count


def test_resets_come_on_schedule_and_their_l0_allowance_anneals_away():
    cfg = SAECfg(
        feature_use_strategy="residual_reset",
        residual_reset_start_step=10,
        residual_reset_every_steps=5,
        residual_reset_delta_l0=2.0,
        residual_reset_anneal_steps=4,
    )

    assert [step for step in range(25) if residual_reset_due(cfg, step)] == [10, 15, 20]
    assert not residual_reset_due(SAECfg(), 10)
    assert [reset_l0_allowance(cfg, step, 10) for step in (10, 12, 14)] == [2.0, 1.0, 0.0]
    assert reset_l0_allowance(cfg, 12, None) == 0.0
