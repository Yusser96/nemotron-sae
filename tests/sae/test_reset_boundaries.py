import pytest
import torch

from sae_pipeline.config import SAECfg
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.train import resample_underused_features


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
