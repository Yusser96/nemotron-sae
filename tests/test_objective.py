import pytest
import torch

from sae_pipeline.config import SAECfg
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.objective import LossCoefficients, coefficients_at, quad_l0_loss, training_loss


def test_coefficients_follow_the_configured_warmups():
    cfg = SAECfg(
        lr=1.0,
        warmup_steps=10,
        lr_schedule="warmup_constant",
        l0_warmup_steps=4,
        inverse_simpson_max_scale=0.5,
        inverse_simpson_warmup_steps=2,
    )

    start = coefficients_at(cfg, 0)
    assert (start.lr, start.lambda_l0, start.lambda_is) == pytest.approx((0.1, 0.0, 0.0))
    middle = coefficients_at(cfg, 2)
    assert (middle.lambda_l0, middle.lambda_is) == pytest.approx((0.5, 0.5))
    done = coefficients_at(cfg, 10)
    assert (done.lr, done.lambda_l0, done.lambda_is) == pytest.approx((1.0, 1.0, 0.5))


def test_inverse_simpson_ramp_defaults_to_the_l0_warmup():
    cfg = SAECfg(l0_warmup_steps=4, inverse_simpson_max_scale=1.0, lr_schedule="warmup_constant")

    assert coefficients_at(cfg, 2).lambda_is == pytest.approx(0.5)


@pytest.mark.parametrize("reconstruction_loss", ["vector_sum", "coordinate_mean"])
def test_training_loss_combines_its_terms(reconstruction_loss):
    torch.manual_seed(0)
    sae = JumpReLUSAE(d_in=6, d_sae=12, bandwidth=0.05)
    x = torch.randn(32, 6)
    cfg = SAECfg(
        reconstruction_loss=reconstruction_loss,
        l0_penalty_scale=2.0,
        inverse_simpson_floor=1.0,
        inverse_simpson_max_scale=3.0,
    )
    coefficients = LossCoefficients(lr=1.0, lambda_l0=0.5, lambda_is=3.0)

    terms = training_loss(sae, x, cfg, coefficients, target_l0=4.0)

    x_hat, f = sae(x)
    squared_error = (x - x_hat).pow(2)
    expected_reconstruction = (
        squared_error.sum(dim=-1).mean() if reconstruction_loss == "vector_sum" else squared_error.mean()
    )
    torch.testing.assert_close(terms.reconstruction, expected_reconstruction)
    torch.testing.assert_close(terms.l0_penalty, quad_l0_loss(sae.l0(x), 4.0))
    assert float(terms.inverse_simpson_penalty.detach()) > 0.0
    assert 0.0 < terms.u_is < 1.0
    torch.testing.assert_close(
        terms.total, terms.reconstruction + 0.5 * 2.0 * terms.l0_penalty + terms.inverse_simpson_penalty
    )
    torch.testing.assert_close(terms.f, f)


def test_training_loss_skips_inverse_simpson_without_a_floor():
    sae = JumpReLUSAE(d_in=4, d_sae=8)
    terms = training_loss(
        sae, torch.randn(8, 4), SAECfg(), LossCoefficients(lr=1.0, lambda_l0=1.0, lambda_is=1.0), target_l0=2.0
    )

    assert float(terms.inverse_simpson_penalty) == 0.0
    assert terms.d_is == terms.u_is == 0.0


def test_cosine_decay_needs_a_run_length():
    cfg = SAECfg(lr=1.0, warmup_steps=0, lr_schedule="cosine_decay")

    with pytest.raises(ValueError, match="n_steps"):
        coefficients_at(cfg, 5)
    assert coefficients_at(cfg, 10, n_steps=10).lr == pytest.approx(0.1)
