import pytest

from sae_pipeline.config import SAECfg
from sae_pipeline.sae.objective import coefficients_at


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


def test_cosine_decay_needs_a_run_length():
    cfg = SAECfg(lr=1.0, warmup_steps=0, lr_schedule="cosine_decay")

    with pytest.raises(ValueError, match="n_steps"):
        coefficients_at(cfg, 5)
    assert coefficients_at(cfg, 10, n_steps=10).lr == pytest.approx(0.1)
