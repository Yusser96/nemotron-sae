import pytest
import torch

from sae_pipeline.config import SAECfg
from sae_pipeline.sae.jumprelu import JumpReLUSAE, inverse_simpson_from_mass
from sae_pipeline.sae.objective import (
    LossCoefficients,
    coefficients_at,
    cosine_warmup_constant,
    optimizer_step,
    quad_l0_loss,
    scheduled_l0_target,
    training_loss,
)


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


def test_a_usage_window_sets_the_floor_and_keeps_the_batch_gradient():
    torch.manual_seed(0)
    sae = JumpReLUSAE(d_in=6, d_sae=12, bandwidth=0.05)
    x = torch.randn(32, 6)
    coefficients = LossCoefficients(lr=1.0, lambda_l0=0.0, lambda_is=3.0)
    batch_rates = sae.jump_relu_gate(x).mean(dim=0).detach()
    previous = torch.rand(12)
    usage = previous.clone()

    windowed = training_loss(
        sae,
        x,
        SAECfg(inverse_simpson_floor=1.0, inverse_simpson_max_scale=3.0, inverse_simpson_window_steps=4),
        coefficients,
        target_l0=4.0,
        usage=usage,
    )
    windowed.inverse_simpson_penalty.backward()

    torch.testing.assert_close(usage, previous + (batch_rates - previous) / 4)
    assert windowed.d_is == pytest.approx(float(inverse_simpson_from_mass(usage, 12)[0]))
    assert sae.log_theta.grad.abs().sum() > 0


def test_a_one_batch_window_measures_each_batch_on_its_own():
    torch.manual_seed(0)
    sae = JumpReLUSAE(d_in=6, d_sae=12, bandwidth=0.05)
    x = torch.randn(32, 6)
    cfg = SAECfg(inverse_simpson_floor=1.0, inverse_simpson_max_scale=3.0)
    coefficients = LossCoefficients(lr=1.0, lambda_l0=0.0, lambda_is=3.0)

    windowed = training_loss(sae, x, cfg, coefficients, target_l0=4.0, usage=torch.rand(12))
    alone = training_loss(sae, x, cfg, coefficients, target_l0=4.0)

    assert windowed.d_is == pytest.approx(alone.d_is)
    torch.testing.assert_close(windowed.inverse_simpson_penalty, alone.inverse_simpson_penalty)


def test_training_loss_skips_inverse_simpson_without_a_floor():
    sae = JumpReLUSAE(d_in=4, d_sae=8)
    terms = training_loss(
        sae, torch.randn(8, 4), SAECfg(), LossCoefficients(lr=1.0, lambda_l0=1.0, lambda_is=1.0), target_l0=2.0
    )

    assert float(terms.inverse_simpson_penalty) == 0.0
    assert terms.d_is == terms.u_is == 0.0


def test_auxk_trains_the_dead_latents_on_the_main_residual():
    torch.manual_seed(0)
    sae = JumpReLUSAE(d_in=6, d_sae=12, bandwidth=0.05)
    x = torch.randn(32, 6)
    cfg = SAECfg(auxk_scale=0.5, auxk_k=2, auxk_dead_tokens=100)
    coefficients = LossCoefficients(lr=1.0, lambda_l0=0.5, lambda_is=0.0)
    tokens_since_fired = torch.zeros(12, dtype=torch.int64)
    tokens_since_fired[8:] = 100

    terms = training_loss(sae, x, cfg, coefficients, target_l0=4.0, tokens_since_fired=tokens_since_fired)
    terms.auxk_penalty.backward()

    assert float(terms.auxk_penalty.detach()) > 0.0
    torch.testing.assert_close(terms.total, terms.reconstruction + 0.5 * terms.l0_penalty + terms.auxk_penalty)
    assert sae.W_enc.grad[:, 8:].abs().sum() > 0
    assert sae.W_enc.grad[:, :8].abs().sum() == 0
    assert sae.log_theta.grad is None


@pytest.mark.parametrize("input_scale", [1.0, 30.0])
def test_auxk_is_the_same_fraction_of_the_reconstruction_loss_under_either_reduction(input_scale):
    ratios = []
    for reconstruction_loss in ("vector_sum", "coordinate_mean"):
        torch.manual_seed(0)
        sae = JumpReLUSAE(d_in=6, d_sae=12, bandwidth=0.05)
        x = torch.randn(32, 6) * input_scale
        cfg = SAECfg(reconstruction_loss=reconstruction_loss, auxk_scale=1 / 32, auxk_k=2, auxk_dead_tokens=100)
        dead_counts = torch.zeros(12, dtype=torch.int64)
        dead_counts[8:] = 100

        terms = training_loss(
            sae, x, cfg, LossCoefficients(lr=1.0, lambda_l0=0.0, lambda_is=0.0), target_l0=4.0,
            tokens_since_fired=dead_counts,
        )
        ratios.append(float(terms.auxk_penalty.detach() / terms.reconstruction.detach()))

    assert ratios[0] == pytest.approx(ratios[1])
    assert ratios[0] > 0.0


def test_auxk_counts_tokens_since_each_latent_fired():
    torch.manual_seed(0)
    sae = JumpReLUSAE(d_in=6, d_sae=12, bandwidth=0.05)
    x = torch.randn(32, 6)
    cfg = SAECfg(auxk_scale=0.5, auxk_dead_tokens=100)
    tokens_since_fired = torch.full((12,), 10, dtype=torch.int64)
    sae.log_theta.data[8:] = 10.0

    terms = training_loss(
        sae, x, cfg, LossCoefficients(lr=1.0, lambda_l0=0.0, lambda_is=0.0), target_l0=4.0,
        tokens_since_fired=tokens_since_fired,
    )

    fired = (terms.f > 0).any(dim=0)
    assert fired.any() and not fired.all()
    assert (tokens_since_fired[fired] == 0).all()
    assert (tokens_since_fired[~fired] == 42).all()


def test_auxk_is_zero_while_no_latent_is_dead_or_without_a_counter():
    torch.manual_seed(0)
    sae = JumpReLUSAE(d_in=6, d_sae=12, bandwidth=0.05)
    x = torch.randn(32, 6)
    cfg = SAECfg(auxk_scale=0.5, auxk_dead_tokens=100)
    coefficients = LossCoefficients(lr=1.0, lambda_l0=0.0, lambda_is=0.0)

    alive = training_loss(
        sae, x, cfg, coefficients, target_l0=4.0, tokens_since_fired=torch.zeros(12, dtype=torch.int64)
    )
    without_counter = training_loss(sae, x, cfg, coefficients, target_l0=4.0)

    assert float(alive.auxk_penalty.detach()) == float(without_counter.auxk_penalty.detach()) == 0.0


def test_cosine_decay_needs_a_run_length():
    cfg = SAECfg(lr=1.0, warmup_steps=0, lr_schedule="cosine_decay")

    with pytest.raises(ValueError, match="n_steps"):
        coefficients_at(cfg, 5)
    assert coefficients_at(cfg, 10, n_steps=10).lr == pytest.approx(0.1)


def _one_update(cfg: SAECfg, *, step: int) -> tuple[JumpReLUSAE, torch.Tensor, torch.optim.Optimizer]:
    cfg = cfg.model_copy(update={"log_every": 1})
    torch.manual_seed(1)
    sae = JumpReLUSAE(d_in=6, d_sae=12, bandwidth=0.05)
    before = sae.W_dec.detach().clone()
    optim = torch.optim.Adam(sae.parameters(), lr=123.0)
    x = torch.randn(32, 6)
    terms = training_loss(sae, x, cfg, LossCoefficients(lr=1e-2, lambda_l0=1.0, lambda_is=0.0), target_l0=3.0)
    stats = optimizer_step(sae, optim, terms.total, x, cfg, step=step, lr=1e-2)
    assert 0.0 <= stats.ste_window_pct <= 100.0
    assert stats.encoder_grad_norm > 0.0
    return sae, before, optim


def test_optimizer_step_skips_gradient_diagnostics_between_logged_steps():
    torch.manual_seed(1)
    sae = JumpReLUSAE(d_in=6, d_sae=12, bandwidth=0.05)
    optim = torch.optim.Adam(sae.parameters())
    cfg = SAECfg(log_every=10)
    x = torch.randn(32, 6)
    terms = training_loss(sae, x, cfg, LossCoefficients(lr=1e-2, lambda_l0=1.0, lambda_is=0.0), target_l0=3.0)
    stats = optimizer_step(sae, optim, terms.total, x, cfg, step=5, lr=1e-2)
    assert (stats.ste_window_pct, stats.theta_grad_norm, stats.encoder_grad_norm) == (0.0, 0.0, 0.0)


def test_optimizer_step_sets_lr_and_keeps_unit_norm_decoder():
    sae, before, optim = _one_update(SAECfg(decoder_unit_norm=True, gradient_clip_norm=1.0), step=1)

    assert all(group["lr"] == 1e-2 for group in optim.param_groups)
    assert not torch.equal(sae.W_dec, before)
    torch.testing.assert_close(sae.W_dec.detach().norm(dim=0), torch.ones(12))


def test_optimizer_step_freezes_the_decoder_during_the_freeze_window():
    sae, before, _ = _one_update(SAECfg(decoder_unit_norm=False, decoder_freeze_steps=5), step=5)

    torch.testing.assert_close(sae.W_dec.detach(), before)


def test_vector_sum_reconstruction_is_coordinate_mean_times_width():
    x = torch.randn(8, 32)
    coordinate_mean = (x.square()).mean()
    vector_sum = x.square().sum(dim=-1).mean()
    torch.testing.assert_close(vector_sum, coordinate_mean * x.shape[-1])


def test_cosine_warmup_then_constant_schedule():
    peak_lr = 7.0e-5
    warmup_steps = 1000
    assert cosine_warmup_constant(0, peak_lr, warmup_steps) == pytest.approx(0.1 * peak_lr)
    assert cosine_warmup_constant(500, peak_lr, warmup_steps) == pytest.approx(0.55 * peak_lr)
    assert cosine_warmup_constant(1000, peak_lr, warmup_steps) == pytest.approx(peak_lr)
    assert cosine_warmup_constant(1500, peak_lr, warmup_steps) == pytest.approx(peak_lr)


def test_l0_penalty_scale_is_explicit():
    l0 = torch.tensor(12.0)
    base = quad_l0_loss(l0, 10)
    torch.testing.assert_close(quad_l0_loss(l0, 10, scale=4.0), base * 4.0)


def test_gradual_l0_schedule_reaches_final_target():
    cfg = SAECfg(l0_target_start=16.7, l0_target_warmup_steps=15_000)
    assert scheduled_l0_target(0, 10, cfg) == 16.7
    assert scheduled_l0_target(15_000, 10, cfg) == 10.0
    assert 10.0 < scheduled_l0_target(7_500, 10, cfg) < 16.7
