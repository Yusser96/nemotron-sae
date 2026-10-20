"""Numerical safety is independent of finite sparsity/concentration diagnostics."""
from types import SimpleNamespace

import pytest
import torch

from sae_pipeline.config import EarlyStoppingCfg, SAECfg
from sae_pipeline.sae.jumprelu import JumpReLUSAE
from sae_pipeline.sae.objective import optimizer_step
from sae_pipeline.sae.train_streaming import (
    _divergence,
    finite_evaluation_error,
    threshold_warnings,
)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_gradient_rejected_before_optimizer_mutation(bad):
    class FiniteValueBadGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight):
            return weight.sum() * 0 + 1

        @staticmethod
        def backward(ctx, grad):
            return torch.full((4, 8), bad)

    sae = JumpReLUSAE(d_in=4, d_sae=8)
    optim = torch.optim.Adam(sae.parameters(), lr=0.001)
    before = {name: value.detach().clone() for name, value in sae.named_parameters()}
    loss = FiniteValueBadGradient.apply(sae.W_enc)
    assert torch.isfinite(loss)
    with pytest.raises(RuntimeError, match="non-finite"):
        optimizer_step(sae, optim, loss, torch.randn(8, 4),
                       SAECfg(gradient_clip_norm=1.0), step=1, lr=0.001)
    assert not optim.state
    for name, value in sae.named_parameters():
        torch.testing.assert_close(value, before[name])


def evaluation():
    return SimpleNamespace(objective={"en": 1.0, "de": 1.0}, mixed_objective=1.0,
                           metrics={lang: SimpleNamespace(l0=2500.0, fvu=0.02)
                                    for lang in ("en", "de")})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["language_objective", "mixed_objective", "metric"])
def test_nonfinite_validation_rejected_even_when_other_values_mask_it(bad, field):
    result = evaluation()
    if field == "language_objective":
        result.objective["de"] = bad
    elif field == "mixed_objective":
        result.mixed_objective = bad
    else:
        result.metrics["de"].fvu = bad
    site = SimpleNamespace(cfg=SimpleNamespace(l0_warmup_steps=50000), l0_target=10,
                           early_stop=SimpleNamespace(best=0.05))
    cfg = EarlyStoppingCfg(divergence_l0_factor=None, divergence_objective_factor=None)
    assert finite_evaluation_error(result) is not None
    assert _divergence(site, result, 50000, cfg) is not None
    assert threshold_warnings(site, result, 50000) == ()


def test_finite_large_l0_and_objective_warn_without_fatal_guard():
    result = evaluation()
    site = SimpleNamespace(cfg=SimpleNamespace(l0_warmup_steps=50000), l0_target=10,
                           early_stop=SimpleNamespace(best=0.05))
    cfg = EarlyStoppingCfg(divergence_l0_factor=None, divergence_objective_factor=None)
    assert _divergence(site, result, 50000, cfg) is None
    assert threshold_warnings(site, result, 50000) == ("l0", "objective")
    assert threshold_warnings(site, result, 49999) == ("objective",)
    assert _divergence(site, result, 50000, EarlyStoppingCfg(divergence_l0_factor=3)) is not None
