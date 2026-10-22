import pytest
import torch

from sae_pipeline.extraction.compare import compare_activations

GATE = dict(min_cosine_mean=0.999, min_cosine_p01=0.99, max_relative_l2=2e-2)


def test_identical_activations_agree_perfectly():
    x = torch.randn(200, 8)
    agreement = compare_activations(x, x.clone())

    assert agreement.cosine_mean == pytest.approx(1.0)
    assert agreement.cosine_min == pytest.approx(1.0)
    assert agreement.max_abs_error == 0.0
    assert agreement.relative_l2 == 0.0
    assert agreement.passes(**GATE)


def test_bf16_rounding_passes_but_a_wrong_layer_fails():
    torch.manual_seed(0)
    x = torch.randn(500, 64)

    assert compare_activations(x, x.to(torch.bfloat16)).passes(**GATE)
    assert not compare_activations(x, torch.randn(500, 64)).passes(**GATE)


def test_relative_l2_and_max_error_are_measured_against_the_reference():
    reference = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    candidate = reference + torch.tensor([[0.0, 0.5], [0.0, 0.0]])

    agreement = compare_activations(reference, candidate)

    assert agreement.max_abs_error == pytest.approx(0.5)
    assert agreement.relative_l2 == pytest.approx(0.5 / (50**0.5))


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_activations(torch.zeros(3, 2), torch.zeros(4, 2))


def test_zero_pairs_agree_but_zero_nonzero_pairs_fail():
    x = torch.zeros(4, 3)
    assert compare_activations(x, x).passes(**GATE)
    assert not compare_activations(x, torch.ones_like(x)).passes(**GATE)
    mixed = torch.cat([x, torch.ones_like(x)])
    assert compare_activations(mixed, mixed).passes(**GATE)
