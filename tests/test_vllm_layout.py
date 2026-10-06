import pytest

from sae_pipeline.extraction.base import UnsupportedComponentError
from sae_pipeline.extraction.vllm_layout import VLLMLayout, residual_layer_id
from sae_pipeline.hooks.components import ComponentSpec


def test_residual_sites_map_to_layer_boundaries():
    assert residual_layer_id(ComponentSpec.parse(0, "resid_pre"), 52) == 0
    assert residual_layer_id(ComponentSpec.parse(0, "resid_post"), 52) == 1
    assert residual_layer_id(ComponentSpec.parse(12, "resid_pre"), 52) == 12
    assert residual_layer_id(ComponentSpec.parse(51, "resid_post"), 52) == 52


@pytest.mark.parametrize("component", ["mamba_out", "attn_out_prelinear", "moe_out", "expert.3", "shared_expert"])
def test_components_inside_a_layer_are_unsupported(component):
    with pytest.raises(UnsupportedComponentError, match="residual-stream"):
        residual_layer_id(ComponentSpec.parse(5, component), 52)


def test_layers_outside_the_model_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        residual_layer_id(ComponentSpec.parse(52, "resid_post"), 52)


def test_layout_deduplicates_shared_boundaries_and_sorts_columns():
    layout = VLLMLayout.from_specs(
        {
            "L26_resid_post": ComponentSpec.parse(26, "resid_post"),
            "L11_resid_post": ComponentSpec.parse(11, "resid_post"),
            "L12_resid_pre": ComponentSpec.parse(12, "resid_pre"),
        },
        num_hidden_layers=52,
    )

    assert layout.layer_ids == (12, 27)
    assert layout.columns == {"L26_resid_post": 1, "L11_resid_post": 0, "L12_resid_pre": 0}
