import pytest
import torch
from torch import nn

from sae_pipeline.hooks.components import ComponentSpec, resolve
from sae_pipeline.hooks.extractor import capture, capture_many
from sae_pipeline.model.topology import HookSite, enumerate_hooks
from tests.extraction.test_vllm_worker import _Worker


class FakeAttention(nn.Module):
    num_heads = 2

    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(8, 4, bias=False)

    def forward(self, hidden_states):
        return self.o_proj(torch.cat([hidden_states, hidden_states * 2], dim=-1))


@pytest.mark.parametrize("length", [8, 128, 2048])
def test_hf_and_worker_capture_the_same_bf16_projection_input(length):
    worker = _Worker()
    model = worker.model_runner.model.to(torch.bfloat16)
    model.model.layers[0].mixer = FakeAttention().to(torch.bfloat16)
    path = "model.layers.0.mixer"
    requests = {kind: (ComponentSpec.parse(0, kind), HookSite(kind, 0, path))
                for kind in ("attn_out", "attn_out_prelinear")}
    worker.sae_install_hooks([(kind, kind, 0) for kind in requests], None)
    ids = torch.arange(length) % 50
    positions = torch.arange(length)
    with capture_many(model, requests) as buffers, torch.no_grad():
        model(ids, positions)
        hf = {kind: buffer.consume() for kind, buffer in buffers.items()}
    step = worker.sae_steps[0]
    assert hf["attn_out"].shape == (length, 4)
    assert hf["attn_out_prelinear"].shape == (length, 8)
    for kind in requests:
        torch.testing.assert_close(hf[kind], step[kind], rtol=0, atol=0)
    worker.sae_remove_hooks()


def test_topology_classifies_mixers_and_resolves_individual_heads():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].mixer = FakeAttention()
    sites = enumerate_hooks(model)
    assert resolve(ComponentSpec.parse(0, "attn_out"), sites).module_path == "layers.0.mixer"
    spec = ComponentSpec.parse(0, "attn_head.1")
    site = resolve(spec, sites)
    x = torch.randn(2, 3, 4)
    with capture(model, spec, site) as buf:
        model.layers[0].mixer(x)
        torch.testing.assert_close(buf.consume(), (x * 2).reshape(-1, 4))
    assert not any(s.component_kind == "mamba_out" for s in sites)


def test_keyword_residual_input_is_captured():
    from tests.hooks.test_extractor import _Model
    model = _Model()
    x = torch.randn(2, 3, 3)
    with capture(model, ComponentSpec.parse(0, "resid_pre"), HookSite("resid_pre", 0, "layers.0")) as buf:
        model.layers[0](hidden_states=x)
        torch.testing.assert_close(buf.consume(), x.reshape(-1, 3))


def test_shared_expert_container_does_not_shadow_indexed_children():
    from sae_pipeline.model.topology import enumerate_hooks
    from sae_pipeline.hooks.components import ComponentSpec, resolve
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].shared_experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    sites = enumerate_hooks(model)
    assert resolve(ComponentSpec.parse(0, "shared_expert.0"), sites).module_path == "layers.0.shared_experts.0"
    assert resolve(ComponentSpec.parse(0, "shared_expert.1"), sites).module_path == "layers.0.shared_experts.1"
