import pytest
import torch
from torch import nn

from sae_pipeline.config import HookTargetCfg
from sae_pipeline.extraction.adapters import NemotronExtractor
from sae_pipeline.extraction.hf import HFHookExtractor
from sae_pipeline.hooks.components import site_requests


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 3)
        self.layers = nn.ModuleList([nn.Linear(3, 3)])

    def forward(self, input_ids, use_cache=False):
        return self.layers[0](self.embed(input_ids))


def test_maintained_hf_adapter_matches_current_hooks_bitwise():
    pytest.importorskip("nemotron_extraction")
    model = Model()
    targets = [HookTargetCfg(slug=kind, layer=0, component=kind, hook_name="layers.0")
               for kind in ("resid_pre", "resid_post")]
    requests = site_requests(targets)
    tokens = torch.tensor([[1, 2, 3], [3, 2, 1]])
    reference = HFHookExtractor(model, requests, model_device="cpu").extract(tokens)
    adapter = NemotronExtractor(requests, backend="nemotron_extraction_hf", model=model, device="cpu")
    try:
        actual = adapter.extract(tokens)
        assert torch.equal(actual.token_ids, reference.token_ids)
        assert actual.activations.keys() == reference.activations.keys()
        for key in actual.activations:
            assert torch.equal(actual.activations[key], reference.activations[key])
            actual.activations[key].requires_grad_()
    finally:
        adapter.close()
    assert not model.layers[0]._forward_hooks and not model.layers[0]._forward_pre_hooks


def test_adapter_rejects_custom_hook_instead_of_silently_substituting():
    pytest.importorskip("nemotron_extraction")
    targets = [HookTargetCfg(slug="custom", layer=0, component="resid_post", hook_name="embed")]
    with pytest.raises(ValueError, match="differs"):
        NemotronExtractor(site_requests(targets), backend="nemotron_extraction_hf", model=Model(), device="cpu")


class Mamba(nn.Linear):
    pass


class MoE(nn.Linear):
    pass


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(3, 3)

    def forward(self, x):
        return self.o_proj(x * 2)


class Block(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer

    def forward(self, x):
        return x + self.mixer(x)


class Hybrid(Model):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(Mamba(3, 3)), Block(Attention()), Block(MoE(3, 3))])

    def forward(self, input_ids, use_cache=False):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def test_hf_parity_covers_residual_attention_mamba_and_moe():
    pytest.importorskip("nemotron_extraction")
    model = Hybrid()
    kinds = [(0, "resid_pre"), (2, "resid_post"), (0, "mamba_out"),
             (1, "attn_out"), (1, "attn_out_prelinear"), (2, "moe_out")]
    targets = [HookTargetCfg(slug=f"custom_{layer}_{kind}", layer=layer, component=kind,
               hook_name=f"layers.{layer}" + ("" if kind.startswith("resid_") else ".mixer"))
               for layer, kind in kinds]
    requests = site_requests(targets)
    ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    expected = HFHookExtractor(model, requests, model_device="cpu").extract(ids)
    adapter = NemotronExtractor(requests, backend="nemotron_extraction_hf", model=model, device="cpu")
    try:
        actual = adapter.extract(ids)
        assert actual.activations.keys() == expected.activations.keys()
        assert all(torch.equal(actual.activations[key], expected.activations[key]) for key in actual.activations)
    finally:
        adapter.close()
