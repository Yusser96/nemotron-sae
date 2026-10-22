from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sae_pipeline.eval.metrics import delta_ce
from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.hooks.patching import patch_sae
from sae_pipeline.model.topology import HookSite


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.eye(4) * 5)
        self.layers = nn.ModuleList([nn.Identity()])

    def forward(self, input_ids, labels):
        logits = self.layers[0](self.embedding(input_ids))
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 4), labels[:, 1:].reshape(-1))
        return SimpleNamespace(loss=loss)


class Reconstruction(nn.Module):
    def __init__(self, zero=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.zero = zero

    def encode(self, x):
        return x

    def decode(self, x):
        return torch.zeros_like(x) if self.zero else x


@pytest.mark.parametrize("kind", ["resid_pre", "resid_post"])
def test_identity_preserves_ce_and_zero_reconstruction_degrades_it(kind):
    model = TinyLM()
    ids = torch.tensor([[0, 0, 0], [1, 1, 1]])
    register = partial(patch_sae, spec=ComponentSpec.parse(0, kind), site=HookSite(kind, 0, "layers.0"))
    clean = delta_ce(model, Reconstruction(), ids, register)
    zero = delta_ce(model, Reconstruction(zero=True), ids, register)
    assert clean["delta_ce"] == pytest.approx(0, abs=1e-8)
    assert zero["delta_ce"] > 1 and zero["ce_patched"] == pytest.approx(float(torch.log(torch.tensor(4.))))
    assert model.training and not model.layers[0]._forward_hooks and not model.layers[0]._forward_pre_hooks


def test_patch_unfolds_normalization_and_centering_and_removes_hook_on_error():
    model = TinyLM()
    ids = torch.tensor([[0, 0, 0]])
    spec = ComponentSpec.parse(0, "resid_post")
    site = HookSite("resid_post", 0, "layers.0")
    reference = model(ids, labels=ids).loss
    with patch_sae(model, Reconstruction(), spec, site, input_scale=0.2, input_center=torch.ones(4)):
        torch.testing.assert_close(model(ids, labels=ids).loss, reference)
    with pytest.raises(RuntimeError, match="interrupt"):
        with patch_sae(model, Reconstruction(), spec, site):
            raise RuntimeError("interrupt")
    assert not model.layers[0]._forward_hooks
