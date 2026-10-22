import torch
from torch import nn

from sae_pipeline.extraction.hf import HFHookExtractor
from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.model.topology import HookSite


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(10, 3)
        self.layers = nn.ModuleList([nn.Linear(3, 3)])

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.layers[0](self.embed(token_ids))


REQUESTS = {
    "pre": (ComponentSpec.parse(0, "resid_pre"), HookSite("resid_pre", 0, "<residual:0:pre>")),
    "post": (ComponentSpec.parse(0, "resid_post"), HookSite("resid_post", 0, "<residual:0:post>")),
}


def test_hf_extractor_returns_each_tokens_residual_before_and_after_the_layer():
    torch.manual_seed(0)
    model = _Model()
    token_ids = torch.randint(0, 10, (2, 5))

    batch = HFHookExtractor(model, REQUESTS, model_device="cpu").extract(token_ids)

    with torch.no_grad():
        embedded = model.embed(token_ids).reshape(-1, 3)
        expected = {"pre": embedded, "post": model.layers[0](embedded)}
    assert torch.equal(batch.token_ids, token_ids)
    assert batch.activations.keys() == expected.keys()
    for slug, value in expected.items():
        torch.testing.assert_close(batch.activations[slug], value)
