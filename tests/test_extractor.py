import torch
from torch import nn

from sae_pipeline.hooks.components import ComponentSpec
from sae_pipeline.hooks.extractor import capture, capture_many
from sae_pipeline.model.topology import HookSite


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block()])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layers[0](hidden_states)


def test_capture_many_matches_single_site_capture():
    model = _Model()
    values = torch.randn(2, 4, 3)
    pre_spec = ComponentSpec.parse(0, "resid_pre")
    post_spec = ComponentSpec.parse(0, "resid_post")
    linear_spec = ComponentSpec.parse(0, "mamba_out")
    pre_site = HookSite("resid_pre", 0, "<residual:0:pre>")
    post_site = HookSite("resid_post", 0, "<residual:0:post>")
    linear_site = HookSite("mamba_out", 0, "layers.0.linear")

    with capture(model, post_spec, post_site) as buffer:
        model(values)
        single = buffer.consume()

    with capture_many(
        model,
        {
            "pre": (pre_spec, pre_site),
            "post": (post_spec, post_site),
            "linear": (linear_spec, linear_site),
        },
    ) as buffers:
        model(values)
        captured = {name: buffer.consume() for name, buffer in buffers.items()}

    expected = model.layers[0].linear(values).reshape(-1, 3)
    torch.testing.assert_close(captured["pre"], values.reshape(-1, 3))
    torch.testing.assert_close(captured["post"], single)
    torch.testing.assert_close(captured["post"], expected)
    torch.testing.assert_close(captured["linear"], expected)
