"""Fake vLLM model, worker and engine shaped like vLLM's NemotronH, for extraction tests."""

from types import SimpleNamespace

import torch
from torch import nn

from sae_pipeline.config import PipelineCfg
from sae_pipeline.extraction.vllm_worker import ActivationHookExtension
from sae_pipeline.hooks.components import ComponentSpec

D = 4


class FakeMambaMixer(nn.Linear):
    pass


class FakeMoE(nn.Linear):
    pass


class DecoderLayer(nn.Module):
    """vLLM-style layer: returns (mixer output, running residual).

    Like vLLM's fused add-RMSNorm (here without the norm), it updates its
    inputs in place: the residual absorbs the previous mixer output, which is
    then overwritten.
    """

    def __init__(self, mixer: nn.Module) -> None:
        super().__init__()
        self.mixer = mixer

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
        else:
            residual.add_(hidden_states)
            hidden_states.copy_(residual)
        return self.mixer(residual), residual


class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(50, D)
        self.layers = nn.ModuleList([DecoderLayer(FakeMambaMixer(D, D)), DecoderLayer(FakeMoE(D, D))])

    def forward(self, input_ids, positions):
        hidden, residual = self.embed(input_ids), None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        return hidden + residual


class CausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids, positions, intermediate_tensors=None):
        return self.model(input_ids, positions)


class Worker(ActivationHookExtension):
    def __init__(self) -> None:
        torch.manual_seed(0)
        self.model_runner = SimpleNamespace(model=CausalLM())


SPECS = {
    "L0_mamba_out": ComponentSpec.parse(0, "mamba_out"),
    "L1_moe_out": ComponentSpec.parse(1, "moe_out"),
    "L0_resid_post": ComponentSpec.parse(0, "resid_post"),
}


class FakeEngine(ActivationHookExtension):
    """Batches every call's prompts into one forward pass, plus one padding row."""

    def __init__(self, **engine_kwargs) -> None:
        torch.manual_seed(0)
        self.kwargs = engine_kwargs
        self.model_runner = SimpleNamespace(model=CausalLM())
        self.calls: list[int] = []

    def collective_rpc(self, method, args=()):
        return [getattr(self, method)(*args)]

    def generate(self, prompts, sampling_params, use_tqdm):
        assert sampling_params == "one-token"
        self.calls.append(len(prompts))
        ids = [token for prompt in prompts for token in prompt["prompt_token_ids"]] + [0]
        positions = [p for prompt in prompts for p in range(len(prompt["prompt_token_ids"]))] + [0]
        with torch.no_grad():
            self.model_runner.model(torch.tensor(ids), torch.tensor(positions))
        return [object() for _ in prompts]


def vllm_cfg(tmp_path, *, sequences_per_call=2) -> PipelineCfg:
    return PipelineCfg.from_yaml("configs/dev.yaml").with_overrides(
        **{
            "model.name": "example/model",
            "data.seq_len": 3,
            "extraction.vllm.sequences_per_call": sequences_per_call,
            "extraction.sink.path": str(tmp_path / "sink"),
        }
    )


def reference(model, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Each sequence run on its own, outside vLLM."""
    rows = {slug: [] for slug in SPECS}
    with torch.no_grad():
        for row in token_ids:
            embedded = model.model.embed(row)
            mamba = model.model.layers[0].mixer(embedded)
            rows["L0_mamba_out"].append(mamba)
            rows["L0_resid_post"].append(mamba + embedded)
            rows["L1_moe_out"].append(model.model.layers[1].mixer(mamba + embedded))
    return {slug: torch.cat(values) for slug, values in rows.items()}
