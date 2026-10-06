"""ActivationHookExtension on a fake model shaped like vLLM's NemotronH."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from sae_pipeline.extraction.vllm_worker import ActivationHookExtension

D = 4


class FakeMambaMixer(nn.Linear):
    pass


class FakeMoE(nn.Linear):
    pass


class _DecoderLayer(nn.Module):
    """vLLM-style layer: returns (mixer output, running residual)."""

    def __init__(self, mixer: nn.Module) -> None:
        super().__init__()
        self.mixer = mixer

    def forward(self, positions, hidden_states, residual):
        residual = hidden_states if residual is None else hidden_states + residual
        return self.mixer(residual), residual


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(50, D)
        self.layers = nn.ModuleList([_DecoderLayer(FakeMambaMixer(D, D)), _DecoderLayer(FakeMoE(D, D))])

    def forward(self, input_ids, positions):
        hidden, residual = self.embed(input_ids), None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        return hidden + residual


class _CausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()

    def forward(self, input_ids, positions, intermediate_tensors=None):
        return self.model(input_ids, positions)


class _Worker(ActivationHookExtension):
    def __init__(self) -> None:
        torch.manual_seed(0)
        self.model_runner = SimpleNamespace(model=_CausalLM())


SITES = [
    ("L0_resid_pre", "resid_pre", 0),
    ("L1_resid_pre", "resid_pre", 1),
    ("L0_resid_post", "resid_post", 0),
    ("L0_mamba_out", "mamba_out", 0),
    ("L1_moe_out", "moe_out", 1),
]


def test_one_file_per_forward_with_inputs_and_every_site(tmp_path):
    worker = _Worker()
    worker.sae_install_hooks(SITES, str(tmp_path))
    model = worker.model_runner.model
    input_ids = torch.tensor([3, 4, 5, 6, 7, 8])
    positions = torch.tensor([0, 1, 2, 3, 0, 1])  # two sequences in one step

    with torch.no_grad():
        model(input_ids, positions)
        model(input_ids[:2], positions[:2])
    paths = worker.sae_flush()
    worker.sae_remove_hooks()

    assert [Path(p).name for p in paths] == ["step_00000000.safetensors", "step_00000001.safetensors"]
    step = load_file(paths[0])
    assert torch.equal(step["input_ids"], input_ids) and torch.equal(step["positions"], positions)
    with torch.no_grad():
        embedded = model.model.embed(input_ids)
        mamba = model.model.layers[0].mixer(embedded)
        resid_1 = mamba + embedded
        moe = model.model.layers[1].mixer(resid_1)
    torch.testing.assert_close(step["L0_resid_pre"], embedded)
    torch.testing.assert_close(step["L0_mamba_out"], mamba)
    torch.testing.assert_close(step["L0_resid_post"], resid_1)
    torch.testing.assert_close(step["L1_resid_pre"], resid_1)
    torch.testing.assert_close(step["L1_moe_out"], moe)
    assert load_file(paths[1])["L1_moe_out"].shape == (2, D)


def test_flush_returns_only_new_files_and_removed_hooks_stop_capturing(tmp_path):
    worker = _Worker()
    worker.sae_install_hooks(SITES[:1], str(tmp_path))
    model = worker.model_runner.model
    with torch.no_grad():
        model(torch.tensor([1, 2]), torch.tensor([0, 1]))
    assert len(worker.sae_flush()) == 1
    assert worker.sae_flush() == []
    worker.sae_remove_hooks()
    with torch.no_grad():
        model(torch.tensor([1, 2]), torch.tensor([0, 1]))
    assert len(list(tmp_path.iterdir())) == 1


def test_a_site_on_the_wrong_kind_of_block_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="FakeMambaMixer, not a MoE block"):
        _Worker().sae_install_hooks([("L0_moe_out", "moe_out", 0)], str(tmp_path))
    with pytest.raises(ValueError, match="no hook point"):
        _Worker().sae_install_hooks([("L0_expert", "expert", 0)], str(tmp_path))
