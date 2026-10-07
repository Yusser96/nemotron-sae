"""ActivationHookExtension on a fake model shaped like vLLM's NemotronH."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from sae_pipeline.extraction.vllm_worker import ActivationHookExtension, assemble_rows

D = 4


class FakeMambaMixer(nn.Linear):
    pass


class FakeMoE(nn.Linear):
    pass


class _DecoderLayer(nn.Module):
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


def _step(ids, positions, **sites) -> dict[str, torch.Tensor]:
    return {"input_ids": torch.tensor(ids), "positions": torch.tensor(positions), **sites}


def test_assemble_rows_reorders_prompts_across_steps_into_submission_order():
    prompts = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
    values = torch.arange(9 * 2, dtype=torch.float32).reshape(9, 2)
    # Step 0 holds prompt 1 then prompt 0 plus a padding row; step 1 holds prompt 2.
    steps = [
        _step([4, 5, 6, 1, 2, 3, 0], [0, 1, 2, 0, 1, 2, 0], a=torch.cat([values[3:6], values[0:3], torch.zeros(1, 2)])),
        _step([1, 2, 3], [0, 1, 2], a=values[6:9]),
    ]

    torch.testing.assert_close(assemble_rows(steps, prompts, ["a"])["a"], values)


def test_assemble_rows_reports_prompts_that_were_never_captured():
    with pytest.raises(RuntimeError, match=r"sequences \[1\]"):
        assemble_rows([_step([1, 2], [0, 1], a=torch.zeros(2, 2))], torch.tensor([[1, 2], [7, 8]]), ["a"])


def test_a_site_on_the_wrong_kind_of_block_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="FakeMambaMixer, not a MoE block"):
        _Worker().sae_install_hooks([("L0_moe_out", "moe_out", 0)], str(tmp_path))
    with pytest.raises(ValueError, match="no hook point"):
        _Worker().sae_install_hooks([("L0_expert", "expert", 0)], str(tmp_path))


def test_a_probe_reports_the_kept_bytes_per_row_and_the_peak_above_them(monkeypatch):
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (5000, 9000))
    worker = _Worker()
    worker.sae_install_hooks([("L0_resid_post", "resid_post", 0), ("L1_moe_out", "moe_out", 1)], None)
    worker.sae_start_probe()
    with torch.no_grad():
        worker.model_runner.model(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))
    # Kept per row: two float32 sites of width D plus int64 input_ids and positions.
    row_bytes = 2 * D * 4 + 2 * 8
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100 + 3 * row_bytes + 40)

    memory = worker.sae_end_probe()

    assert memory == {"free": 5000, "transient": 40, "row_bytes": row_bytes, "largest_site_row_bytes": D * 4}
    assert worker.sae_steps == []


def test_a_saved_fill_holds_each_sites_rows_in_submission_order(tmp_path):
    worker = _Worker()
    sites = [("L0_resid_post", "resid_post", 0), ("L0_mamba_out", "mamba_out", 0), ("L1_moe_out", "moe_out", 1)]
    worker.sae_install_hooks(sites, None)
    with torch.no_grad():
        worker.model_runner.model(torch.tensor([4, 5, 1, 2]), torch.tensor([0, 1, 0, 1]))
    prompts = [[1, 2], [4, 5]]
    expected = assemble_rows(worker.sae_steps, torch.tensor(prompts), [slug for slug, _, _ in sites])

    report = worker.sae_save(prompts, str(tmp_path / "fill_0"), threads=2)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["fill_0"]
    fill = tmp_path / "fill_0"
    assert load_file(fill / "token_ids.safetensors")["token_ids"].tolist() == prompts
    for slug, rows in expected.items():
        torch.testing.assert_close(load_file(fill / f"{slug}.safetensors")["activations"], rows)
    assert report["bytes"] == 3 * 4 * D * 4 and worker.sae_steps == []
