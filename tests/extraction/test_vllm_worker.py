"""ActivationHookExtension on a fake model shaped like vLLM's NemotronH."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from sae_pipeline.extraction.vllm_worker import assemble_rows
from tests.extraction.fakes import D, Worker

SITES = [
    ("L0_resid_pre", "resid_pre", 0),
    ("L1_resid_pre", "resid_pre", 1),
    ("L0_resid_post", "resid_post", 0),
    ("L0_mamba_out", "mamba_out", 0),
    ("L1_moe_out", "moe_out", 1),
]


def test_one_file_per_forward_with_inputs_and_every_site(tmp_path):
    worker = Worker()
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
    worker = Worker()
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
        Worker().sae_install_hooks([("L0_moe_out", "moe_out", 0)], str(tmp_path))
    with pytest.raises(ValueError, match="no hook point"):
        Worker().sae_install_hooks([("L0_expert", "expert", 0)], str(tmp_path))


def test_a_probe_reports_the_kept_bytes_per_row_and_the_peak_above_them(monkeypatch):
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (5000, 9000))
    worker = Worker()
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
    worker = Worker()
    sites = [("L0_resid_post", "resid_post", 0), ("L0_mamba_out", "mamba_out", 0), ("L1_moe_out", "moe_out", 1)]
    worker.sae_install_hooks(sites, None)
    with torch.no_grad():
        worker.model_runner.model(torch.tensor([4, 5, 1, 2]), torch.tensor([0, 1, 0, 1]))
    prompts = [[1, 2], [4, 5]]
    expected = assemble_rows(worker.sae_steps, torch.tensor(prompts), [slug for slug, _, _ in sites])

    assert worker.sae_save(prompts, str(tmp_path / "fill_0"), threads=2, extra_files={"state.pt": b"s"}) is None
    assert worker.sae_steps == []

    report = worker.sae_flush_saves()

    assert sorted(p.name for p in tmp_path.iterdir()) == ["fill_0"]
    fill = tmp_path / "fill_0"
    assert load_file(fill / "token_ids.safetensors")["token_ids"].tolist() == prompts
    assert (fill / "state.pt").read_bytes() == b"s"
    for slug, rows in expected.items():
        torch.testing.assert_close(load_file(fill / f"{slug}.safetensors")["activations"], rows)
    assert report["bytes"] == 3 * 4 * D * 4 and worker.sae_flush_saves() is None


def test_a_fill_is_published_when_the_next_one_is_saved(tmp_path):
    worker = Worker()
    worker.sae_install_hooks([("L0_resid_post", "resid_post", 0)], None)
    saved, expected = [], []
    for index, prompt in enumerate([[1, 2], [4, 5]]):
        with torch.no_grad():
            worker.model_runner.model(torch.tensor(prompt), torch.tensor([0, 1]))
        expected.append(assemble_rows(worker.sae_steps, torch.tensor([prompt]), ["L0_resid_post"])["L0_resid_post"])
        saved.append(worker.sae_save([prompt], str(tmp_path / f"fill_{index}"), threads=1))
        # The fill being written stays hidden until the next save publishes it.
        assert sorted(p.name for p in tmp_path.iterdir()) == [f"fill_{i}" for i in range(index)] + [f"fill_{index}.partial"]

    assert saved[0] is None and saved[1]["bytes"] == 2 * D * 4
    worker.sae_remove_hooks()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fill_0", "fill_1"]
    # The host buffers are reused: the first fill was written before the second overwrote them.
    for index, rows in enumerate(expected):
        torch.testing.assert_close(load_file(tmp_path / f"fill_{index}" / "L0_resid_post.safetensors")["activations"], rows)
