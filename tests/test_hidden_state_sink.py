from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.config import ActivationSinkCfg
from sae_pipeline.extraction.sink import ActivationSink


def _sink(tmp_path: Path, lifecycle: str = "consume") -> ActivationSink:
    sink = ActivationSink(ActivationSinkCfg(path=str(tmp_path / "sink"), lifecycle=lifecycle))
    sink.prepare()
    return sink


def _step(path: Path, ids, positions, **sites) -> str:
    save_file({"input_ids": torch.tensor(ids), "positions": torch.tensor(positions), **sites}, str(path))
    return str(path)


def test_collect_reassembles_prompts_across_steps_in_submission_order(tmp_path):
    sink = _sink(tmp_path)
    prompts = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
    values = torch.arange(9 * 2, dtype=torch.float32).reshape(9, 2)
    # Step 0 holds prompt 1 then prompt 0 plus a padding row; step 1 holds prompt 2.
    first = _step(sink.path / "step_0.safetensors", [4, 5, 6, 1, 2, 3, 0], [0, 1, 2, 0, 1, 2, 0],
                  a=torch.cat([values[3:6], values[0:3], torch.zeros(1, 2)]))
    second = _step(sink.path / "step_1.safetensors", [1, 2, 3], [0, 1, 2], a=values[6:9])

    activations = sink.collect([first, second], prompts, ["a"], device="cpu")

    torch.testing.assert_close(activations["a"], values)
    assert list(sink.path.iterdir()) == []  # consumed


def test_collect_keeps_files_under_the_cache_lifecycle(tmp_path):
    sink = _sink(tmp_path, lifecycle="cache")
    step = _step(sink.path / "step_0.safetensors", [1, 2], [0, 1], a=torch.zeros(2, 2))

    sink.collect([step], torch.tensor([[1, 2]]), ["a"], device="cpu")
    sink.cleanup()

    assert Path(step).exists()


def test_collect_reports_prompts_that_were_never_captured(tmp_path):
    sink = _sink(tmp_path)
    step = _step(sink.path / "step_0.safetensors", [1, 2], [0, 1], a=torch.zeros(2, 2))

    with pytest.raises(RuntimeError, match=r"sequences \[1\]"):
        sink.collect([step], torch.tensor([[1, 2], [7, 8]]), ["a"], device="cpu")


def test_cleanup_removes_leftovers_of_a_consuming_run(tmp_path):
    sink = _sink(tmp_path)
    _step(sink.path / "step_0.safetensors", [1], [0], a=torch.zeros(1, 2))

    sink.cleanup()

    assert not sink.path.exists()
