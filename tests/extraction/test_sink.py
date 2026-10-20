from pathlib import Path

import torch
from safetensors.torch import save_file

from sae_pipeline.config import ActivationSinkCfg
from sae_pipeline.extraction.sink import ActivationSink


def _sink(tmp_path: Path, lifecycle: str = "consume") -> ActivationSink:
    sink = ActivationSink(ActivationSinkCfg(path=str(tmp_path / "sink"), lifecycle=lifecycle))
    sink.prepare()
    return sink


def _step(path: Path, **tensors) -> str:
    save_file(tensors, str(path))
    return str(path)


def test_read_loads_steps_in_order_and_consumes_them(tmp_path):
    sink = _sink(tmp_path)
    first = _step(sink.path / "step_1.safetensors", a=torch.ones(2))
    second = _step(sink.path / "step_0.safetensors", a=torch.zeros(3))

    steps = sink.read([first, second])

    assert [step["a"].numel() for step in steps] == [2, 3]
    assert list(sink.path.iterdir()) == []


def test_read_keeps_files_under_the_cache_lifecycle(tmp_path):
    sink = _sink(tmp_path, lifecycle="cache")
    step = _step(sink.path / "step_0.safetensors", a=torch.zeros(2, 2))

    sink.read([step])
    sink.cleanup()

    assert Path(step).exists()


def test_cleanup_removes_leftovers_of_a_consuming_run(tmp_path):
    sink = _sink(tmp_path)
    _step(sink.path / "step_0.safetensors", a=torch.zeros(1, 2))

    sink.cleanup()

    assert not sink.path.exists()
