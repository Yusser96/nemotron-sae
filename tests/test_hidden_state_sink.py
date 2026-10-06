from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.config import ActivationSinkCfg
from sae_pipeline.extraction.sink import HiddenStateSink
from sae_pipeline.extraction.vllm_layout import VLLMLayout

LAYOUT = VLLMLayout(layer_ids=(3, 7), columns={"a": 0, "b": 1, "c": 1})


def _write(path: Path, token_ids: torch.Tensor, hidden: torch.Tensor) -> Path:
    save_file({"token_ids": token_ids, "hidden_states": hidden}, str(path))
    return path


def _sink(tmp_path: Path, lifecycle: str = "consume") -> HiddenStateSink:
    sink = HiddenStateSink(ActivationSinkCfg(path=str(tmp_path / "sink"), lifecycle=lifecycle))
    sink.prepare()
    return sink


def test_read_slices_each_site_and_drops_generated_tokens(tmp_path):
    sink = _sink(tmp_path)
    prompt = torch.tensor([5, 6, 7])
    hidden = torch.randn(4, 2, 3)  # 3 prompt tokens + 1 generated
    path = _write(sink.path / "req-0.safetensors", torch.tensor([5, 6, 7, 9]), hidden)

    activations = sink.read(path, expected_token_ids=prompt, layout=LAYOUT, device="cpu")

    torch.testing.assert_close(activations["a"], hidden[:3, 0])
    torch.testing.assert_close(activations["b"], hidden[:3, 1])
    torch.testing.assert_close(activations["c"], hidden[:3, 1])
    assert not path.exists()  # consumed


def test_cache_lifecycle_keeps_files(tmp_path):
    sink = _sink(tmp_path, lifecycle="cache")
    path = _write(sink.path / "req-0.safetensors", torch.tensor([1, 2]), torch.randn(2, 2, 3))

    sink.read(path, expected_token_ids=torch.tensor([1, 2]), layout=LAYOUT, device="cpu")

    assert path.exists()


def test_token_mismatch_or_wrong_layer_count_is_rejected(tmp_path):
    sink = _sink(tmp_path)
    path = _write(sink.path / "req-0.safetensors", torch.tensor([1, 2]), torch.randn(2, 2, 3))
    with pytest.raises(ValueError, match="token_ids"):
        sink.read(path, expected_token_ids=torch.tensor([1, 3]), layout=LAYOUT, device="cpu")

    path = _write(sink.path / "req-1.safetensors", torch.tensor([1, 2]), torch.randn(2, 3, 3))
    with pytest.raises(ValueError, match="2 layers"):
        sink.read(path, expected_token_ids=torch.tensor([1, 2]), layout=LAYOUT, device="cpu")


def test_cleanup_removes_leftovers_of_a_consuming_run(tmp_path):
    sink = _sink(tmp_path)
    _write(sink.path / "req-0.safetensors", torch.tensor([1]), torch.randn(1, 2, 3))

    sink.cleanup()

    assert not sink.path.exists()


def test_cleanup_keeps_cached_files(tmp_path):
    sink = _sink(tmp_path, lifecycle="cache")
    path = _write(sink.path / "req-0.safetensors", torch.tensor([1]), torch.randn(1, 2, 3))

    sink.cleanup()

    assert path.exists()
