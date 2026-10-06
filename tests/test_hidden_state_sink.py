import sys
import threading
import time
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


def test_consume_also_removes_the_writer_lock_file(tmp_path):
    sink = _sink(tmp_path)
    path = _write(sink.path / "req-0.safetensors", torch.tensor([1, 2]), torch.randn(2, 2, 3))
    lock = sink.path / "req-0.safetensors.lock"
    lock.touch()

    sink.read(path, expected_token_ids=torch.tensor([1, 2]), layout=LAYOUT, device="cpu")

    assert not path.exists() and not lock.exists()


def test_cleanup_removes_leftover_lock_files(tmp_path):
    sink = _sink(tmp_path)
    (sink.path / "req-9.safetensors.lock").touch()

    sink.cleanup()

    assert not sink.path.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX-only")
def test_read_waits_until_the_writer_releases_its_lock(tmp_path):
    import fcntl

    sink = _sink(tmp_path)
    path = sink.path / "req-0.safetensors"
    lock = open(sink.path / "req-0.safetensors.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)

    def write_then_release():
        time.sleep(0.3)
        _write(path, torch.tensor([1, 2]), torch.randn(2, 2, 3))
        lock.close()

    writer = threading.Thread(target=write_then_release)
    writer.start()
    activations = sink.read(path, expected_token_ids=torch.tensor([1, 2]), layout=LAYOUT, device="cpu")
    writer.join()

    assert activations["a"].shape == (2, 3)


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


def test_collect_reports_prompts_that_were_never_captured(tmp_path):
    sink = _sink(tmp_path)
    step = _step(sink.path / "step_0.safetensors", [1, 2], [0, 1], a=torch.zeros(2, 2))

    with pytest.raises(RuntimeError, match=r"sequences \[1\]"):
        sink.collect([step], torch.tensor([[1, 2], [7, 8]]), ["a"], device="cpu")
