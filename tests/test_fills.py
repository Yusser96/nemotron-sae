"""The fill directory shared by a producer and its consumers."""

import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.extraction.fills import (
    FillQueue,
    FillStream,
    fill_dir,
    mark_done,
    mark_position,
    published_fills,
    write_header,
)


def _publish(root: Path, index: int, nbytes: int = 10) -> None:
    fill = fill_dir(root, index)
    fill.mkdir(parents=True)
    (fill / "site.safetensors").write_bytes(b"x" * nbytes)


def test_only_published_fills_count(tmp_path):
    _publish(tmp_path, 1)
    _publish(tmp_path, 0)
    (tmp_path / "fill_000002.partial").mkdir()

    assert published_fills(tmp_path) == [0, 1]
    assert published_fills(tmp_path / "missing") == []


def test_fills_go_once_every_consumer_moved_past_them_but_the_newest_stays(tmp_path):
    for index in range(4):
        _publish(tmp_path, index)
    queue = FillQueue(tmp_path, ["a", "b"])
    mark_position(tmp_path, "a", 3)

    # "b" has not started: everything is still needed.
    assert queue.release() == 0
    mark_position(tmp_path, "b", 2)
    assert queue.release() == 2 and published_fills(tmp_path) == [2, 3]
    mark_position(tmp_path, "b", 9)
    assert queue.release() == 1 and published_fills(tmp_path) == [3]


def test_the_producer_waits_for_room_until_the_consumers_read_or_finish(tmp_path, monkeypatch):
    for index in range(3):
        _publish(tmp_path, index, nbytes=100)
    queue = FillQueue(tmp_path, ["a"], poll_seconds=0)
    assert queue.buffered_bytes() == 300
    assert queue.wait_for_room(400)

    sleeps = []
    monkeypatch.setattr("time.sleep", lambda seconds: (sleeps.append(seconds), mark_position(tmp_path, "a", 2)))
    assert queue.wait_for_room(250) and len(sleeps) == 1 and published_fills(tmp_path) == [2]

    (tmp_path / "consumers" / "a.done").touch()
    assert not queue.wait_for_room(50) and queue.all_done()


def test_a_full_buffer_waits_only_for_the_fastest_consumer(tmp_path):
    for index in range(4):
        _publish(tmp_path, index, nbytes=100)
    queue = FillQueue(tmp_path, ["fast", "idle"], poll_seconds=0)
    mark_position(tmp_path, "fast", 2)

    # "idle" never started: the fills "fast" has read go, down to the cap; the rest stay.
    assert queue.wait_for_room(250)
    assert published_fills(tmp_path) == [2, 3]
    assert not queue.wait_for_room(150, deadline=0.0)
    assert published_fills(tmp_path) == [2, 3]


def test_a_finished_consumer_holds_no_fill_back(tmp_path):
    for index in range(4):
        _publish(tmp_path, index, nbytes=100)
    queue = FillQueue(tmp_path, ["done", "running"], poll_seconds=0)
    mark_position(tmp_path, "done", 1)
    mark_done(tmp_path, "done")
    mark_position(tmp_path, "running", 3)

    assert queue.release() == 3 and published_fills(tmp_path) == [3]
    assert not queue.all_done()


def test_the_producer_stops_waiting_at_its_deadline(tmp_path):
    _publish(tmp_path, 0, nbytes=100)
    assert not FillQueue(tmp_path, ["a"], poll_seconds=0).wait_for_room(50, deadline=0.0)


def test_a_producer_with_other_settings_cannot_add_to_the_fills(tmp_path):
    write_header(tmp_path, {"model": "m", "seq_len": 4})
    write_header(tmp_path, {"model": "m", "seq_len": 4})
    with pytest.raises(ValueError, match="holds fills"):
        write_header(tmp_path, {"model": "m", "seq_len": 8})


SEQ_LEN = 2
HEADER = {"model": "m", "seq_len": SEQ_LEN}


def _write_fill(root: Path, index: int, sequences: int, first_token: int) -> None:
    """A fill whose rows are their token ids, so batches can be checked against them."""
    token_ids = torch.arange(first_token, first_token + sequences * SEQ_LEN).reshape(sequences, SEQ_LEN)
    partial = root / f"fill_{index:06d}.partial"
    partial.mkdir(parents=True)
    save_file({"token_ids": token_ids}, str(partial / "token_ids.safetensors"))
    for slug, scale in [("a", 1.0), ("b", -1.0)]:
        rows = scale * token_ids.reshape(-1, 1).float().repeat(1, 3)
        save_file({"activations": rows}, str(partial / f"{slug}.safetensors"))
    partial.rename(fill_dir(root, index))


def _stream(root: Path, **kwargs) -> FillStream:
    write_header(root, {**HEADER, "slugs": ["a", "b"]})
    return FillStream(root, "c0", ["a"], header=HEADER, device="cpu", poll_seconds=0, **kwargs)


def test_batches_run_across_fills_with_each_sequences_rows(tmp_path):
    _write_fill(tmp_path, 0, sequences=3, first_token=0)
    _write_fill(tmp_path, 1, sequences=3, first_token=6)
    stream = _stream(tmp_path)

    batches = []
    for _ in range(3):
        token_ids = stream.next_batch(2)
        batches.append(token_ids)
        rows = stream.extract(token_ids).activations
        assert list(rows) == ["a"]
        torch.testing.assert_close(rows["a"][:, 0], token_ids.reshape(-1).float())

    assert torch.cat(batches).reshape(-1).tolist() == list(range(12))
    assert stream.state_dict() == {"fill": 2, "sequence": 0}
    # The consumer has reached fill 1; fill 0 may be deleted.
    assert FillQueue(tmp_path, ["c0"]).position("c0") == 1
    stream.close()


def test_a_resumed_stream_continues_mid_fill_or_after_deleted_fills(tmp_path):
    _write_fill(tmp_path, 0, sequences=3, first_token=0)
    stream = _stream(tmp_path)
    stream.next_batch(2)
    state = stream.state_dict()
    stream.close()

    resumed = _stream(tmp_path)
    resumed.load_state_dict(state)
    assert resumed.next_batch(1).tolist() == [[4, 5]]
    resumed.close()

    _write_fill(tmp_path, 2, sequences=1, first_token=20)
    shutil.rmtree(fill_dir(tmp_path, 0))
    after_deletion = _stream(tmp_path)
    after_deletion.load_state_dict(state)
    assert after_deletion.next_batch(1).tolist() == [[20, 21]]
    after_deletion.close()


def test_a_consumer_waiting_past_its_deadline_times_out(tmp_path):
    _write_fill(tmp_path, 0, sequences=1, first_token=0)
    stream = _stream(tmp_path, deadline=0.0)
    stream.next_batch(1)
    with pytest.raises(TimeoutError):
        stream.next_batch(1)
    stream.close()


def test_a_consumer_refuses_fills_of_another_model_or_without_its_sites(tmp_path):
    write_header(tmp_path, {**HEADER, "slugs": ["a"]})
    with pytest.raises(ValueError, match="holds fills"):
        FillStream(tmp_path, "c0", ["b"], header=HEADER, device="cpu")
    with pytest.raises(ValueError, match="holds fills"):
        FillStream(tmp_path, "c0", ["a"], header={**HEADER, "model": "other"}, device="cpu")


def test_a_fill_deleted_while_being_read_is_skipped(tmp_path, monkeypatch):
    for index in range(3):
        _write_fill(tmp_path, index, sequences=1, first_token=10 * index)
    read = FillStream._read

    def deleted_first(self, index):
        if index == 1:
            shutil.rmtree(fill_dir(tmp_path, 1), ignore_errors=True)
        return read(self, index)

    monkeypatch.setattr(FillStream, "_read", deleted_first)
    stream = _stream(tmp_path)

    assert stream.next_batch(1).tolist() == [[0, 1]]
    assert stream.next_batch(1).tolist() == [[20, 21]]
    stream.close()
