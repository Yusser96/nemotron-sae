"""The fill directory shared by a producer and its consumers."""

from pathlib import Path

import pytest

from sae_pipeline.extraction.fills import FillQueue, fill_dir, mark_position, published_fills, write_header


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


def test_the_producer_stops_waiting_at_its_deadline(tmp_path):
    _publish(tmp_path, 0, nbytes=100)
    assert not FillQueue(tmp_path, ["a"], poll_seconds=0).wait_for_room(50, deadline=0.0)


def test_a_producer_with_other_settings_cannot_add_to_the_fills(tmp_path):
    write_header(tmp_path, {"model": "m", "seq_len": 4})
    write_header(tmp_path, {"model": "m", "seq_len": 4})
    with pytest.raises(ValueError, match="holds fills"):
        write_header(tmp_path, {"model": "m", "seq_len": 8})
