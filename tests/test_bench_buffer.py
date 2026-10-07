import threading

import pytest

from sae_pipeline.cli import bench_buffer
from sae_pipeline.cli.bench_buffer import read_chunks, write_chunks


def test_the_reader_consumes_every_chunk_the_writer_publishes_without_exceeding_the_queue(tmp_path, monkeypatch):
    published = []
    rename = bench_buffer.Path.rename

    def record(path, target):
        published.append(len(list(tmp_path.glob("chunk_*"))))
        return rename(path, target)

    monkeypatch.setattr(bench_buffer.Path, "rename", record)
    results = {}
    reader = threading.Thread(target=lambda: results.update(read=read_chunks(tmp_path, 6, "cpu")))
    reader.start()

    results["write"] = write_chunks(tmp_path, [3, 5], tokens=4, chunks=6, queue=2)
    reader.join(timeout=60)

    assert not reader.is_alive() and list(tmp_path.iterdir()) == []
    assert max(published) < 2
    for role in ("write", "read"):
        assert results[role]["chunks"] == 6 and results[role]["tokens"] == 24
        assert results[role]["gb_per_s"] == pytest.approx(results[role]["tokens_per_s"] * (3 + 5) * 2 / 1e9)
