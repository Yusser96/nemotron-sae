import hashlib
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from sae_pipeline.extraction.rolling import RollingPool
from sae_pipeline.extraction.rolling_reader import RollingCacheStream


def digest(t):
    return hashlib.sha256(
        memoryview(t.contiguous().view(torch.uint8).numpy())
    ).hexdigest()


def test_per_site_read_checksum_and_durable_reclamation(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    pool = RollingPool.create(
        tmp_path / "pool",
        cap_bytes=10000,
        consumers={"S": {"checkpoint_dir": tmp_path / "ckpt"}},
        owned_roots=[cache],
        frame_tokens=8,
    )
    directory = cache / "wave"
    directory.mkdir()
    ids = torch.arange(8).reshape(4, 2).long()
    value = torch.arange(8).reshape(8, 1).float()
    save_file({"token_ids": ids}, str(directory / "token_ids.safetensors"))
    save_file({"S": value}, str(directory / "S.safetensors"))
    entry = dict(
        format="per_site_v2",
        path=str(directory),
        token_ids_path=str(directory / "token_ids.safetensors"),
        site_paths={"S": str(directory / "S.safetensors")},
        start_frame=0,
        frame_count=1,
        site_sha256={"S": digest(value)},
        token_ids_sha256=digest(ids),
        finite_validation="device",
        bytes=sum(p.stat().st_size for p in directory.iterdir()),
    )
    key = pool.reserve(0, 1, entry["bytes"])
    pool.publish(key, entry)
    stream = RollingCacheStream(pool.root, ["S"], "cpu")
    try:
        tokens = stream.next_batch(4)
        assert torch.equal(stream.extract(tokens).activations["S"], value)
        save_file({"S": value + 1}, str(directory / "S.safetensors"))
        with pytest.raises(RuntimeError, match="checksum"):
            stream._load(entry)
    finally:
        stream.close()
    state = pool.read()
    state["consumers"]["S"]["frame"] = 1
    pool._gc(state)
    assert not directory.exists()


def test_metadata_retry_preserves_the_real_state(tmp_path, monkeypatch):
    pool = RollingPool.create(
        tmp_path / "pool",
        cap_bytes=100,
        consumers={"S": {"checkpoint_dir": tmp_path / "c"}},
        owned_roots=[tmp_path],
    )
    expected = pool.read()
    read = Path.read_text
    count = [0]

    def transient(path, *args, **kwargs):
        if path == pool.path and count[0] < 2:
            count[0] += 1
            raise FileNotFoundError(path)
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", transient)
    assert pool.read() == expected and count[0] == 2


def test_missing_metadata_does_not_reset_after_retry_deadline(tmp_path, monkeypatch):
    pool = RollingPool(tmp_path)
    ticks = iter([0.0, 5.1])
    monkeypatch.setattr(
        "sae_pipeline.extraction.rolling.time.monotonic", lambda: next(ticks)
    )
    with pytest.raises(FileNotFoundError):
        pool.read()
    assert not pool.path.exists()
