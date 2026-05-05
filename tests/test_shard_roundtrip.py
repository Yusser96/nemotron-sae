"""Round-trip activation shards through writer + reader on synthetic data."""

import tempfile
from pathlib import Path

import torch

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.cache.reader import ActivationBuffer, ShardReader
from sae_pipeline.cache.writer import ShardWriter


def test_shard_roundtrip_basic(tmp_path: Path):
    d = 64
    n_tokens = 5000
    rng = torch.Generator().manual_seed(0)
    activations = torch.randn(n_tokens, d, generator=rng, dtype=torch.float32)

    out = tmp_path / "cache"
    manifest = CacheManifest(
        run_id="t", model="m", dtype="float32", layer=0,
        component="x", d_activation=d,
    )
    writer = ShardWriter(
        out_dir=out, d_activation=d,
        shard_size_bytes=64 * 1024,    # ~256 tokens/shard at f32
        dtype=torch.float32, manifest=manifest,
    )
    writer.add(activations)
    final = writer.close()

    assert final.n_shards >= 2
    assert final.total_tokens == n_tokens

    # Read back and check shape + content (sum invariant under shuffle).
    reader = ShardReader(out)
    seen_tokens = 0
    seen_sum = torch.zeros(d, dtype=torch.float32)
    for shard in reader.iter_shards():
        assert shard.shape[1] == d
        seen_tokens += shard.shape[0]
        seen_sum += shard.sum(dim=0)

    assert seen_tokens == n_tokens
    torch.testing.assert_close(seen_sum, activations.sum(dim=0), atol=1e-3, rtol=1e-3)


def test_activation_buffer_yields_correct_shape(tmp_path: Path):
    d = 32
    n_tokens = 4096
    activations = torch.randn(n_tokens, d, dtype=torch.float32)

    manifest = CacheManifest(
        run_id="t", model="m", dtype="float32", layer=0,
        component="x", d_activation=d,
    )
    writer = ShardWriter(
        out_dir=tmp_path, d_activation=d,
        shard_size_bytes=4096,    # ~32 tokens/shard
        dtype=torch.float32, manifest=manifest,
    )
    writer.add(activations)
    writer.close()

    buf = ActivationBuffer(cache_dir=tmp_path, batch_size=128, n_batches_in_buffer=4)
    for _ in range(20):
        b = buf.next_batch()
        assert b.shape == (128, d)
