import math

import torch
from safetensors.torch import save_file

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.sae.normalization import (
    _cpu_shard_squared_norm,
    resolve_whole_vector_normalization,
)


def test_cpu_shard_squared_norm_reduces_in_chunks(tmp_path):
    activations = torch.arange(35, dtype=torch.float32).reshape(7, 5) - 9
    shard = tmp_path / "chunked.safetensors"
    save_file({"x": activations}, str(shard))

    squared_norm, n_tokens = _cpu_shard_squared_norm(shard, chunk_rows=2)

    assert n_tokens == 7
    assert math.isclose(squared_norm, float(activations.square().sum()))


def test_parallel_whole_vector_normalization_matches_all_shards(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    shards = [
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[-1.0, 0.0, 2.0]]),
    ]
    shard_paths = []
    for index, shard in enumerate(shards):
        name = f"shard_{index:05d}.safetensors"
        save_file({"x": shard}, str(cache_dir / name))
        shard_paths.append(name)
    CacheManifest(
        run_id="run",
        model="model",
        dtype="float32",
        layer=0,
        component="resid_post",
        d_activation=3,
        n_shards=2,
        total_tokens=3,
        shard_paths=shard_paths,
        complete=True,
    ).write(cache_dir / "manifest.json")

    result = resolve_whole_vector_normalization(
        cache_dir, tmp_path / "metadata", device="cpu", workers=2
    )

    assert result.n_tokens == 3
    assert result.d_activation == 3
    assert math.isclose(result.mean_squared_norm, 32.0)
    assert math.isclose(result.input_scale, 1.0 / math.sqrt(32.0))
