import math

import pytest
import torch
from safetensors.torch import save_file

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.sae.normalization import (
    WholeVectorNormalization,
    _cpu_shard_squared_norm,
    load_whole_vector_normalization,
    normalization_path,
    resolve_activation_center,
    resolve_whole_vector_normalization,
    write_normalization,
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


def test_load_whole_vector_normalization_reads_and_checks_dimension(tmp_path):
    stored = WholeVectorNormalization(input_scale=0.25, mean_squared_norm=16.0, n_tokens=10, d_activation=3)
    write_normalization(normalization_path(tmp_path), stored)

    assert load_whole_vector_normalization(tmp_path, d_activation=3) == stored
    with pytest.raises(ValueError, match="d_activation=3"):
        load_whole_vector_normalization(tmp_path, d_activation=4)
    with pytest.raises(FileNotFoundError):
        load_whole_vector_normalization(tmp_path / "missing", d_activation=3)


@pytest.mark.parametrize("change", ["scale", "sample", "recipe"])
def test_center_rejects_changed_preprocessing(tmp_path, change):
    cache = tmp_path / "cache"
    cache.mkdir()
    save_file({"x": torch.tensor([[2., 4.], [4., 8.]])}, str(cache / "x.safetensors"))
    manifest = CacheManifest("run", "model", "float32", 0, "resid_post", 2,
                             total_tokens=2, shard_paths=["x.safetensors"])
    manifest.write(cache / "manifest.json")
    resolve_activation_center(cache, tmp_path / "metadata", sample_tokens=2)
    if change == "recipe":
        manifest.dataset_fingerprint = "changed"
        manifest.write(cache / "manifest.json")
    with pytest.raises(ValueError, match="does not match"):
        resolve_activation_center(cache, tmp_path / "metadata",
                                  sample_tokens=1 if change == "sample" else 2,
                                  input_scale=.5 if change == "scale" else 1.)
