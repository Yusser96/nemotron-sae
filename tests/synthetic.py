"""Synthetic activation caches for trainer and evaluator tests."""

from pathlib import Path

import torch

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.cache.writer import ShardWriter


def write_cache(
    path: Path,
    x: torch.Tensor,
    *,
    language: str | None = None,
    run_id: str = "synthetic",
    shard_size_bytes: int = 4096,
    shuffle_seed: int = 42,
) -> Path:
    """Write ``x`` as a complete float32 cache: a training cache, or a validation one for ``language``."""
    manifest = CacheManifest(
        run_id=run_id,
        model="synthetic",
        dtype="float32",
        layer=0,
        component="resid_post",
        d_activation=x.shape[1],
        partition="validation" if language else "train",
        language=language,
    )
    with ShardWriter(
        out_dir=path,
        d_activation=x.shape[1],
        shard_size_bytes=shard_size_bytes,
        dtype=torch.float32,
        shuffle_seed=shuffle_seed,
        manifest=manifest,
    ) as writer:
        writer.add(x)
    writer.manifest.complete = True
    writer.manifest.write(path / "manifest.json")
    return path
