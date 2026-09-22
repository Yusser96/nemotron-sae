"""Per-site activation normalisation for JumpReLU SAE training."""

from __future__ import annotations

import json
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from sae_pipeline.cache.manifest import CacheManifest
from sae_pipeline.cache.reader import ShardReader


@dataclass(frozen=True)
class WholeVectorNormalization:
    """The Gemma Scope whole-vector normalisation statistic for one cache."""

    input_scale: float
    mean_squared_norm: float
    n_tokens: int
    d_activation: int


@dataclass(frozen=True)
class ActivationCenter:
    """A bounded per-site mean used by the optional centring intervention."""

    mean: list[float]
    n_tokens: int
    d_activation: int
    sample_tokens: int


def normalization_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "normalization.json"


def _read(path: Path) -> WholeVectorNormalization:
    with path.open() as handle:
        return WholeVectorNormalization(**json.load(handle))


def _write_atomic(path: Path, value: WholeVectorNormalization) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(asdict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_normalization(path: str | Path, value: WholeVectorNormalization) -> None:
    """Persist an already verified normalisation statistic without recomputing it."""
    _write_atomic(Path(path), value)


def _cpu_shard_squared_norm(path: Path) -> tuple[float, int]:
    """Return a shard's squared-norm sum and row count using one CPU thread."""

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        activations = handle.get_tensor("x")
    squared_norm = activations.float().square().sum(dtype=torch.float64).item()
    return float(squared_norm), int(activations.shape[0])


def centering_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "centering.json"


def resolve_activation_center(
    cache_dir: str | Path,
    out_dir: str | Path,
    *,
    sample_tokens: int,
    input_scale: float = 1.0,
) -> ActivationCenter:
    """Estimate and cache a deterministic activation mean from a cache prefix."""

    cache_dir = Path(cache_dir)
    manifest = CacheManifest.read(cache_dir / "manifest.json")
    path = centering_path(out_dir)
    if path.exists():
        with path.open() as handle:
            saved = ActivationCenter(**json.load(handle))
        if saved.d_activation != manifest.d_activation:
            raise ValueError(f"Centring metadata at {path} has the wrong dimension")
        return saved

    total = torch.zeros(manifest.d_activation, dtype=torch.float64)
    seen = 0
    limit = min(sample_tokens, manifest.total_tokens)
    for shard in ShardReader(cache_dir).iter_shards():
        if seen >= limit:
            break
        part = shard[: limit - seen].to(dtype=torch.float32)
        total += (part * input_scale).sum(dim=0, dtype=torch.float64)
        seen += part.shape[0]
    if seen <= 0:
        raise ValueError(f"Cannot estimate centring mean from empty cache {cache_dir}")
    result = ActivationCenter(
        mean=(total / seen).to(dtype=torch.float32).tolist(),
        n_tokens=manifest.total_tokens,
        d_activation=manifest.d_activation,
        sample_tokens=seen,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(asdict(result), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def resolve_whole_vector_normalization(
    cache_dir: str | Path,
    out_dir: str | Path,
    *,
    device: str = "cpu",
    workers: int = 1,
) -> WholeVectorNormalization:
    """Load or calculate ``1 / sqrt(E[||x||²])`` over an immutable cache.

    The statistic is cached beside canonical checkpoints.  A stale statistic is
    rejected rather than silently being used for a cache with different shape
    or token count.
    """

    cache_dir = Path(cache_dir)
    manifest = CacheManifest.read(cache_dir / "manifest.json")
    path = normalization_path(out_dir)
    if path.exists():
        result = _read(path)
        if (result.n_tokens, result.d_activation) != (
            manifest.total_tokens,
            manifest.d_activation,
        ):
            raise ValueError(
                f"Normalisation at {path} does not match cache {cache_dir}: "
                f"saved ({result.n_tokens}, {result.d_activation}), cache "
                f"({manifest.total_tokens}, {manifest.d_activation})"
            )
        return result

    manifest_shards = [cache_dir / name for name in manifest.shard_paths]
    if device == "cpu" and workers > 1:
        # This is a one-time reduction over a large immutable cache. Parallel
        # CPU readers avoid occupying a GPU while the cache is read from
        # shared storage. Memory is bounded by the number of workers.
        workers = min(int(workers), len(manifest_shards))
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                partials = list(pool.map(_cpu_shard_squared_norm, manifest_shards))
        finally:
            torch.set_num_threads(previous_threads)
        total_squared_norm = math.fsum(value for value, _ in partials)
        n_tokens = sum(count for _, count in partials)
    else:
        total_squared_norm = torch.zeros((), device=device, dtype=torch.float64)
        n_tokens = 0
        for path in manifest_shards:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                activations = handle.get_tensor("x").to(device=device, dtype=torch.float32)
            total_squared_norm += activations.square().sum(dtype=torch.float64)
            n_tokens += activations.shape[0]
        total_squared_norm = total_squared_norm.item()

    if n_tokens != manifest.total_tokens or n_tokens <= 0:
        raise ValueError(
            f"Cache {cache_dir} contains {n_tokens} readable tokens; expected "
            f"{manifest.total_tokens}"
        )
    mean_squared_norm = total_squared_norm / n_tokens
    if not math.isfinite(mean_squared_norm) or mean_squared_norm <= 0:
        raise ValueError(
            f"Invalid activation squared norm {mean_squared_norm} for {cache_dir}"
        )
    result = WholeVectorNormalization(
        input_scale=1.0 / math.sqrt(mean_squared_norm),
        mean_squared_norm=mean_squared_norm,
        n_tokens=n_tokens,
        d_activation=manifest.d_activation,
    )
    _write_atomic(path, result)
    return result
