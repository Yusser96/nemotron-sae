"""Fixed validation probes: deterministic batches, cache identity and probe setup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import torch

from sae_pipeline.cache.reader import ShardReader
from sae_pipeline.config import EarlyStoppingCfg

log = logging.getLogger(__name__)


def validation_batches(
    reader: ShardReader,
    *,
    n_tokens: int,
    batch_size: int,
    input_scale: float,
    input_center: torch.Tensor | None,
    device: str,
) -> Iterable[torch.Tensor]:
    """Yield the same bounded validation prefix deterministically each time."""
    if n_tokens <= 0 or n_tokens > reader.manifest.total_tokens:
        raise ValueError(
            f"Requested {n_tokens} validation vectors from a cache containing "
            f"{reader.manifest.total_tokens}"
        )
    seen = 0
    for shard in reader.iter_shards():
        for start in range(0, shard.shape[0], batch_size):
            if seen >= n_tokens:
                return
            take = min(batch_size, n_tokens - seen)
            # CPU float32 caches may alias the mapped shard after .to().
            batch = shard[start : start + take].to(device, dtype=torch.float32, copy=True)
            if input_scale != 1.0:
                batch.mul_(input_scale)
            if input_center is not None:
                batch.sub_(input_center)
            seen += batch.shape[0]
            yield batch
    if seen != n_tokens:
        raise ValueError(
            f"Validation cache yielded {seen} vectors, expected {n_tokens}"
        )


def probe_cache_signature(readers: dict[str, ShardReader]) -> dict[str, dict[str, object]]:
    """Identify the ordered probe datasets used by validation trajectories."""
    return {
        language: {
            "run_id": reader.manifest.run_id,
            "model": reader.manifest.model,
            "layer": reader.manifest.layer,
            "component": reader.manifest.component,
            "partition": reader.manifest.partition,
            "language": reader.manifest.language,
            "d_activation": reader.manifest.d_activation,
            "total_tokens": reader.manifest.total_tokens,
            "shard_paths": list(reader.manifest.shard_paths),
            "dataset_fingerprint": reader.manifest.dataset_fingerprint,
            "extraction_fingerprint": reader.manifest.extraction_fingerprint,
        }
        for language, reader in sorted(readers.items())
    }


def prepare_validation_probes(
    cache_dir: str | Path,
    d_in: int,
    *,
    validation_cache_dirs: dict[str, str | Path],
    validation_tokens: int | None,
    validation_dead_tokens: int,
    early_stopping: EarlyStoppingCfg,
) -> tuple[dict[str, ShardReader], dict[str, dict[str, object]], ShardReader | None, int]:
    """Validate config and open the validation + fixed train-probe readers.

    Returns (validation_readers, validation_cache_signature, train_probe_reader,
    probe_tokens).
    """
    if early_stopping.enabled:
        if not validation_cache_dirs:
            raise ValueError("Validation-based early stopping requires validation caches")
        if validation_tokens is None or validation_tokens <= 0:
            raise ValueError("Validation-based early stopping requires a positive token count")
        if len(validation_cache_dirs) < 2:
            raise ValueError("Bilingual early stopping requires two language validation caches")
    validation_readers = {
        language: ShardReader(path)
        for language, path in sorted(validation_cache_dirs.items())
    }
    validation_cache_signature = probe_cache_signature(validation_readers)
    train_probe_reader: ShardReader | None = None
    probe_tokens = validation_tokens if validation_tokens is not None else validation_dead_tokens
    if validation_readers:
        if probe_tokens <= 0:
            raise ValueError("Validation trajectories require a positive fixed-probe token count")
        undersized = {
            language: reader.manifest.total_tokens
            for language, reader in validation_readers.items()
            if reader.manifest.total_tokens < probe_tokens
        }
        if undersized:
            raise ValueError(
                f"Validation caches are smaller than the {probe_tokens}-token probe: {undersized}"
            )
        train_probe_reader = ShardReader(cache_dir)
        train_manifest = train_probe_reader.manifest
        if not train_manifest.complete or train_manifest.partition != "train":
            raise ValueError("Training probe must come from a complete training cache")
        if train_manifest.d_activation != d_in:
            raise ValueError("Training probe activation dimension differs from the SAE")
        if train_probe_reader.manifest.total_tokens < probe_tokens:
            raise ValueError(
                f"Training cache has {train_probe_reader.manifest.total_tokens} tokens; "
                f"the fixed probe requires {probe_tokens}"
            )
        for language, reader in validation_readers.items():
            manifest = reader.manifest
            if not manifest.complete or manifest.partition != "validation":
                raise ValueError(f"{language} probe must come from a complete validation cache")
            if manifest.language != language:
                raise ValueError(
                    f"Validation cache keyed as {language!r} declares language "
                    f"{manifest.language!r}"
                )
            if (manifest.model, manifest.layer, manifest.component) != (
                train_manifest.model,
                train_manifest.layer,
                train_manifest.component,
            ):
                raise ValueError(f"{language} validation cache belongs to a different site/model")
            if manifest.d_activation != d_in:
                raise ValueError(f"{language} validation activation dimension differs from the SAE")
        log.info(
            "Initialized fixed %d-token train probe from %s",
            probe_tokens,
            cache_dir,
        )
    return validation_readers, validation_cache_signature, train_probe_reader, probe_tokens
