"""Stream activations from a cache directory for SAE training.

Uses safetensors `safe_open` to memory-map each shard; we never materialize the full
cache. Yields fixed-size batches.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open

from sae_pipeline.cache.manifest import CacheManifest

log = logging.getLogger(__name__)


class ShardReader:
    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = CacheManifest.read(self.cache_dir / "manifest.json")

    def iter_shards(self) -> Iterator[torch.Tensor]:
        for name in self.manifest.shard_paths:
            path = self.cache_dir / name
            with safe_open(str(path), framework="pt", device="cpu") as f:
                yield f.get_tensor("x")

    def iter_batches(self, batch_size: int, infinite: bool = True) -> Iterator[torch.Tensor]:
        """Yield (batch_size, d) batches by concatenating across shards as needed.

        Robust to shards smaller than batch_size: we buffer leftovers and stitch them
        with the next shard. Drops the final partial batch each pass.
        """
        leftover: torch.Tensor | None = None
        while True:
            for shard in self.iter_shards():
                buf = shard if leftover is None else torch.cat([leftover, shard], dim=0)
                leftover = None
                n = buf.shape[0]
                i = 0
                while i + batch_size <= n:
                    yield buf[i : i + batch_size]
                    i += batch_size
                if i < n:
                    leftover = buf[i:].contiguous()
            if not infinite:
                return
            # New pass: keep leftover bridging into the next iteration of shards.


class ActivationBuffer:
    """Rolling buffer that pre-fetches several shards and serves random-mini-batches.

    Mirrors sae_lens / dictionary_learning's ActivationsStore: keep
    `n_batches_in_buffer * batch_size` activations, refill from the underlying
    iterator when half-empty.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        batch_size: int,
        n_batches_in_buffer: int = 8,
    ) -> None:
        self.batch_size = batch_size
        self.n_batches_in_buffer = n_batches_in_buffer
        self._reader = ShardReader(cache_dir)
        self._underlying = self._reader.iter_batches(
            batch_size=batch_size, infinite=True
        )
        self._buf: torch.Tensor | None = None
        self._cursor = 0
        self.d_activation = self._reader.manifest.d_activation

    def _capacity(self) -> int:
        return self.batch_size * self.n_batches_in_buffer

    def _refill(self) -> None:
        new_chunks: list[torch.Tensor] = []
        # Keep the unread tail of the existing buffer.
        if self._buf is not None and self._cursor < self._buf.shape[0]:
            new_chunks.append(self._buf[self._cursor :])
        while sum(c.shape[0] for c in new_chunks) < self._capacity():
            new_chunks.append(next(self._underlying))
        merged = torch.cat(new_chunks, dim=0)
        # Shuffle within the buffer.
        perm = torch.randperm(merged.shape[0])
        self._buf = merged[perm]
        self._cursor = 0

    def next_batch(self) -> torch.Tensor:
        if (
            self._buf is None
            or self._cursor + self.batch_size > self._buf.shape[0]
            or self._cursor >= self._buf.shape[0] // 2
        ):
            self._refill()
        assert self._buf is not None
        b = self._buf[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size
        return b
