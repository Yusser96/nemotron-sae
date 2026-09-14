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


class _BatchStream:
    """Serializable cyclic traversal of the cache's concatenated shard rows."""

    def __init__(self, reader: ShardReader, batch_size: int) -> None:
        if not reader.manifest.shard_paths:
            raise ValueError(f"Activation cache {reader.cache_dir} has no shards")
        self.reader = reader
        self.batch_size = batch_size
        self.shard_index = 0
        self.row_offset = 0
        self._shard: torch.Tensor | None = None

    def _current_shard(self) -> torch.Tensor:
        if self._shard is None:
            name = self.reader.manifest.shard_paths[self.shard_index]
            path = self.reader.cache_dir / name
            with safe_open(str(path), framework="pt", device="cpu") as f:
                self._shard = f.get_tensor("x")
            if self._shard.shape[0] == 0:
                raise ValueError(f"Activation shard {path} is empty")
        return self._shard

    def _advance_shard(self) -> None:
        self.shard_index = (
            self.shard_index + 1
        ) % len(self.reader.manifest.shard_paths)
        self.row_offset = 0
        self._shard = None

    def next_batch(self) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        remaining = self.batch_size
        while remaining:
            shard = self._current_shard()
            available = shard.shape[0] - self.row_offset
            take = min(available, remaining)
            chunks.append(shard[self.row_offset : self.row_offset + take])
            self.row_offset += take
            remaining -= take
            if self.row_offset == shard.shape[0]:
                self._advance_shard()
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=0)

    def state_dict(self) -> dict[str, int]:
        return {
            "shard_index": self.shard_index,
            "row_offset": self.row_offset,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        shard_index = int(state["shard_index"])
        row_offset = int(state["row_offset"])
        n_shards = len(self.reader.manifest.shard_paths)
        if not 0 <= shard_index < n_shards:
            raise ValueError(
                f"Invalid activation stream shard index {shard_index} for {n_shards} shards"
            )
        self.shard_index = shard_index
        self.row_offset = row_offset
        self._shard = None
        shard = self._current_shard()
        if not 0 <= row_offset < shard.shape[0]:
            raise ValueError(
                f"Invalid activation stream row offset {row_offset} for shard with "
                f"{shard.shape[0]} rows"
            )


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
        self._underlying = _BatchStream(self._reader, batch_size=batch_size)
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
            new_chunks.append(self._underlying.next_batch())
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

    def state_dict(self) -> dict[str, object]:
        """Return all state required to produce the exact next mini-batch."""

        return {
            "version": 1,
            "batch_size": self.batch_size,
            "n_batches_in_buffer": self.n_batches_in_buffer,
            "d_activation": self.d_activation,
            "cache_manifest": {
                "run_id": self._reader.manifest.run_id,
                "model": self._reader.manifest.model,
                "layer": self._reader.manifest.layer,
                "component": self._reader.manifest.component,
                "total_tokens": self._reader.manifest.total_tokens,
                "shard_paths": list(self._reader.manifest.shard_paths),
                "dataset_fingerprint": self._reader.manifest.dataset_fingerprint,
            },
            "buffer": None if self._buf is None else self._buf.detach().cpu().clone(),
            "cursor": self._cursor,
            "stream": self._underlying.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a state produced by :meth:`state_dict`."""

        for key, expected in (
            ("batch_size", self.batch_size),
            ("n_batches_in_buffer", self.n_batches_in_buffer),
            ("d_activation", self.d_activation),
        ):
            actual = int(state[key])  # type: ignore[arg-type]
            if actual != expected:
                raise ValueError(
                    f"Activation buffer {key} mismatch: checkpoint has {actual}, "
                    f"current run has {expected}"
                )

        saved_manifest = state.get("cache_manifest")
        current_manifest = {
            "run_id": self._reader.manifest.run_id,
            "model": self._reader.manifest.model,
            "layer": self._reader.manifest.layer,
            "component": self._reader.manifest.component,
            "total_tokens": self._reader.manifest.total_tokens,
            "shard_paths": list(self._reader.manifest.shard_paths),
            "dataset_fingerprint": self._reader.manifest.dataset_fingerprint,
        }
        if saved_manifest != current_manifest:
            raise ValueError(
                "Activation cache manifest does not match the checkpoint; exact resume "
                "requires the same cached activations"
            )

        buffer = state["buffer"]
        if buffer is not None:
            if not isinstance(buffer, torch.Tensor) or buffer.ndim != 2:
                raise ValueError("Invalid activation buffer tensor in checkpoint")
            if buffer.shape[1] != self.d_activation:
                raise ValueError(
                    "Activation buffer dictionary input dimension mismatch: "
                    f"checkpoint has {buffer.shape[1]}, current cache has {self.d_activation}"
                )
            self._buf = buffer.detach().cpu().clone()
        else:
            self._buf = None

        cursor = int(state["cursor"])  # type: ignore[arg-type]
        if cursor < 0 or (self._buf is not None and cursor > self._buf.shape[0]):
            raise ValueError(f"Invalid activation buffer cursor {cursor}")
        self._cursor = cursor
        stream = state["stream"]
        if not isinstance(stream, dict):
            raise ValueError("Invalid activation stream state in checkpoint")
        self._underlying.load_state_dict(stream)  # type: ignore[arg-type]
