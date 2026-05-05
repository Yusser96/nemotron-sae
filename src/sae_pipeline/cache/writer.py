"""Write activation shards to disk in safetensors format.

Each shard is a single safetensors file with two keys:
  - "x": (n_tokens_in_shard, d_activation) tensor (BF16 by default)
  - "token_index" (optional): int64 indices for sparse-occupancy components like experts

Shards are shuffled internally on close so train-time iteration is sequential.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from sae_pipeline.cache.manifest import CacheManifest

log = logging.getLogger(__name__)


class ShardWriter:
    """Buffer activations in CPU memory; flush a shard when buffer crosses size threshold."""

    def __init__(
        self,
        out_dir: Path | str,
        d_activation: int,
        shard_size_bytes: int = 1 << 30,
        dtype: torch.dtype = torch.bfloat16,
        shuffle_seed: int = 42,
        manifest: CacheManifest | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.d_activation = d_activation
        self.shard_size_bytes = shard_size_bytes
        self.dtype = dtype
        self.rng = np.random.default_rng(shuffle_seed)
        self.manifest = manifest or CacheManifest(
            run_id="unknown", model="unknown", dtype=str(dtype).removeprefix("torch."),
            layer=-1, component="unknown", d_activation=d_activation,
            shuffle_seed=shuffle_seed,
        )
        self._buf: list[torch.Tensor] = []
        self._buf_tokens = 0
        self._element_size = torch.tensor([], dtype=dtype).element_size()

    @property
    def _bytes_per_token(self) -> int:
        return self.d_activation * self._element_size

    def add(self, x: torch.Tensor) -> None:
        """Append a (n_tokens, d) chunk to the buffer; flush if it crosses the threshold."""
        if x.numel() == 0:
            return
        if x.ndim != 2 or x.shape[1] != self.d_activation:
            raise ValueError(
                f"Expected (n, {self.d_activation}); got {tuple(x.shape)}"
            )
        x = x.to(self.dtype).contiguous()
        self._buf.append(x)
        self._buf_tokens += x.shape[0]
        while self._buf_tokens * self._bytes_per_token >= self.shard_size_bytes:
            self._flush(target_tokens=self.shard_size_bytes // self._bytes_per_token)

    def _flush(self, target_tokens: int | None = None) -> None:
        if self._buf_tokens == 0:
            return
        all_x = torch.cat(self._buf, dim=0)
        self._buf.clear()
        self._buf_tokens = 0

        if target_tokens is not None and all_x.shape[0] > target_tokens:
            head, tail = all_x[:target_tokens], all_x[target_tokens:]
            self._buf.append(tail)
            self._buf_tokens = tail.shape[0]
            all_x = head

        # Shuffle within-shard so SAE training can iterate sequentially.
        perm = torch.from_numpy(self.rng.permutation(all_x.shape[0]))
        all_x = all_x[perm]

        n = self.manifest.n_shards
        shard_path = self.out_dir / f"shard_{n:05d}.safetensors"
        save_file({"x": all_x.contiguous()}, str(shard_path))
        self.manifest.shard_paths.append(shard_path.name)
        self.manifest.n_shards += 1
        self.manifest.total_tokens += all_x.shape[0]
        log.info("Wrote %s (%d tokens)", shard_path.name, all_x.shape[0])

    def close(self) -> CacheManifest:
        # Final flush even if below threshold.
        self._flush(target_tokens=None)
        manifest_path = self.out_dir / "manifest.json"
        self.manifest.write(manifest_path)
        log.info("Wrote manifest: %s (n_shards=%d, total_tokens=%d)",
                 manifest_path, self.manifest.n_shards, self.manifest.total_tokens)
        return self.manifest

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()
