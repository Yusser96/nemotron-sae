"""The interface every activation-extraction backend implements.

An extractor turns a batch of token ids into activations for a fixed set of
sites, keyed by site slug. Consumers (SAE trainers, caching, benchmarks) do
not need to know which inference engine produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class ActivationBatch:
    token_ids: torch.Tensor  # (sequences, seq_len)
    activations: dict[str, torch.Tensor]  # slug -> (sequences * seq_len, d_in), sequence-major


class ActivationExtractor(Protocol):
    def extract(self, token_ids: torch.Tensor) -> ActivationBatch:
        """Run the model on ``token_ids`` and return every configured site's activations."""
        ...

    def close(self) -> None:
        """Release the engine and any temporary storage."""
        ...
