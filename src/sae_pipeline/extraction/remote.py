"""Activations streamed GPU to GPU from a separate producer process.

The SAE trainer and a vLLM producer (``cli.serve_activations``) run as two
processes on one node. The trainer hosts a TCPStore, which carries the
control messages: the sites it expects, the producer's ``sequences_per_call``
and every token batch (an empty batch means stop). The store also sets up a
two-rank group (NCCL on GPUs) that carries the activations: for each
``generate`` call the producer's vLLM worker sends every site's rows, in site
order, straight into the trainer's receive buffers.
"""

from __future__ import annotations

import io
import json
from datetime import timedelta

import torch
import torch.distributed as dist

from sae_pipeline.extraction.base import ActivationBatch

TRAINER_RANK, PRODUCER_RANK = 0, 1
# The producer idles on the store while the trainer validates and checkpoints.
CONTROL_TIMEOUT = timedelta(hours=6)
SITES_KEY = "trainer/sites"
CALL_SIZE_KEY = "producer/sequences_per_call"


def open_store(host: str, port: int, *, trainer: bool) -> dist.TCPStore:
    # A few small messages per batch: the classic backend, built on every platform, is plenty.
    return dist.TCPStore(
        host, port, is_master=trainer, timeout=CONTROL_TIMEOUT, wait_for_workers=False, use_libuv=False
    )


def peer_group(store: dist.Store, rank: int, device: torch.device) -> dist.ProcessGroup:
    """The two-rank group for activations on ``device``: NCCL on GPUs, Gloo on CPU."""
    store = dist.PrefixStore("activations", store)
    if device.type == "cuda":
        return dist.ProcessGroupNCCL(store, rank, 2)
    return dist.ProcessGroupGloo(store, rank, 2)


def put_tokens(store: dist.Store, index: int, token_ids: torch.Tensor | None) -> None:
    payload = b""
    if token_ids is not None:
        buffer = io.BytesIO()
        torch.save(token_ids.detach().cpu(), buffer)
        payload = buffer.getvalue()
    store.set(f"tokens/{index}", payload)


def take_tokens(store: dist.Store, index: int) -> torch.Tensor | None:
    """Batch ``index`` once the trainer has put it; None is the stop signal."""
    key = f"tokens/{index}"
    payload = store.get(key)
    if not payload:
        # The trainer may already be gone: no further store calls.
        return None
    store.delete_key(key)
    return torch.load(io.BytesIO(payload))


class RemoteExtractor:
    """Trainer side: each ``extract`` call is answered by the producer process.

    ``widths`` maps every site slug, in the producer's site order, to its
    activation width; rows arrive as ``dtype`` on ``device``.
    """

    def __init__(
        self, *, host: str, port: int, widths: dict[str, int], dtype: torch.dtype, device: str | torch.device
    ) -> None:
        self.widths, self.dtype, self.device = widths, dtype, torch.device(device)
        self.store = open_store(host, port, trainer=True)
        self.store.set(SITES_KEY, json.dumps({"slugs": list(widths), "widths": list(widths.values()), "dtype": str(dtype)}))
        self.group: dist.ProcessGroup | None = None
        self.batches = 0

    def extract(self, token_ids: torch.Tensor) -> ActivationBatch:
        if self.group is None:
            # Blocks until the producer has started its engine.
            self.sequences_per_call = int(self.store.get(CALL_SIZE_KEY))
            self.group = peer_group(self.store, TRAINER_RANK, self.device)
        put_tokens(self.store, self.batches, token_ids)
        self.batches += 1
        sequences, seq_len = token_ids.shape
        activations = {
            slug: torch.empty(sequences * seq_len, width, dtype=self.dtype, device=self.device)
            for slug, width in self.widths.items()
        }
        receives = []
        for start in range(0, sequences, self.sequences_per_call):
            rows = slice(start * seq_len, min(start + self.sequences_per_call, sequences) * seq_len)
            for slug in self.widths:
                receives.append(self.group.recv([activations[slug][rows]], PRODUCER_RANK, 0))
        for receive in receives:
            receive.wait()
        return ActivationBatch(token_ids=token_ids, activations=activations)

    def close(self) -> None:
        put_tokens(self.store, self.batches, None)
